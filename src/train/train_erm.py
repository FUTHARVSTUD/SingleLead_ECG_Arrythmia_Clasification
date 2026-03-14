from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.augment.ecg_aug import ECGAugmentor
from src.datasets.npz_dataset import NpzDataset
from src.eval.metrics import classification_metrics
from src.models.resnet1d import resnet1d_teacher
from src.models.tinydscnn1d import tinydscnn1d_student
from src.train.train_utils import coral_loss, build_sampler, get_device, save_checkpoint, save_metrics, set_seed


def build_model(name: str, num_classes: int, in_channels: int = 1):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes, in_channels=in_channels)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown model '{name}'")


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    debug=False,
    target_loader=None,
    coral_weight: float = 0.0,
    rehearsal_loader=None,
    rehearsal_ratio: float = 0.1,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_coral = target_loader is not None and coral_weight > 0.0
    if use_coral and not (hasattr(model, "forward_with_features") and hasattr(model, "forward_features")):
        raise RuntimeError("Model must implement forward_with_features and forward_features for CORAL")
    target_iter = iter(target_loader) if use_coral else None
    rehearsal_iter = iter(rehearsal_loader) if rehearsal_loader is not None else None
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if use_coral:
            logits, source_feats = model.forward_with_features(x)
        else:
            logits = model(x)
        loss = criterion(logits, y)
        if use_coral:
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)
            if rehearsal_loader is not None and rehearsal_ratio > 0.0:
                try:
                    rehearse_x, _ = next(rehearsal_iter)
                except StopIteration:
                    rehearsal_iter = iter(rehearsal_loader)
                    rehearse_x, _ = next(rehearsal_iter)
                mix_count = max(1, int(len(target_x) * rehearsal_ratio))
                rehearse_x = rehearse_x[:mix_count]
                target_x = torch.cat([target_x, rehearse_x], dim=0)
            target_x = target_x.to(device)
            target_feats = model.forward_features(target_x)
            loss = loss + coral_weight * coral_loss(source_feats, target_feats)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        total_samples += len(x)
        if debug and step >= 5:
            break
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def evaluate(model, loader, criterion, device, debug=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    targets = []
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
            total_samples += len(x)
            preds.append(torch.argmax(logits, dim=1).cpu())
            targets.append(y.cpu())
            if debug and step >= 5:
                break
    if total_samples == 0:
        return 0.0, {"macro_f1": 0.0}
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = float(total_loss / total_samples)
    metrics["accuracy"] = float((y_pred == y_true).mean())
    return total_loss / total_samples, metrics


def _parse_boost_classes(spec: str):
    if not spec:
        return []
    items = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid class index '{token}' in boost list")
    return items


def _parse_float_list(spec: str):
    values = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _build_class_weights(counts, mode: str, boost_classes=None, boost_factor: float = 1.0):
    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    weights = torch.ones_like(counts_tensor)
    mask = counts_tensor > 0
    if not torch.any(mask):
        return None
    if mode == "balanced":
        weights[mask] = counts_tensor.sum() / (len(counts_tensor) * counts_tensor[mask])
    elif mode == "balanced_sqrt":
        weights[mask] = torch.sqrt(counts_tensor.sum()) / (
            len(counts_tensor) ** 0.5 * torch.sqrt(counts_tensor[mask])
        )
    elif mode != "none":
        raise ValueError(f"Unknown class_weight mode '{mode}'")
    weights[~mask] = 0.0
    if boost_classes:
        for idx in boost_classes:
            if 0 <= idx < len(weights):
                weights[idx] = weights[idx] * boost_factor
    if torch.allclose(weights, torch.ones_like(weights)):
        return None
    mean = weights[mask].mean().clamp_min(1e-6)
    return weights / mean


def run(args: argparse.Namespace):
    set_seed(args.seed)
    if args.augment:
        if args.model == "student_aug":
            augmentor = ECGAugmentor(warmup_epochs=2, mid_epochs=4)
        else:
            augmentor = ECGAugmentor(warmup_epochs=5, mid_epochs=10)
    else:
        augmentor = None
    train_dataset = NpzDataset(args.train_npz, augment=augmentor)
    val_dataset = NpzDataset(args.val_npz)

    sampler = None if args.no_sampler else build_sampler(train_dataset.y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    target_loader = None
    rehearsal_loader = None
    if args.coral_npz:
        coral_dataset = NpzDataset(args.coral_npz)
        coral_batch = args.coral_batch or args.batch
        if coral_batch < 2:
            raise ValueError("CORAL batch size must be >=2")
        target_loader = DataLoader(coral_dataset, batch_size=coral_batch, shuffle=True, drop_last=True)
        if args.coral_rehearsal_npz:
            rehearsal_dataset = NpzDataset(args.coral_rehearsal_npz)
            rehearsal_batch = args.coral_rehearsal_batch or coral_batch
            if rehearsal_batch < 2:
                raise ValueError("Rehearsal batch size must be >=2")
            sampler = None
            if args.coral_rehearsal_weights:
                weight_values = _parse_float_list(args.coral_rehearsal_weights)
                if len(weight_values) not in {0, rehearsal_dataset.num_classes}:
                    raise ValueError("coral_rehearsal_weights must match number of classes")
                class_weights = torch.tensor(weight_values, dtype=torch.double)
                sample_weights = class_weights[rehearsal_dataset.y]
                sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            rehearsal_loader = DataLoader(
                rehearsal_dataset,
                batch_size=rehearsal_batch,
                sampler=sampler,
                shuffle=sampler is None,
                drop_last=True,
            )

    device = get_device()
    model = build_model(args.model, train_dataset.num_classes, train_dataset.in_channels)
    if args.init_ckpt:
        state_dict = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)

    boost_targets = _parse_boost_classes(args.boost_classes)
    class_weights = _build_class_weights(
        train_dataset.class_counts(), args.class_weight, boost_targets, args.boost_factor
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = -float("inf")
    history = []

    for epoch in range(args.epochs):
        if augmentor is not None:
            augmentor.set_epoch(epoch + 1)
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.debug,
            target_loader=target_loader,
            coral_weight=args.coral_weight,
            rehearsal_loader=rehearsal_loader,
            rehearsal_ratio=args.coral_rehearsal_ratio,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.debug)
        metric_value = val_metrics.get("macro_f1", 0.0)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_f1": float(metric_value),
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log))
        if metric_value > best_metric:
            best_metric = metric_value
            save_checkpoint(model, Path(args.out_dir) / "best.pt")
            metrics_path = Path(args.out_dir) / "metrics.json"
            save_metrics({"best_epoch": epoch + 1, **val_metrics}, metrics_path)

    (Path(args.out_dir) / "history.json").write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train ERM baseline for ECG models")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "student_aug"])
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Initialize model from checkpoint")
    parser.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=["none", "balanced", "balanced_sqrt"],
        help="Apply class weighting in the loss to emphasize rare beats",
    )
    parser.add_argument(
        "--boost_classes",
        type=str,
        default="",
        help="Comma-separated class indices to upweight (e.g. '3,4' for F/Q)",
    )
    parser.add_argument(
        "--boost_factor",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to boost_classes weights",
    )
    parser.add_argument("--coral_npz", type=str, default=None, help="NPZ path with unlabeled target data for CORAL")
    parser.add_argument("--coral_weight", type=float, default=0.0, help="Weight for CORAL loss")
    parser.add_argument("--coral_batch", type=int, default=None, help="Batch size for CORAL target loader")
    parser.add_argument(
        "--coral_rehearsal_npz",
        type=str,
        default=None,
        help="Optional NPZ with source-domain rehearsal samples to mix into CORAL batches",
    )
    parser.add_argument(
        "--coral_rehearsal_batch",
        type=int,
        default=None,
        help="Batch size for the rehearsal loader (defaults to coral batch)",
    )
    parser.add_argument(
        "--coral_rehearsal_ratio",
        type=float,
        default=0.25,
        help="Fraction of the CORAL batch replaced by rehearsal samples",
    )
    parser.add_argument(
        "--coral_rehearsal_weights",
        type=str,
        default="0.25,0.25,0.25,0.125,0.125",
        help="Comma-separated class weights for sampling rehearsal examples",
    )
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

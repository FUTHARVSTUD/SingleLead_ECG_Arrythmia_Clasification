from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.augment.ecg_aug import ECGAugmentor
from src.datasets.npz_dataset import NpzDataset
from src.eval.metrics import classification_metrics
from src.models.resnet1d import resnet1d_teacher
from src.models.tinydscnn1d import tinydscnn1d_student
from src.train.train_utils import (
    build_sampler,
    compute_class_weights,
    get_device,
    save_checkpoint,
    save_metrics,
    set_seed,
)


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=0.0, debug=False):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
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


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise ValueError(f"Unknown scheduler '{name}'")


def run(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device()
    augmentor = ECGAugmentor() if args.augment else None
    train_dataset = NpzDataset(args.train_npz, augment=augmentor)
    val_dataset = NpzDataset(args.val_npz)

    sampler = None if args.no_sampler else build_sampler(train_dataset.y)
    use_class_weights = not args.no_class_weights and (
        sampler is None or args.class_weights_with_sampler
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(args.model, train_dataset.num_classes)
    model.to(device)

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset.y).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

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
            grad_clip=args.grad_clip,
            debug=args.debug,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.debug)
        metric_value = val_metrics.get("macro_f1", 0.0)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_f1": float(metric_value),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log))
        if metric_value > best_metric:
            best_metric = metric_value
            save_checkpoint(model, Path(args.out_dir) / "best.pt")
            metrics_path = Path(args.out_dir) / "metrics.json"
            save_metrics({"best_epoch": epoch + 1, **val_metrics}, metrics_path)
        if scheduler is not None:
            scheduler.step()

    (Path(args.out_dir) / "history.json").write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train ERM baseline for ECG models")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "student_aug"])
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler")
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable inverse-frequency loss weighting",
    )
    parser.add_argument(
        "--class_weights_with_sampler",
        action="store_true",
        help="Keep class weights even when using WeightedRandomSampler",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

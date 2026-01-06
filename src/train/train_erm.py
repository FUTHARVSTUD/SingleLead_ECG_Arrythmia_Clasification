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
from src.train.train_utils import build_sampler, get_device, save_checkpoint, save_metrics, set_seed


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")


def train_epoch(model, loader, criterion, optimizer, device, debug=False):
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


def run(args: argparse.Namespace):
    set_seed(args.seed)
    augmentor = ECGAugmentor() if args.augment else None
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

    device = get_device()
    model = build_model(args.model, train_dataset.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = -float("inf")
    history = []

    for epoch in range(args.epochs):
        if augmentor is not None:
            augmentor.set_epoch(epoch + 1)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.debug)
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
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

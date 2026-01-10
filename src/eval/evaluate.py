from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.npz_dataset import NpzDataset
from src.eval.metrics import classification_metrics
from src.models.resnet1d import resnet1d_teacher
from src.models.tinydscnn1d import tinydscnn1d_student
from src.train.train_utils import get_device, save_metrics


def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")


def adapt_batchnorm(model: torch.nn.Module, npz_path: str, batch_size: int, device: torch.device, steps: int):
    adapt_dataset = NpzDataset(npz_path)
    loader = DataLoader(adapt_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    with torch.no_grad():
        for step, (batch_x, _) in enumerate(loader):
            batch_x = batch_x.to(device)
            model(batch_x)
            if steps > 0 and (step + 1) >= steps:
                break
    model.eval()


def evaluate(args: argparse.Namespace):
    dataset = NpzDataset(args.npz)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    device = get_device()
    model = build_model(args.model, dataset.num_classes)
    ckpt_path = Path(args.ckpt)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.adapt_npz:
        adapt_batchnorm(model, args.adapt_npz, args.batch, device, args.adapt_steps)

    preds = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.append(pred)
            targets.append(batch_y)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    metrics = classification_metrics(y_true, y_pred)
    metrics["accuracy"] = float((y_pred == y_true).mean())
    metrics["bn_adapted"] = bool(args.adapt_npz)
    if args.out_json:
        save_metrics(metrics, Path(args.out_json))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on an NPZ split")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "student_aug"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--adapt_npz", type=str, default=None, help="Optional NPZ split for AdaBN")
    parser.add_argument("--adapt_steps", type=int, default=-1, help="Limit AdaBN batches (-1 for all)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

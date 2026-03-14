from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.npz_dataset import NpzDataset
from src.models.resnet1d import resnet1d_teacher
from src.models.tinydscnn1d import tinydscnn1d_student
from src.train.train_utils import get_device, save_checkpoint


def build_model(name: str, num_classes: int, in_channels: int = 1):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes, in_channels=in_channels)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"Unknown model '{name}'")


def adapt_batchnorm(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run AdaBN by recomputing running statistics on target-domain data."""
    model.train()
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            model(batch_x)
    model.eval()


def main():
    parser = argparse.ArgumentParser(description="Adapt BatchNorm statistics using unlabeled NPZ data")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "student_aug"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to source-domain checkpoint")
    parser.add_argument("--npz", type=str, required=True, help="Target-domain NPZ used for BN adaptation")
    parser.add_argument("--out_ckpt", type=str, required=True, help="Output path for adapted checkpoint")
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    dataset = NpzDataset(args.npz)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    device = get_device()
    model = build_model(args.model, dataset.num_classes, dataset.in_channels)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    adapt_batchnorm(model, loader, device)
    save_checkpoint(model, Path(args.out_ckpt))


if __name__ == "__main__":
    main()

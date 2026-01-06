from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from src.models.resnet1d import resnet1d_teacher
from src.models.tinydscnn1d import tinydscnn1d_student


def build_model(name: str, num_classes: int = 5):
    name = name.lower()
    if name == "teacher":
        return resnet1d_teacher(num_classes=num_classes)
    if name in {"student", "student_aug"}:
        return tinydscnn1d_student(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_macs(model: nn.Module, input_len: int) -> float:
    macs = 0.0

    def hook(module, _inp, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            out = output
            if isinstance(out, tuple):
                out = out[0]
            batch, out_channels, out_len = out.shape
            kernel = module.kernel_size[0]
            in_per_group = module.in_channels // module.groups
            macs_layer = out_len * out_channels * in_per_group * kernel
            macs += macs_layer

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(hook))
    x = torch.zeros(1, 1, input_len)
    model.eval()
    with torch.no_grad():
        model(x)
    for handle in hooks:
        handle.remove()
    return macs


def main():
    parser = argparse.ArgumentParser(description="Edge readiness report for ECG models")
    parser.add_argument("--model", type=str, required=True, choices=["teacher", "student", "student_aug"])
    parser.add_argument("--input_len", type=int, default=256)
    args = parser.parse_args()

    model = build_model(args.model)
    params = param_count(model)
    macs = estimate_macs(model, args.input_len)
    size_fp32_kb = params * 4 / 1024
    size_int8_kb = params / 1024

    report = {
        "model": args.model,
        "parameters": params,
        "size_kb_fp32": round(size_fp32_kb, 2),
        "size_kb_int8": round(size_int8_kb, 2),
        "approx_macs": round(macs / 1e6, 2),
    }
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.task_loader import load_seed_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a seed task and execute it for manual ncu profiling.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--layout", default=None, choices=[None, "channels_last", "transposed", "non_contiguous"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_seed_task(args.task, device=args.device, dtype=args.dtype, layout=args.layout)
    model = bundle["model"]
    inputs = bundle["inputs"]

    import torch

    for _ in range(args.warmup):
        model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(args.iters):
        model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

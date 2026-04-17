#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.evaluator import evaluate_callable_pair
from newkernelbench.task_loader import load_seed_task
from newkernelbench.taxonomy import ExecutionMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a runtime smoke validation against a real KernelBench seed task.")
    parser.add_argument("--task", required=True, help="Path to a seed-task Python file.")
    parser.add_argument("--device", default="cuda" if _has_cuda() else "cpu", help="Runtime device.")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="Floating-point dtype override.")
    parser.add_argument("--layout", default=None, choices=[None, "channels_last", "transposed", "non_contiguous"], help="Optional layout transform.")
    parser.add_argument("--mode", default="forward", choices=["forward", "backward", "forward_backward"], help="Execution mode.")
    return parser.parse_args()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    bundle = load_seed_task(args.task, device=args.device, dtype=args.dtype, layout=args.layout)
    model = bundle["model"]
    inputs = bundle["inputs"]

    # Candidate matches the reference implementation. This is a smoke test for the evaluator pipeline.
    def reference_fn(*model_inputs):
        return model(*model_inputs)

    def candidate_fn(*model_inputs):
        return model(*model_inputs)

    result = evaluate_callable_pair(
        task_id=bundle["task_name"],
        variant_id=f"smoke-{args.dtype}-{args.layout or 'contiguous'}-{args.mode}",
        reference_fn=reference_fn,
        candidate_fn=candidate_fn,
        inputs=inputs,
        execution_mode=ExecutionMode(args.mode),
        requested_tools=["torch_profiler", "cuda_memory"],
    )
    print(json.dumps({
        "task": bundle["task_name"],
        "device": args.device,
        "dtype": args.dtype,
        "layout": args.layout or "contiguous",
        "mode": args.mode,
        "compiled": result.compiled,
        "correct": result.correct,
        "grad_correct": result.grad_correct,
        "speedup_vs_eager": result.speedup_vs_eager,
        "analysis_artifact_keys": sorted(result.analysis_artifacts.keys()),
        "failure_stage": result.failure_stage,
        "failure_reason": result.failure_reason,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

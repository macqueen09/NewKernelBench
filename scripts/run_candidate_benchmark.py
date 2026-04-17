#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.candidate_loader import build_candidate_model
from newkernelbench.evaluator import evaluate_callable_pair
from newkernelbench.schema import json_ready
from newkernelbench.task_loader import load_seed_task
from newkernelbench.taxonomy import ExecutionMode


try:
    import torch
except ImportError:  # pragma: no cover - runtime dependent
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a candidate implementation against a seed task and save a result bundle.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", default=None, help="Candidate python file defining ModelNew, Model, or build_candidate_model().")
    parser.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--layout", default=None, choices=[None, "channels_last", "transposed", "non_contiguous"])
    parser.add_argument("--mode", default="forward", choices=["forward", "backward", "forward_backward"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--with-compile-baseline", action="store_true")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def default_output_path(args: argparse.Namespace) -> Path:
    task_stem = Path(args.task).stem
    candidate_stem = Path(args.candidate).stem if args.candidate else "reference_copy"
    filename = f"{task_stem}__{candidate_stem}__{args.dtype}__{(args.layout or 'contiguous')}__{args.mode}.json"
    return ROOT / "results" / filename


def main() -> int:
    args = parse_args()
    bundle = load_seed_task(args.task, device=args.device, dtype=args.dtype, layout=args.layout)
    reference_model = bundle["model"]
    inputs = bundle["inputs"]
    init_inputs = bundle["init_inputs"]

    dtype_map = {
        "fp32": torch.float32 if torch is not None else None,
        "fp16": torch.float16 if torch is not None else None,
        "bf16": torch.bfloat16 if torch is not None else None,
    }
    candidate_model, candidate_notes = build_candidate_model(
        candidate_path=args.candidate,
        reference_model=reference_model,
        init_inputs=init_inputs,
        device=args.device,
        dtype=dtype_map[args.dtype],
        layout=args.layout,
    )

    compile_model = None
    compile_note = None
    if args.with_compile_baseline and torch is not None and hasattr(torch, "compile"):
        try:
            compile_model = torch.compile(copy.deepcopy(reference_model), mode=args.compile_mode)
            compile_note = f"compile baseline enabled with mode={args.compile_mode}"
        except Exception as exc:
            compile_note = f"compile baseline unavailable: {exc!r}"
    elif args.with_compile_baseline:
        compile_note = "compile baseline requested but torch.compile is unavailable"

    def reference_fn(*model_inputs):
        return reference_model(*model_inputs)

    def candidate_fn(*model_inputs):
        return candidate_model(*model_inputs)

    def compile_fn(*model_inputs):
        return compile_model(*model_inputs)

    result = evaluate_callable_pair(
        task_id=bundle["task_name"],
        variant_id=f"{args.dtype}-{args.layout or 'contiguous'}-{args.mode}",
        reference_fn=reference_fn,
        candidate_fn=candidate_fn,
        inputs=inputs,
        execution_mode=ExecutionMode(args.mode),
        requested_tools=["torch_profiler", "cuda_memory"],
        compile_baseline_fn=compile_fn if compile_model is not None else None,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "task": {
            "name": bundle["task_name"],
            "path": bundle["task_path"],
        },
        "candidate": {
            "path": args.candidate,
            "notes": candidate_notes,
        },
        "variant": {
            "device": args.device,
            "dtype": args.dtype,
            "layout": args.layout or "contiguous",
            "mode": args.mode,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": getattr(torch, "__version__", None),
            "device_name": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else "cpu",
            "compile_note": compile_note,
        },
        "result": json_ready(result),
    }

    profiler_text = result.analysis_artifacts.get("torch_profiler") if isinstance(result.analysis_artifacts, dict) else None
    if profiler_text:
        profiler_path = output_path.with_suffix(".torch_profiler.txt")
        profiler_path.write_text(profiler_text, encoding="utf-8")
        payload["result"]["analysis_artifacts"]["torch_profiler"] = str(profiler_path)

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.evaluator import maybe_profile, measure_runtime
from newkernelbench.task_loader import load_seed_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually run profiling tools on a seed task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--layout", default=None, choices=[None, "channels_last", "transposed", "non_contiguous"])
    parser.add_argument("--tool", default="all", choices=["all", "torch_profiler", "cuda_memory", "ncu"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--run-ncu", action="store_true")
    parser.add_argument("--ncu-set", default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_seed_task(args.task, device=args.device, dtype=args.dtype, layout=args.layout)
    model = bundle["model"]
    inputs = bundle["inputs"]

    runtime = measure_runtime(lambda: model(*inputs), warmup=args.warmup, repeat=args.repeat)
    requested_tools = []
    if args.tool in {"all", "torch_profiler"}:
        requested_tools.append("torch_profiler")
    if args.tool in {"all", "cuda_memory"}:
        requested_tools.append("cuda_memory")
    artifacts = maybe_profile(lambda: model(*inputs), requested_tools)

    summary = {
        "task": bundle["task_name"],
        "device": args.device,
        "dtype": args.dtype,
        "layout": args.layout or "contiguous",
        "runtime_seconds": runtime,
        "artifact_keys": sorted(artifacts.keys()),
    }

    if "torch_profiler" in artifacts:
        profiler_path = output_dir / "torch_profiler.txt"
        profiler_path.write_text(artifacts["torch_profiler"], encoding="utf-8")
        summary["torch_profiler_path"] = str(profiler_path)
    if "cuda_memory" in artifacts:
        memory_path = output_dir / "cuda_memory.json"
        memory_path.write_text(json.dumps(artifacts["cuda_memory"], indent=2), encoding="utf-8")
        summary["cuda_memory_path"] = str(memory_path)

    if args.tool in {"all", "ncu"}:
        ncu_prefix = output_dir / "ncu_report"
        ncu_command = [
            "ncu",
            "--target-processes",
            "all",
            "--set",
            args.ncu_set,
            "--force-overwrite",
            "--export",
            str(ncu_prefix),
            sys.executable,
            str(ROOT / "scripts" / "ncu_profile_task.py"),
            "--task",
            args.task,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
        ]
        if args.layout:
            ncu_command.extend(["--layout", args.layout])
        summary["ncu_command"] = " ".join(ncu_command)
        if args.run_ncu:
            env = os.environ.copy()
            completed = subprocess.run(ncu_command, capture_output=True, text=True, env=env)
            (output_dir / "ncu_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output_dir / "ncu_stderr.txt").write_text(completed.stderr, encoding="utf-8")
            summary["ncu_returncode"] = completed.returncode
            summary["ncu_report_prefix"] = str(ncu_prefix)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

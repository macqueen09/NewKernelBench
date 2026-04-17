#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.planner import build_manifest, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a starter NewKernelBench manifest from KernelBench seed tasks.")
    parser.add_argument(
        "--kernelbench-root",
        default=str(ROOT.parent / "KernelBench" / "KernelBench"),
        help="Path to the original KernelBench task root (contains level1/level2/level3).",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "configs" / "starter_manifest.json"),
        help="Where to write the generated manifest JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on imported seed tasks for quick iteration.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Subset of KernelBench levels to import.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        kernelbench_root=args.kernelbench_root,
        include_levels=args.levels,
        limit=args.limit,
    )
    output = save_manifest(manifest, args.output)

    summary = manifest["summary"]
    print(f"Wrote manifest to: {output}")
    print(f"Source tasks: {summary['source_task_count']}")
    print(f"Variants: {summary['variant_count']}")
    print(f"By level: {summary['counts_by_level']}")
    print(f"By analysis tier: {summary['counts_by_analysis_tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

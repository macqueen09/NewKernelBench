#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a NewKernelBench manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest JSON.")
    parser.add_argument("--show-tasks", type=int, default=3, help="Number of task previews to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print("\nSample tasks:")
    for entry in manifest["tasks"][: args.show_tasks]:
        source = entry["source"]
        variants = entry["variants"]
        print("-" * 80)
        print(f"Task {source['task_id']}: {source['name']}")
        print(f"  level={source['level']} categories={source['categories']} differentiable={source['differentiable']}")
        print(f"  variants={len(variants)}")
        if variants:
            print(f"  first_variant={variants[0]['variant_id']}")
            print(f"  tools={variants[0]['analysis']['selected_tools']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

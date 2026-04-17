#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from newkernelbench.taxonomy import CATEGORY_KEYWORDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest candidate operators or workloads from a Python module.")
    parser.add_argument("--module", required=True, help="Base module to inspect, for example torch.nn.")
    parser.add_argument("--walk-submodules", action="store_true", help="Recursively inspect importable submodules.")
    parser.add_argument("--max-submodules", type=int, default=20, help="Cap walked submodules to avoid huge imports.")
    parser.add_argument("--member-kind", choices=["all", "class", "function"], default="all")
    parser.add_argument("--limit", type=int, default=200, help="Maximum harvested members.")
    parser.add_argument("--name-pattern", default=None, help="Optional regex filter for member names.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def classify_name(name: str) -> list[str]:
    lowered = name.lower()
    categories = []
    for keywords, category in CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            categories.append(category.value)
    return categories or ["other"]


def iter_modules(base_module_name: str, walk_submodules: bool, max_submodules: int):
    root = importlib.import_module(base_module_name)
    yield root
    if not walk_submodules or not hasattr(root, "__path__"):
        return

    count = 0
    for module_info in pkgutil.walk_packages(root.__path__, prefix=root.__name__ + "."):
        if count >= max_submodules:
            return
        try:
            yield importlib.import_module(module_info.name)
            count += 1
        except Exception:
            continue


def member_allowed(member, member_kind: str) -> bool:
    if member_kind == "class":
        return inspect.isclass(member)
    if member_kind == "function":
        return inspect.isfunction(member) or inspect.isbuiltin(member)
    return inspect.isclass(member) or inspect.isfunction(member) or inspect.isbuiltin(member)


def main() -> int:
    args = parse_args()
    regex = re.compile(args.name_pattern) if args.name_pattern else None

    records = []
    seen = set()
    for module in iter_modules(args.module, args.walk_submodules, args.max_submodules):
        for name, member in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if regex and not regex.search(name):
                continue
            if not member_allowed(member, args.member_kind):
                continue
            qualname = getattr(member, "__qualname__", name)
            key = (module.__name__, qualname)
            if key in seen:
                continue
            seen.add(key)
            try:
                signature = str(inspect.signature(member))
            except Exception:
                signature = "<signature unavailable>"
            records.append(
                {
                    "module": module.__name__,
                    "name": name,
                    "qualname": qualname,
                    "member_type": "class" if inspect.isclass(member) else "function",
                    "signature": signature,
                    "guessed_categories": classify_name(name),
                }
            )
            if len(records) >= args.limit:
                break
        if len(records) >= args.limit:
            break

    category_counts = Counter()
    for item in records:
        for category in item["guessed_categories"]:
            category_counts[category] += 1

    payload = {
        "module": args.module,
        "walk_submodules": args.walk_submodules,
        "record_count": len(records),
        "counts_by_category": dict(sorted(category_counts.items())),
        "records": records,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote harvest output to {output_path}")

    print(json.dumps({
        "module": payload["module"],
        "record_count": payload["record_count"],
        "counts_by_category": payload["counts_by_category"],
        "sample_names": [item["name"] for item in records[:10]],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

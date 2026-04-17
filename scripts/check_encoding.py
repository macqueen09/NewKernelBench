#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".json", ".toml"}
REPLACEMENT_CHAR = chr(0xFFFD)


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that project text files are valid UTF-8.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Project root to scan.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures = []

    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            failures.append((path, f"decode_error: {exc!r}"))
            continue
        if REPLACEMENT_CHAR in text:
            failures.append((path, "contains replacement character U+FFFD"))

    if failures:
        print("UTF-8 check failed:")
        for path, reason in failures:
            print(f"- {path}: {reason}")
        return 1

    print(f"UTF-8 check passed for {sum(1 for _ in iter_files(root))} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

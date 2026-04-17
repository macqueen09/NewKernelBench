from __future__ import annotations

from pathlib import Path

from .schema import SourceTask
from .taxonomy import CATEGORY_KEYWORDS, ComplexityTier, FunctionalCategory, NON_DIFFERENTIABLE_KEYWORDS

LEVEL_DIRS = {
    1: "level1",
    2: "level2",
    3: "level3",
}


def default_kernelbench_root() -> Path:
    return Path(__file__).resolve().parents[3] / "KernelBench" / "KernelBench"


def _guess_complexity(level: int) -> ComplexityTier:
    if level == 1:
        return ComplexityTier.PRIMITIVE
    if level == 2:
        return ComplexityTier.FUSION
    return ComplexityTier.FULL_MODEL


def _guess_categories(name: str, code: str, level: int) -> list[FunctionalCategory]:
    haystack = f"{name}\n{code}".lower()
    categories: list[FunctionalCategory] = []
    for keywords, category in CATEGORY_KEYWORDS:
        if any(keyword in haystack for keyword in keywords) and category not in categories:
            categories.append(category)

    if level == 2 and FunctionalCategory.FUSION not in categories:
        categories.insert(0, FunctionalCategory.FUSION)
    if level == 3 and FunctionalCategory.FULL_MODEL not in categories:
        categories.insert(0, FunctionalCategory.FULL_MODEL)

    if not categories:
        categories.append(FunctionalCategory.OTHER)
    return categories


def _guess_libraries(code: str) -> list[str]:
    code_lower = code.lower()
    libraries = ["torch"]
    if "import torch.nn.functional" in code_lower or "f." in code_lower:
        libraries.append("torch.nn.functional")
    if "nn." in code_lower or "import torch.nn" in code_lower:
        libraries.append("torch.nn")
    if "transformers" in code_lower:
        libraries.append("transformers")
    if "timm" in code_lower:
        libraries.append("timm")
    return libraries


def _is_differentiable(name: str, code: str) -> bool:
    haystack = f"{name}\n{code}".lower()
    return not any(keyword in haystack for keyword in NON_DIFFERENTIABLE_KEYWORDS)


def scan_kernelbench(kernelbench_root: str | Path | None = None, include_levels: list[int] | None = None, limit: int | None = None) -> list[SourceTask]:
    root = Path(kernelbench_root) if kernelbench_root else default_kernelbench_root()
    levels = include_levels or [1, 2, 3]

    tasks: list[SourceTask] = []
    for level in levels:
        level_dir = root / LEVEL_DIRS[level]
        if not level_dir.exists():
            continue

        for file_path in sorted(level_dir.glob("*.py")):
            code = file_path.read_text(encoding="utf-8")
            categories = _guess_categories(file_path.name, code, level)
            notes: list[str] = []
            if level == 2:
                notes.append("Fusion-level seed task from original KernelBench.")
            if level == 3:
                notes.append("Full-model seed task; prioritize deeper profiling and microbatch variants.")
            if any(category in categories for category in (FunctionalCategory.CONVOLUTION, FunctionalCategory.MATMUL, FunctionalCategory.ATTENTION)):
                notes.append("Candidate for non-contiguous layout and deeper analysis tiers.")

            tasks.append(
                SourceTask(
                    task_id=file_path.stem.split("_", 1)[0],
                    name=file_path.name,
                    relative_path=file_path.relative_to(root).as_posix(),
                    origin="kernelbench",
                    level=level,
                    complexity=_guess_complexity(level),
                    categories=categories,
                    libraries=_guess_libraries(code),
                    differentiable=_is_differentiable(file_path.name, code),
                    notes=notes,
                )
            )
            if limit and len(tasks) >= limit:
                return tasks
    return tasks

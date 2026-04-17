from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .analysis_plan import detect_available_tools
from .catalog import default_harvest_catalog
from .kernelbench_adapter import default_kernelbench_root, scan_kernelbench
from .schema import BenchmarkTaskPlan, json_ready
from .variants import build_variants


def _plan_rationale(task) -> list[str]:
    rationale = [f"Imported from KernelBench level {task.level}."]
    if task.differentiable:
        rationale.append("Includes backward-capable variants to cover training scenarios.")
    else:
        rationale.append("Backward variants disabled by heuristic because the seed task looks non-differentiable.")
    if any(category.value in {"matmul", "convolution", "attention"} for category in task.categories):
        rationale.append("Expanded with layout stress and deeper analysis because this family is often performance-critical.")
    if task.level == 3:
        rationale.append("Full-model tasks prioritize microbatch and serving-style variants instead of exploding the variant grid.")
    return rationale


def build_manifest(kernelbench_root: str | Path | None = None, include_levels: list[int] | None = None, limit: int | None = None) -> dict:
    root = Path(kernelbench_root) if kernelbench_root else default_kernelbench_root()
    available_tools = detect_available_tools()
    source_tasks = scan_kernelbench(root, include_levels=include_levels, limit=limit)

    task_plans: list[BenchmarkTaskPlan] = []
    for source in source_tasks:
        task_plans.append(
            BenchmarkTaskPlan(
                source=source,
                variants=build_variants(source, available_tools=available_tools),
                rationale=_plan_rationale(source),
            )
        )

    summary = summarize_manifest(task_plans)
    return {
        "meta": {
            "manifest_version": 1,
            "kernelbench_root": root.as_posix(),
            "available_tools": available_tools,
            "harvest_catalog": [target.to_dict() for target in default_harvest_catalog()],
        },
        "summary": summary,
        "tasks": [json_ready(plan) for plan in task_plans],
    }


def summarize_manifest(task_plans: list[BenchmarkTaskPlan] | dict) -> dict:
    if isinstance(task_plans, dict):
        tasks = task_plans.get("tasks", [])
        source_count = len(tasks)
        variant_count = 0
        level_counter = Counter()
        complexity_counter = Counter()
        category_counter = Counter()
        dtype_counter = Counter()
        layout_counter = Counter()
        mode_counter = Counter()
        tier_counter = Counter()

        for entry in tasks:
            source = entry["source"]
            variants = entry["variants"]
            variant_count += len(variants)
            level_counter[str(source["level"])] += 1
            complexity_counter[source["complexity"]] += 1
            for category in source["categories"]:
                category_counter[category] += 1
            for variant in variants:
                dtype_counter[variant["dtype"]] += 1
                layout_counter[variant["layout"]] += 1
                mode_counter[variant["execution_mode"]] += 1
                tier_counter[variant["analysis"]["tier"]] += 1
    else:
        source_count = len(task_plans)
        variant_count = 0
        level_counter = Counter()
        complexity_counter = Counter()
        category_counter = Counter()
        dtype_counter = Counter()
        layout_counter = Counter()
        mode_counter = Counter()
        tier_counter = Counter()

        for plan in task_plans:
            variant_count += len(plan.variants)
            level_counter[str(plan.source.level)] += 1
            complexity_counter[plan.source.complexity.value] += 1
            for category in plan.source.categories:
                category_counter[category.value] += 1
            for variant in plan.variants:
                dtype_counter[variant.dtype.value] += 1
                layout_counter[variant.layout.value] += 1
                mode_counter[variant.execution_mode.value] += 1
                tier_counter[variant.analysis.tier.value] += 1

    return {
        "source_task_count": source_count,
        "variant_count": variant_count,
        "counts_by_level": dict(sorted(level_counter.items())),
        "counts_by_complexity": dict(sorted(complexity_counter.items())),
        "counts_by_category": dict(sorted(category_counter.items())),
        "counts_by_dtype": dict(sorted(dtype_counter.items())),
        "counts_by_layout": dict(sorted(layout_counter.items())),
        "counts_by_execution_mode": dict(sorted(mode_counter.items())),
        "counts_by_analysis_tier": dict(sorted(tier_counter.items())),
    }


def save_manifest(manifest: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output

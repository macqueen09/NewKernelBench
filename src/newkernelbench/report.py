from __future__ import annotations

from collections import Counter

from .schema import TaskExecutionResult


def summarize_results(results: list[TaskExecutionResult]) -> dict:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "compile_rate": 0.0,
            "correct_rate": 0.0,
            "grad_rate": 0.0,
            "speedup_gt_1_eager": 0,
            "speedup_gt_1_compile": 0,
            "failure_breakdown": {},
        }

    compiled = sum(1 for result in results if result.compiled)
    correct = sum(1 for result in results if result.correct)
    grad_checks = [result.grad_correct for result in results if result.grad_correct is not None]
    grad_correct = sum(1 for value in grad_checks if value)

    failure_breakdown = Counter(result.failure_stage or "none" for result in results)

    return {
        "total": total,
        "compile_rate": compiled / total,
        "correct_rate": correct / total,
        "grad_rate": (grad_correct / len(grad_checks)) if grad_checks else 0.0,
        "speedup_gt_1_eager": sum(1 for result in results if (result.speedup_vs_eager or 0.0) > 1.0),
        "speedup_gt_1_compile": sum(1 for result in results if (result.speedup_vs_compile or 0.0) > 1.0),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
    }


def render_markdown_summary(summary: dict) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        f"- Total variants: {summary['total']}",
        f"- Compile rate: {summary['compile_rate']:.2%}",
        f"- Correct rate: {summary['correct_rate']:.2%}",
        f"- Gradient correct rate: {summary['grad_rate']:.2%}",
        f"- Variants faster than eager: {summary['speedup_gt_1_eager']}",
        f"- Variants faster than compile baseline: {summary['speedup_gt_1_compile']}",
        "",
        "## Failure Breakdown",
        "",
    ]
    for key, value in summary["failure_breakdown"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

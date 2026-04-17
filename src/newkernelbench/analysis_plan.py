from __future__ import annotations

from importlib.util import find_spec
from shutil import which

from .schema import AnalysisPlan, ShapeBucket, SourceTask
from .taxonomy import AnalysisTier, DTypeTag, ExecutionMode, FunctionalCategory, LayoutTag


TOOL_REGISTRY = {
    "correctness": "Forward correctness check against the reference implementation.",
    "grad_correctness": "Gradient correctness check for backward-enabled variants.",
    "eager_timing": "Wall-time benchmark against eager baseline.",
    "compile_timing": "Wall-time benchmark against compile baseline if available.",
    "stability_sweep": "Repeat checks across dtype/layout edge cases to catch flaky kernels.",
    "torch_profiler": "Operator and kernel breakdown using torch profiler.",
    "cuda_memory": "Track allocated and peak CUDA memory for stress cases.",
    "ncu": "Nsight Compute metrics for kernel occupancy, throughput, and roofline-style analysis.",
}


def detect_available_tools() -> dict[str, bool]:
    torch_available = find_spec("torch") is not None
    torch_profiler = False
    cuda_memory = False
    if torch_available:
        try:
            import torch

            torch_profiler = hasattr(torch, "profiler")
            cuda_memory = bool(torch.cuda.is_available())
        except Exception:
            torch_profiler = False
            cuda_memory = False

    return {
        "correctness": True,
        "grad_correctness": True,
        "eager_timing": True,
        "compile_timing": torch_available,
        "stability_sweep": True,
        "torch_profiler": torch_profiler,
        "cuda_memory": cuda_memory,
        "ncu": which("ncu") is not None,
    }


def choose_analysis_tier(task: SourceTask, shape_bucket: ShapeBucket, dtype: DTypeTag, layout: LayoutTag, execution_mode: ExecutionMode) -> AnalysisTier:
    score = 0

    if task.level == 2:
        score += 1
    elif task.level == 3:
        score += 2

    if any(category in task.categories for category in (FunctionalCategory.ATTENTION, FunctionalCategory.FULL_MODEL)):
        score += 2
    elif any(category in task.categories for category in (FunctionalCategory.CONVOLUTION, FunctionalCategory.MATMUL, FunctionalCategory.OPTIMIZER)):
        score += 1

    if shape_bucket.scale == "large":
        score += 1
    if layout != LayoutTag.CONTIGUOUS:
        score += 1
    if dtype == DTypeTag.INT8:
        score += 1
    if execution_mode != ExecutionMode.FORWARD:
        score += 1

    if score <= 1:
        return AnalysisTier.LIGHT
    if score <= 3:
        return AnalysisTier.STANDARD
    return AnalysisTier.DEEP


def build_analysis_plan(task: SourceTask, shape_bucket: ShapeBucket, dtype: DTypeTag, layout: LayoutTag, execution_mode: ExecutionMode, available_tools: dict[str, bool] | None = None) -> AnalysisPlan:
    availability = available_tools or detect_available_tools()
    tier = choose_analysis_tier(task, shape_bucket, dtype, layout, execution_mode)

    selected = ["correctness", "eager_timing"]
    notes: list[str] = []

    if execution_mode != ExecutionMode.FORWARD:
        selected.append("grad_correctness")

    if dtype != DTypeTag.FP32 or layout != LayoutTag.CONTIGUOUS:
        selected.append("stability_sweep")

    if tier in (AnalysisTier.STANDARD, AnalysisTier.DEEP):
        if availability.get("compile_timing", False):
            selected.append("compile_timing")
        else:
            notes.append("compile_timing not available in current environment; fall back to eager timing only.")

        if availability.get("torch_profiler", False):
            selected.append("torch_profiler")
        else:
            notes.append("torch.profiler unavailable; standard-tier profiling will be skipped.")

    if tier == AnalysisTier.DEEP:
        if availability.get("cuda_memory", False):
            selected.append("cuda_memory")
        else:
            notes.append("CUDA memory metrics unavailable; deep analysis will omit memory breakdown.")

        if availability.get("ncu", False):
            selected.append("ncu")
        else:
            notes.append("ncu unavailable; deep analysis falls back to torch-profiler-only diagnostics.")

    selected = list(dict.fromkeys(selected))
    return AnalysisPlan(tier=tier, selected_tools=selected, notes=notes)

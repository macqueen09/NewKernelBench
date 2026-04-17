from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .schema import TaskExecutionResult
from .taxonomy import ExecutionMode

try:
    import torch
except ImportError:  # pragma: no cover - depends on remote runtime env
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Torch is required for runtime evaluation, but it is not installed in the current environment.")


def _tree_clone(value: Any, requires_grad: bool = False) -> Any:
    if torch is not None and torch.is_tensor(value):
        cloned = value.detach().clone()
        if requires_grad and cloned.dtype.is_floating_point:
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, list):
        return [_tree_clone(item, requires_grad=requires_grad) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_clone(item, requires_grad=requires_grad) for item in value)
    if isinstance(value, dict):
        return {key: _tree_clone(inner, requires_grad=requires_grad) for key, inner in value.items()}
    return value


def _tree_allclose(lhs: Any, rhs: Any, rtol: float, atol: float) -> bool:
    if torch is not None and torch.is_tensor(lhs) and torch.is_tensor(rhs):
        return torch.allclose(lhs, rhs, rtol=rtol, atol=atol)
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)) and len(lhs) == len(rhs):
        return all(_tree_allclose(left, right, rtol=rtol, atol=atol) for left, right in zip(lhs, rhs))
    if isinstance(lhs, dict) and isinstance(rhs, dict) and lhs.keys() == rhs.keys():
        return all(_tree_allclose(lhs[key], rhs[key], rtol=rtol, atol=atol) for key in lhs)
    return lhs == rhs


def _sum_tensor_tree(value: Any):
    if torch is not None and torch.is_tensor(value):
        return value.sum()
    if isinstance(value, (list, tuple)):
        acc = None
        for item in value:
            current = _sum_tensor_tree(item)
            acc = current if acc is None else acc + current
        return acc
    if isinstance(value, dict):
        acc = None
        for item in value.values():
            current = _sum_tensor_tree(item)
            acc = current if acc is None else acc + current
        return acc
    raise TypeError(f"Unsupported output type for backward: {type(value)!r}")


def _collect_input_grads(inputs: list[Any]) -> list[Any]:
    grads = []
    for item in inputs:
        if torch is not None and torch.is_tensor(item) and item.requires_grad:
            grads.append(None if item.grad is None else item.grad.detach().clone())
    return grads


def measure_runtime(fn: Callable[[], Any], warmup: int = 3, repeat: int = 10) -> float:
    _require_torch()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / repeat


def maybe_profile(fn: Callable[[], Any], requested_tools: list[str]) -> dict[str, Any]:
    _require_torch()
    artifacts: dict[str, Any] = {}

    if "torch_profiler" in requested_tools and hasattr(torch, "profiler"):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            fn()
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        artifacts["torch_profiler"] = prof.key_averages().table(sort_by=sort_key, row_limit=10)

    if "cuda_memory" in requested_tools and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        artifacts["cuda_memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    return artifacts


def evaluate_callable_pair(
    task_id: str,
    variant_id: str,
    reference_fn: Callable[..., Any],
    candidate_fn: Callable[..., Any],
    inputs: list[Any],
    execution_mode: ExecutionMode = ExecutionMode.FORWARD,
    requested_tools: list[str] | None = None,
    compile_baseline_fn: Callable[..., Any] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    warmup: int = 3,
    repeat: int = 10,
) -> TaskExecutionResult:
    _require_torch()
    requested_tools = requested_tools or []

    ref_inputs = _tree_clone(inputs, requires_grad=execution_mode != ExecutionMode.FORWARD)
    cand_inputs = _tree_clone(inputs, requires_grad=execution_mode != ExecutionMode.FORWARD)

    try:
        ref_output = reference_fn(*ref_inputs)
        cand_output = candidate_fn(*cand_inputs)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return TaskExecutionResult(
            task_id=task_id,
            variant_id=variant_id,
            compiled=False,
            correct=False,
            failure_stage="execution",
            failure_reason=repr(exc),
        )

    correct = _tree_allclose(ref_output, cand_output, rtol=rtol, atol=atol)
    grad_correct = None

    if execution_mode != ExecutionMode.FORWARD:
        try:
            _sum_tensor_tree(ref_output).backward()
            _sum_tensor_tree(cand_output).backward()
            grad_correct = _tree_allclose(_collect_input_grads(ref_inputs), _collect_input_grads(cand_inputs), rtol=rtol, atol=atol)
        except Exception as exc:  # pragma: no cover - runtime dependent
            return TaskExecutionResult(
                task_id=task_id,
                variant_id=variant_id,
                compiled=True,
                correct=correct,
                grad_correct=False,
                failure_stage="backward",
                failure_reason=repr(exc),
            )

    try:
        candidate_runtime = measure_runtime(lambda: candidate_fn(*_tree_clone(inputs, requires_grad=False)), warmup=warmup, repeat=repeat)
        eager_runtime = measure_runtime(lambda: reference_fn(*_tree_clone(inputs, requires_grad=False)), warmup=warmup, repeat=repeat)
        compile_runtime = None
        if compile_baseline_fn is not None:
            compile_runtime = measure_runtime(lambda: compile_baseline_fn(*_tree_clone(inputs, requires_grad=False)), warmup=warmup, repeat=repeat)
        artifacts = maybe_profile(lambda: candidate_fn(*_tree_clone(inputs, requires_grad=False)), requested_tools)
    except Exception as exc:  # pragma: no cover - runtime dependent
        return TaskExecutionResult(
            task_id=task_id,
            variant_id=variant_id,
            compiled=True,
            correct=correct,
            grad_correct=grad_correct,
            failure_stage="timing",
            failure_reason=repr(exc),
        )

    speedup_vs_eager = (eager_runtime / candidate_runtime) if candidate_runtime else None
    speedup_vs_compile = (compile_runtime / candidate_runtime) if compile_runtime else None

    return TaskExecutionResult(
        task_id=task_id,
        variant_id=variant_id,
        compiled=True,
        correct=correct,
        grad_correct=grad_correct,
        speedup_vs_eager=speedup_vs_eager,
        speedup_vs_compile=speedup_vs_compile,
        analysis_artifacts=artifacts,
    )

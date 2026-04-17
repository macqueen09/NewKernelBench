from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from .taxonomy import AnalysisTier, ComplexityTier, DTypeTag, ExecutionMode, FunctionalCategory, LayoutTag


@dataclass(slots=True)
class ShapeBucket:
    name: str
    description: str
    scale: str


@dataclass(slots=True)
class AnalysisPlan:
    tier: AnalysisTier
    selected_tools: list[str]
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceTask:
    task_id: str
    name: str
    relative_path: str
    origin: str
    level: int
    complexity: ComplexityTier
    categories: list[FunctionalCategory]
    libraries: list[str]
    differentiable: bool
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskVariant:
    variant_id: str
    shape_bucket: ShapeBucket
    dtype: DTypeTag
    layout: LayoutTag
    execution_mode: ExecutionMode
    analysis: AnalysisPlan


@dataclass(slots=True)
class BenchmarkTaskPlan:
    source: SourceTask
    variants: list[TaskVariant]
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskExecutionResult:
    task_id: str
    variant_id: str
    compiled: bool
    correct: bool
    grad_correct: bool | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_compile: float | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    analysis_artifacts: dict[str, Any] = field(default_factory=dict)


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: json_ready(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(inner) for inner in value]
    return value

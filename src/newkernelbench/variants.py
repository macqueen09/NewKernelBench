from __future__ import annotations

from .analysis_plan import build_analysis_plan
from .schema import ShapeBucket, SourceTask, TaskVariant
from .taxonomy import DTypeTag, ExecutionMode, FunctionalCategory, INT8_FRIENDLY_CATEGORIES, LayoutTag


PRIMITIVE_BUCKETS = [
    ShapeBucket(name="baseline", description="Matches the upstream seed-task scale.", scale="medium"),
    ShapeBucket(name="stress", description="Larger or more awkward shape bucket for stress testing.", scale="large"),
]

FUSION_BUCKETS = [
    ShapeBucket(name="baseline", description="Default fused workload bucket.", scale="medium"),
    ShapeBucket(name="stress", description="Large fused workload bucket with more pressure on memory traffic.", scale="large"),
]

FULL_MODEL_BUCKETS = [
    ShapeBucket(name="microbatch", description="Training-style microbatch bucket.", scale="medium"),
    ShapeBucket(name="serving", description="Inference-style serving bucket.", scale="large"),
]


def default_shape_buckets(task: SourceTask) -> list[ShapeBucket]:
    if task.level == 1:
        return PRIMITIVE_BUCKETS
    if task.level == 2:
        return FUSION_BUCKETS
    return FULL_MODEL_BUCKETS


def default_dtypes(task: SourceTask) -> list[DTypeTag]:
    dtypes = [DTypeTag.FP32]
    if any(category in task.categories for category in (FunctionalCategory.CONVOLUTION, FunctionalCategory.MATMUL, FunctionalCategory.NORMALIZATION, FunctionalCategory.ATTENTION, FunctionalCategory.FULL_MODEL, FunctionalCategory.FUSION)):
        dtypes.extend([DTypeTag.FP16, DTypeTag.BF16])
    if task.level != 3 and any(category in task.categories for category in INT8_FRIENDLY_CATEGORIES):
        dtypes.append(DTypeTag.INT8)
    return list(dict.fromkeys(dtypes))


def default_layouts(task: SourceTask) -> list[LayoutTag]:
    layouts = [LayoutTag.CONTIGUOUS]
    if any(category in task.categories for category in (FunctionalCategory.CONVOLUTION, FunctionalCategory.FULL_MODEL)):
        layouts.append(LayoutTag.CHANNELS_LAST)
    if any(category in task.categories for category in (FunctionalCategory.MATMUL, FunctionalCategory.NORMALIZATION, FunctionalCategory.REDUCTION, FunctionalCategory.ATTENTION)):
        layouts.extend([LayoutTag.TRANSPOSED, LayoutTag.NON_CONTIGUOUS])
    elif task.level == 2:
        layouts.append(LayoutTag.NON_CONTIGUOUS)
    return list(dict.fromkeys(layouts))


def default_execution_modes(task: SourceTask, dtype: DTypeTag, bucket: ShapeBucket) -> list[ExecutionMode]:
    modes = [ExecutionMode.FORWARD]
    if task.differentiable and dtype != DTypeTag.INT8 and bucket.name in {"baseline", "microbatch"}:
        modes.extend([ExecutionMode.BACKWARD, ExecutionMode.FORWARD_BACKWARD])
    return modes


def build_variants(task: SourceTask, available_tools: dict[str, bool] | None = None) -> list[TaskVariant]:
    variants: list[TaskVariant] = []

    for bucket in default_shape_buckets(task):
        for dtype in default_dtypes(task):
            for layout in default_layouts(task):
                for mode in default_execution_modes(task, dtype, bucket):
                    variant_id = f"{bucket.name}-{dtype.value}-{layout.value}-{mode.value}"
                    analysis = build_analysis_plan(
                        task=task,
                        shape_bucket=bucket,
                        dtype=dtype,
                        layout=layout,
                        execution_mode=mode,
                        available_tools=available_tools,
                    )
                    variants.append(
                        TaskVariant(
                            variant_id=variant_id,
                            shape_bucket=bucket,
                            dtype=dtype,
                            layout=layout,
                            execution_mode=mode,
                            analysis=analysis,
                        )
                    )
    return variants

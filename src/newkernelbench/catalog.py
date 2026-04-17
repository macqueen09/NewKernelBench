from __future__ import annotations

from dataclasses import dataclass

from .schema import json_ready
from .taxonomy import ComplexityTier, FunctionalCategory


@dataclass(slots=True)
class LibraryHarvestTarget:
    name: str
    complexity: ComplexityTier
    categories: list[FunctionalCategory]
    candidate_modules: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return json_ready(self)


def default_harvest_catalog() -> list[LibraryHarvestTarget]:
    return [
        LibraryHarvestTarget(
            name="kernelbench_seed_tasks",
            complexity=ComplexityTier.PRIMITIVE,
            categories=[FunctionalCategory.MATMUL, FunctionalCategory.CONVOLUTION, FunctionalCategory.NORMALIZATION],
            candidate_modules=["../KernelBench/KernelBench/level1", "../KernelBench/KernelBench/level2", "../KernelBench/KernelBench/level3"],
            notes=["Primary seed source for the first phase of NewKernelBench."],
        ),
        LibraryHarvestTarget(
            name="torch_core_ops",
            complexity=ComplexityTier.PRIMITIVE,
            categories=[FunctionalCategory.ACTIVATION, FunctionalCategory.CONVOLUTION, FunctionalCategory.LOSS, FunctionalCategory.POOLING],
            candidate_modules=["torch.nn", "torch.nn.functional"],
            notes=["Best next source for filling missing primitive operator families."],
        ),
        LibraryHarvestTarget(
            name="transformer_blocks",
            complexity=ComplexityTier.FUSION,
            categories=[FunctionalCategory.ATTENTION, FunctionalCategory.FUSION, FunctionalCategory.NORMALIZATION],
            candidate_modules=["transformers.models", "xformers.ops", "flash_attn"],
            notes=["Good source for backward-sensitive and memory-bound fused subgraphs."],
        ),
        LibraryHarvestTarget(
            name="vision_backbones",
            complexity=ComplexityTier.FULL_MODEL,
            categories=[FunctionalCategory.CONVOLUTION, FunctionalCategory.FULL_MODEL],
            candidate_modules=["timm.models", "torchvision.models"],
            notes=["Useful for full-model and channels-last stress cases."],
        ),
        LibraryHarvestTarget(
            name="quantization_and_optimizer",
            complexity=ComplexityTier.FUSION,
            categories=[FunctionalCategory.OPTIMIZER, FunctionalCategory.MATMUL],
            candidate_modules=["bitsandbytes", "torchao"],
            notes=["Targets int8 and optimizer-heavy training workflows."],
        ),
    ]

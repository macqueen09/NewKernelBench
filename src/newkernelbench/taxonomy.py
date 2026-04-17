from __future__ import annotations

from enum import Enum


class ComplexityTier(str, Enum):
    PRIMITIVE = "primitive"
    FUSION = "fusion"
    FULL_MODEL = "full_model"


class FunctionalCategory(str, Enum):
    ACTIVATION = "activation"
    ATTENTION = "attention"
    BROADCAST = "broadcast"
    CONVOLUTION = "convolution"
    ELEMENTWISE = "elementwise"
    FULL_MODEL = "full_model"
    FUSION = "fusion"
    INDEXING = "indexing"
    LOSS = "loss"
    MATMUL = "matmul"
    NORMALIZATION = "normalization"
    OPTIMIZER = "optimizer"
    OTHER = "other"
    POOLING = "pooling"
    REDUCTION = "reduction"
    RESIZE = "resize"


class ExecutionMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    FORWARD_BACKWARD = "forward_backward"


class DTypeTag(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class LayoutTag(str, Enum):
    CONTIGUOUS = "contiguous"
    TRANSPOSED = "transposed"
    NON_CONTIGUOUS = "non_contiguous"
    CHANNELS_LAST = "channels_last"


class AnalysisTier(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


CATEGORY_KEYWORDS: tuple[tuple[tuple[str, ...], FunctionalCategory], ...] = (
    (("attention", "flashattn", "scaleddotproductattention", "mha"), FunctionalCategory.ATTENTION),
    (("optimizer", "adam", "sgd", "momentum"), FunctionalCategory.OPTIMIZER),
    (("loss", "crossentropy", "hingeloss", "mseloss", "huberloss", "kldiv", "triplet"), FunctionalCategory.LOSS),
    (("conv", "convolution"), FunctionalCategory.CONVOLUTION),
    (("matmul", "gemm", "bmm", "matrix"), FunctionalCategory.MATMUL),
    (("batchnorm", "layernorm", "instancenorm", "groupnorm", "rmsnorm", "norm"), FunctionalCategory.NORMALIZATION),
    (("pool",), FunctionalCategory.POOLING),
    (("softmax", "relu", "gelu", "sigmoid", "tanh", "swish", "mish", "selu", "elu", "hardtanh", "hardswish"), FunctionalCategory.ACTIVATION),
    (("reduce", "sum", "mean", "max", "min", "logsumexp", "cumsum", "cumprod"), FunctionalCategory.REDUCTION),
    (("argmax", "argmin", "gather", "scatter", "index"), FunctionalCategory.INDEXING),
    (("resize", "upsample", "interpolate"), FunctionalCategory.RESIZE),
    (("bias", "broadcast"), FunctionalCategory.BROADCAST),
    (("add", "multiply", "divide", "subtract", "clamp", "scale"), FunctionalCategory.ELEMENTWISE),
)

NON_DIFFERENTIABLE_KEYWORDS = {
    "argmax",
    "argmin",
    "scatter",
    "topk",
    "sort",
    "unique",
}

INT8_FRIENDLY_CATEGORIES = {
    FunctionalCategory.BROADCAST,
    FunctionalCategory.CONVOLUTION,
    FunctionalCategory.ELEMENTWISE,
    FunctionalCategory.MATMUL,
    FunctionalCategory.NORMALIZATION,
    FunctionalCategory.REDUCTION,
}

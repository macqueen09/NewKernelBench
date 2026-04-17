from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependent
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Torch is required to load runtime seed tasks.")


def load_module_from_path(path: str | Path):
    file_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_tree_to_device(value: Any, device: str):
    if torch is not None and torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tree_to_device(inner, device) for key, inner in value.items()}
    return value


def _convert_tree_dtype(value: Any, dtype: torch.dtype | None):
    if dtype is None:
        return value
    if torch is not None and torch.is_tensor(value) and value.dtype.is_floating_point:
        return value.to(dtype=dtype)
    if isinstance(value, list):
        return [_convert_tree_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert_tree_dtype(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _convert_tree_dtype(inner, dtype) for key, inner in value.items()}
    return value


def _apply_layout(value: Any, layout: str | None):
    if layout is None:
        return value
    if torch is not None and torch.is_tensor(value):
        if layout == "channels_last" and value.dim() == 4:
            return value.contiguous(memory_format=torch.channels_last)
        if layout == "transposed" and value.dim() >= 2:
            return value.transpose(-1, -2).contiguous()
        if layout == "non_contiguous" and value.dim() >= 2:
            return value.transpose(-1, -2)
        return value.contiguous()
    if isinstance(value, list):
        return [_apply_layout(item, layout) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_layout(item, layout) for item in value)
    if isinstance(value, dict):
        return {key: _apply_layout(inner, layout) for key, inner in value.items()}
    return value


def _prepare_model(model, device: str, dtype: torch.dtype | None, layout: str | None):
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if layout == "channels_last":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return model


def load_seed_task(task_path: str | Path, device: str = "cpu", dtype: str | None = None, layout: str | None = None) -> dict[str, Any]:
    _require_torch()
    module = load_module_from_path(task_path)
    Model = getattr(module, "Model")
    get_inputs = getattr(module, "get_inputs")
    get_init_inputs = getattr(module, "get_init_inputs")

    init_inputs = list(get_init_inputs())

    dtype_map = {
        None: None,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    model = _prepare_model(Model(*init_inputs), device=device, dtype=torch_dtype, layout=layout)

    inputs = list(get_inputs())
    inputs = _move_tree_to_device(inputs, device)
    inputs = _convert_tree_dtype(inputs, torch_dtype)
    inputs = _apply_layout(inputs, layout)

    return {
        "module": module,
        "model": model,
        "inputs": inputs,
        "init_inputs": init_inputs,
        "task_name": Path(task_path).name,
        "task_path": str(Path(task_path).resolve()),
    }

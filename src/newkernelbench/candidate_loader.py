from __future__ import annotations

import copy
from pathlib import Path

from .task_loader import load_module_from_path

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependent
    torch = None


def _prepare_model(model, device: str, dtype, layout: str | None):
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if torch is not None and layout == "channels_last":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return model


def build_candidate_model(
    candidate_path: str | Path | None,
    reference_model,
    init_inputs: list,
    device: str,
    dtype,
    layout: str | None,
):
    notes: list[str] = []

    if candidate_path is None:
        notes.append("No candidate file provided; using a deepcopy of the reference model as the candidate.")
        return _prepare_model(copy.deepcopy(reference_model), device=device, dtype=dtype, layout=layout), notes

    module = load_module_from_path(candidate_path)

    if hasattr(module, "build_candidate_model"):
        candidate = module.build_candidate_model(reference_model=copy.deepcopy(reference_model), init_inputs=list(init_inputs))
        notes.append("Loaded candidate via build_candidate_model().")
        return _prepare_model(candidate, device=device, dtype=dtype, layout=layout), notes

    candidate_class = getattr(module, "ModelNew", None) or getattr(module, "Model", None)
    if candidate_class is None:
        raise RuntimeError(f"Candidate file {candidate_path} must define ModelNew, Model, or build_candidate_model().")

    candidate = candidate_class(*list(init_inputs))
    if hasattr(module, "copy_weights_from_reference"):
        module.copy_weights_from_reference(candidate, reference_model)
        notes.append("Weights copied via copy_weights_from_reference().")
    else:
        load_result = candidate.load_state_dict(reference_model.state_dict(), strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        notes.append(f"Attempted state_dict sync with strict=False. missing_keys={missing}, unexpected_keys={unexpected}")

    return _prepare_model(candidate, device=device, dtype=dtype, layout=layout), notes

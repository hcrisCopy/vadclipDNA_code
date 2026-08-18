"""Load and validate the frozen normality-circuit discovery artifact."""
from __future__ import annotations

from pathlib import Path

import torch


REQUIRED_KEYS = {
    "version", "dataset", "hidden_layers", "hidden_width", "selected_layers", "selected_dimensions",
    "context_centers", "state_mean", "state_std", "transition_mean", "transition_std",
    "ln_post_weight", "ln_post_bias", "ln_post_eps", "visual_projection", "text_features",
}


def load_assets(path: str | Path, device: torch.device | str = "cpu") -> dict:
    artifact = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path}: expected a CTNC assets dictionary")
    missing = REQUIRED_KEYS - set(artifact)
    if missing:
        raise ValueError(f"{path}: CTNC assets are missing {sorted(missing)}")
    layers = artifact["selected_layers"]
    dims = artifact["selected_dimensions"]
    if not isinstance(layers, torch.Tensor) or not isinstance(dims, torch.Tensor) or layers.ndim != 1 or dims.ndim != 1:
        raise ValueError(f"{path}: selected_layers and selected_dimensions must be one-dimensional tensors")
    if len(layers) == 0 or len(layers) != len(dims):
        raise ValueError(f"{path}: invalid sparse circuit dimension list")
    if int(artifact["hidden_width"]) != 768:
        raise ValueError(f"{path}: current CLIP contract requires hidden width 768")
    selected_width = len(layers)
    for key in ("state_mean", "state_std", "transition_mean", "transition_std"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != selected_width:
            raise ValueError(f"{path}: {key} must have shape [contexts,{selected_width}]")
    return artifact


def asset_selected_width(artifact: dict) -> int:
    return int(len(artifact["selected_layers"]))

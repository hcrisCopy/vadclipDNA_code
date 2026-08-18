"""Load and validate the frozen normality-circuit discovery artifact."""
from __future__ import annotations

from pathlib import Path

import torch


REQUIRED_KEYS = {
    "version", "dataset", "hidden_layers", "hidden_width", "selected_layers", "selected_dimensions",
    "selected_text_direction", "selected_text_class", "selected_text_affinity",
    "context_centers", "normal_prototypes",
    "global_normal_subspace_basis", "global_subspace_rank",
    "normal_variance", "normal_pca_coordinate_energy",
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
    for key in ("selected_text_direction", "selected_text_class"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) != selected_width:
            raise ValueError(f"{path}: {key} must have shape [{selected_width}]")
    affinity = artifact["selected_text_affinity"]
    if not isinstance(affinity, torch.Tensor) or affinity.ndim != 2 or affinity.shape[0] != selected_width:
        raise ValueError(f"{path}: selected_text_affinity must have shape [{selected_width}, anomaly_classes]")
    prototypes = artifact["normal_prototypes"]
    if (
        not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 3
        or prototypes.shape[0] != artifact["context_centers"].shape[0]
        or prototypes.shape[2] != selected_width
    ):
        raise ValueError(f"{path}: normal_prototypes must have shape [contexts,prototypes,{selected_width}]")
    hidden_layers = int(artifact["hidden_layers"])
    global_basis = artifact["global_normal_subspace_basis"]
    global_rank = int(artifact["global_subspace_rank"])
    if (
        not isinstance(global_basis, torch.Tensor)
        or global_basis.ndim != 3
        or tuple(global_basis.shape[:2]) != (hidden_layers, int(artifact["hidden_width"]))
        or global_basis.shape[2] != global_rank
        or global_rank <= 0
        or global_rank >= int(artifact["hidden_width"])
    ):
        raise ValueError(f"{path}: invalid all-channel normal SVD basis")
    for key in ("normal_variance", "normal_pca_coordinate_energy"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (hidden_layers, int(artifact["hidden_width"])):
            raise ValueError(f"{path}: {key} must contain one value per original hidden coordinate")
    return artifact


def asset_selected_width(artifact: dict) -> int:
    return int(len(artifact["selected_layers"]))

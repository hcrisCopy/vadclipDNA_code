"""Load and validate the frozen normality-circuit discovery artifact."""
from __future__ import annotations

from pathlib import Path

import torch


REQUIRED_KEYS = {
    "version", "dataset", "hidden_layers", "hidden_width", "selected_layers", "selected_dimensions",
    "selected_text_direction", "selected_text_class", "selected_text_affinity",
    "context_centers", "state_mean", "state_std", "transition_mean", "transition_std", "normal_prototypes",
    "normal_subspace_basis", "subspace_rank", "global_normal_subspace_basis", "global_subspace_rank", "frame_topk",
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
    for key in ("state_mean", "state_std", "transition_mean", "transition_std"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != selected_width:
            raise ValueError(f"{path}: {key} must have shape [contexts,{selected_width}]")
    prototypes = artifact["normal_prototypes"]
    if (
        not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 3
        or prototypes.shape[0] != artifact["state_mean"].shape[0]
        or prototypes.shape[2] != selected_width
    ):
        raise ValueError(f"{path}: normal_prototypes must have shape [contexts,prototypes,{selected_width}]")
    basis = artifact["normal_subspace_basis"]
    rank = int(artifact["subspace_rank"])
    hidden_layers = int(artifact["hidden_layers"])
    per_layer = selected_width // hidden_layers
    if (
        selected_width % hidden_layers != 0
        or not isinstance(basis, torch.Tensor) or basis.ndim != 4
        or tuple(basis.shape[:2]) != (artifact["state_mean"].shape[0], hidden_layers)
        or basis.shape[2] != per_layer or basis.shape[3] != rank
        or rank <= 0 or rank > per_layer
    ):
        raise ValueError(f"{path}: invalid per-context normal subspace basis")
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
    frame_topk = int(artifact["frame_topk"])
    anomaly_classes = int(affinity.shape[1])
    if frame_topk <= 0 or frame_topk > selected_width // max(1, anomaly_classes):
        raise ValueError(f"{path}: frame_topk must fit the selected witnesses of every anomaly text")
    return artifact


def asset_selected_width(artifact: dict) -> int:
    return int(len(artifact["selected_layers"]))

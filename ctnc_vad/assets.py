"""Load and validate the frozen normality-circuit discovery artifact."""
from __future__ import annotations

from pathlib import Path

import torch


REQUIRED_KEYS = {
    "version", "dataset", "hidden_layers", "hidden_width", "selected_layers", "selected_dimensions",
    "selected_text_direction", "selected_text_class", "selected_text_affinity",
    "context_centers", "state_mean", "state_std", "transition_mean", "transition_std", "normal_prototypes",
    "normal_video_signatures", "normal_video_visual_prototypes", "normal_video_neighbor_count",
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
    signatures = artifact["normal_video_signatures"]
    visual_prototypes = artifact["normal_video_visual_prototypes"]
    neighbor_count = int(artifact["normal_video_neighbor_count"])
    if (
        not isinstance(signatures, torch.Tensor) or signatures.ndim != 2 or signatures.shape[1] != 768
        or not isinstance(visual_prototypes, torch.Tensor) or visual_prototypes.ndim != 3
        or visual_prototypes.shape[0] != signatures.shape[0] or visual_prototypes.shape[2] != 512
        or neighbor_count <= 0 or neighbor_count > signatures.shape[0]
    ):
        raise ValueError(f"{path}: invalid normal-video visual counterfactual memory")
    return artifact


def asset_selected_width(artifact: dict) -> int:
    return int(len(artifact["selected_layers"]))

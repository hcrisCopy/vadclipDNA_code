"""Validation for CTSC discovery assets."""
from __future__ import annotations

from pathlib import Path

import torch


REQUIRED = {
    "version", "dataset", "prompts", "hidden_layers", "hidden_width",
    "selected_layers", "selected_dimensions", "semantic_response",
    "context_centers", "context_mean", "context_std", "normal_variance",
    "normal_pca_coordinate_energy", "context_temporal_mean", "context_temporal_std",
    "temporal_short_window", "temporal_long_window", "temporal_persistence_window",
}


def load_assets(path: str | Path, device: str | torch.device = "cpu") -> dict:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a CTSC discovery asset dictionary")
    missing = REQUIRED - set(value)
    if missing:
        raise ValueError(f"{path}: incompatible CTSC discovery assets; missing={sorted(missing)}")
    if int(value["version"]) != 2:
        raise ValueError(f"{path}: CTSC temporal circuit requires discovery assets version 2; rerun ctsc_vad.discover --clean")
    layers = value["selected_layers"]
    dimensions = value["selected_dimensions"]
    if (
        not isinstance(layers, torch.Tensor) or not isinstance(dimensions, torch.Tensor)
        or layers.ndim != 1 or dimensions.ndim != 1 or len(layers) != len(dimensions)
    ):
        raise ValueError(f"{path}: selected layer/dimension arrays are invalid")
    width = len(layers)
    classes = len(value["prompts"]) - 1
    if width == 0 or value["semantic_response"].shape != (width, classes):
        raise ValueError(f"{path}: semantic_response must be [selected_channels, anomaly_classes]")
    contexts = value["context_centers"].shape[0]
    for name in ("context_mean", "context_std"):
        if tuple(value[name].shape) != (contexts, width):
            raise ValueError(f"{path}: {name} must be [contexts, selected_channels]")
    for name in ("context_temporal_mean", "context_temporal_std"):
        if tuple(value[name].shape) != (contexts, width, 2):
            raise ValueError(f"{path}: {name} must be [contexts, selected_channels, 2]")
    if (value["context_std"] <= 0).any():
        raise ValueError(f"{path}: context_std must be positive")
    if (value["context_temporal_std"] <= 0).any():
        raise ValueError(f"{path}: context_temporal_std must be positive")
    short, long, persistence = (int(value[name]) for name in ("temporal_short_window", "temporal_long_window", "temporal_persistence_window"))
    if min(short, long, persistence) <= 0 or any(width % 2 == 0 for width in (short, long, persistence)) or short >= long:
        raise ValueError(f"{path}: temporal windows must be positive odd values with short < long")
    return value


def selected_width(assets: dict) -> int:
    return int(len(assets["selected_layers"]))

"""Reusable frozen-baseline score cache for CTNC training.

The cache is an input feature for the sidecar, never a pseudo-label source.
Keeping it under the CTNC output root makes training resumable and avoids
re-running the frozen VadCLIP model for every reader epoch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import score_sequence
from .common import atomic_save_npz


def score_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def valid_score(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = np.load(path, allow_pickle=False)
        try:
            return (
                "prob2" in artifact.files
                and "prob2_all" in artifact.files
                and len(artifact["prob2"]) == length
                and np.asarray(artifact["prob2_all"]).ndim == 2
                and len(artifact["prob2_all"]) == length
                and np.asarray(artifact["prob2_all"]).shape[1] >= 2
            )
        finally:
            artifact.close()
    except Exception:
        return False


@torch.no_grad()
def prepare_score_cache(
    dataset,
    baseline_model,
    visual_length: int,
    dataset_name: str,
    device: torch.device,
    cache_dir: Path,
    reuse: bool,
    progress: str,
) -> dict[str, Path]:
    """Cache frozen VadCLIP probabilities in the exact source-video order."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    baseline_model.eval()
    for item in tqdm(dataset, desc=progress, unit="video"):
        key = str(item["key"])
        length = int(item["length"])
        target = score_path(cache_dir, key)
        if not (reuse and valid_score(target, length)):
            probability1, probability2, probability2_all = score_sequence(
                baseline_model, item["clip_feature"].numpy(), visual_length, dataset_name, device
            )
            atomic_save_npz(target, prob1=probability1, prob2=probability2, prob2_all=probability2_all)
        result[key] = target
    return result


def load_cached_probabilities(path: Path, expected_length: int) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    try:
        value = np.asarray(artifact["prob2_all"], dtype=np.float32)
    finally:
        artifact.close()
    if value.ndim != 2 or len(value) != expected_length or value.shape[1] < 2:
        raise ValueError(f"{path}: expected cached [T,classes] prob2_all, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{path}: cached class probabilities are invalid")
    value = value / np.maximum(value.sum(axis=1, keepdims=True), 1e-6)
    return value

"""Resumable frozen-baseline probability cache; it is never a pseudo-label."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import score_sequence
from .common import atomic_save_npz


def _path(root: Path, key: str) -> Path:
    return root / f"{key}.npz"


def _valid(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, allow_pickle=False)
        try:
            return "prob2_all" in value.files and value["prob2_all"].ndim == 2 and len(value["prob2_all"]) == length
        finally:
            value.close()
    except Exception:
        return False


@torch.no_grad()
def prepare_cache(dataset, baseline, visual_length: int, dataset_name: str, device: torch.device, root: Path, reuse: bool) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    baseline.eval()
    for item in tqdm(dataset, desc="cache frozen VadCLIP train scores", unit="video"):
        key, length, target = str(item["key"]), int(item["length"]), _path(root, str(item["key"]))
        if not (reuse and _valid(target, length)):
            prob1, prob2, prob2_all = score_sequence(baseline, item["clip_feature"].numpy(), visual_length, dataset_name, device)
            atomic_save_npz(target, prob1=prob1, prob2=prob2, prob2_all=prob2_all)
        result[key] = target
    return result


def load_cached_probability(path: Path, length: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    try:
        probability = np.asarray(value["prob2_all"], dtype=np.float32)
    finally:
        value.close()
    if probability.ndim != 2 or len(probability) != length or probability.shape[1] < 2:
        raise ValueError(f"{path}: invalid cached baseline probability")
    return probability / np.maximum(probability.sum(axis=-1, keepdims=True), 1e-6)

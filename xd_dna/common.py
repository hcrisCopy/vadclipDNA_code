"""Repository-local utilities for the XD DNA-on-VadCLIP experiment.

This package never imports code from sibling projects. Inputs such as the
previously extracted CLIP hidden features are ordinary data files supplied via
command-line paths; every artifact created here is kept under
``../vadclipDNA_data`` by default.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


XD_LABELS = {
    "A": "normal",
    "B1": "fighting",
    "B2": "shooting",
    "B4": "riot",
    "B5": "abuse",
    "B6": "car accident",
    "G": "explosion",
}
CHUNK_SUFFIX = re.compile(r"__(\d+)$")


def default_output_root() -> Path:
    return Path("../vadclipDNA_data/xd_normal_negative_top64")


def output_root(path: str | Path) -> Path:
    """Keep every generated artifact under the sibling vadclipDNA_data root."""
    root = Path(path).expanduser().resolve()
    code_root = Path(__file__).resolve().parents[1]
    data_root = code_root.parent / "vadclipDNA_data"
    if root != data_root and data_root not in root.parents:
        raise ValueError("outputs must stay under the sibling vadclipDNA_data directory")
    return root


def stage_dir(root: str | Path, name: str, clean: bool = False) -> Path:
    """Create one named output stage, optionally replacing only that stage."""
    base = output_root(root)
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError(f"invalid output stage: {name!r}")
    if clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array)
    temporary.replace(target)


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(target)


def atomic_torch_save(path: str | Path, content: object) -> None:
    """Atomically write a PyTorch checkpoint without leaving a partial model."""
    import torch

    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(content, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(path: str | Path, content: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp",
        encoding="utf-8", delete=False,
    ) as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp",
        encoding="utf-8", newline="", delete=False,
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(target)


def read_path_label_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return frame


def feature_key(path_or_key: str) -> str:
    return Path(str(path_or_key)).stem


def base_key(path_or_key: str) -> str:
    return CHUNK_SUFFIX.sub("", feature_key(path_or_key))


def chunk_index(path_or_key: str) -> int:
    matched = CHUNK_SUFFIX.search(feature_key(path_or_key))
    return int(matched.group(1)) if matched else 0


def is_pure_normal_xd(label: str) -> bool:
    """Only official XD label A is admitted as a negative source."""
    return str(label).strip() == "A"


def grouped_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["_base_key"] = work["path"].map(base_key)
    work["_chunk_index"] = work["path"].map(chunk_index)
    groups: dict[str, pd.DataFrame] = {}
    for key, group in work.groupby("_base_key", sort=True):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError(f"video {key!r} has inconsistent labels: {sorted(labels)}")
        groups[str(key)] = group.sort_values("_chunk_index")
    return groups


def rewrite_prefix(path: str, prefix_from: str = "", prefix_to: str = "") -> str:
    if prefix_from and str(path).startswith(prefix_from):
        return prefix_to + str(path)[len(prefix_from):]
    return str(path)


def relpath(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def resolve_recorded_path(path: str | Path, relative_to: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (Path(relative_to) / candidate).resolve()


def load_clip_feature(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        try:
            key = "features" if "features" in artifact.files else artifact.files[0]
            value = artifact[key]
        finally:
            artifact.close()
    else:
        value = artifact
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [T,D] feature array, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: feature contains non-finite values")
    return result


def load_hidden(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    artifact = np.load(path, allow_pickle=False)
    if not isinstance(artifact, np.lib.npyio.NpzFile) or "hidden" not in artifact.files:
        raise ValueError(f"{path}: expected .npz with a 'hidden' array")
    try:
        hidden = np.asarray(artifact["hidden"], dtype=np.float32)
        metadata: dict[str, object] = {}
        for key in artifact.files:
            if key == "hidden":
                continue
            value = artifact[key]
            metadata[key] = value.item() if value.shape == () else value
    finally:
        artifact.close()
    if hidden.ndim != 3 or hidden.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [T,L,D] hidden, got {hidden.shape}")
    if not np.isfinite(hidden).all():
        raise ValueError(f"{path}: hidden features contain non-finite values")
    return hidden, metadata


def manifest_hidden_paths(
    path: str | Path,
    prefix_from: str = "",
    prefix_to: str = "",
) -> tuple[dict[str, str], str]:
    """Read a staged hidden manifest without relying on the project that made it."""
    manifest_path = Path(path).resolve()
    frame = pd.read_csv(manifest_path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path}: hidden manifest is missing {sorted(missing)}")
    if "token_pool" in frame.columns:
        pools = set(frame["token_pool"].astype(str))
        if pools != {"cls"}:
            raise ValueError(f"{manifest_path}: only CLS hidden features are supported, got {sorted(pools)}")
    hidden_by_key: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        key, hidden_path = str(row.key), rewrite_prefix(str(row.hidden_path), prefix_from, prefix_to)
        hidden_path = str(resolve_recorded_path(hidden_path, manifest_path.parent))
        if key in hidden_by_key and hidden_by_key[key] != hidden_path:
            raise ValueError(f"{manifest_path}: duplicate key with different hidden paths: {key}")
        hidden_by_key[key] = hidden_path
    if not hidden_by_key:
        raise RuntimeError(f"{manifest_path}: no hidden entries")
    return hidden_by_key, "cls"


def deterministic_split(keys: list[str], validation_fraction: float, rng: np.random.Generator) -> dict[str, str]:
    if len(keys) < 2:
        raise ValueError("at least two videos are required for a train/validation split")
    ordered = np.asarray(sorted(keys), dtype=object)
    shuffled = ordered[rng.permutation(len(ordered))]
    validation_size = min(max(1, int(round(len(ordered) * validation_fraction))), len(ordered) - 1)
    validation = set(str(value) for value in shuffled[:validation_size])
    return {str(key): ("validation" if str(key) in validation else "train") for key in ordered}


def uniform_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    count = min(max(1, int(count)), int(length))
    return np.linspace(0, length - 1, count, dtype=np.int64)


def resample_scores(scores: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(values) == target_length:
        return values
    if len(values) == 0:
        raise ValueError("cannot resample an empty pseudo-score sequence")
    return np.interp(
        np.linspace(0.0, 1.0, target_length, dtype=np.float32),
        np.linspace(0.0, 1.0, len(values), dtype=np.float32),
        values,
    ).astype(np.float32)


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

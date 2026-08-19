"""Repository-local IO, labels, and safety helpers for CTSC-VAD.

This package reads VadCLIP only as an unchanged baseline.  It never imports
code from sibling projects and writes every new artifact below the sibling
``../vadclipDNA_data`` directory.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


XD_LABELS = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}
UCF_LABELS = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "road accidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}
CHUNK_SUFFIX = re.compile(r"__(\d+)$")


def labels_for_dataset(dataset: str) -> dict[str, str]:
    if dataset == "xd":
        return XD_LABELS
    if dataset == "ucf":
        return UCF_LABELS
    raise ValueError(f"unsupported dataset={dataset!r}")


def normal_label(dataset: str) -> str:
    return "A" if dataset == "xd" else "Normal" if dataset == "ucf" else _unsupported(dataset)


def _unsupported(dataset: str):
    raise ValueError(f"unsupported dataset={dataset!r}")


def is_normal_video(dataset: str, label: str) -> bool:
    value = str(label).strip()
    return value == "A" or value.split("-", 1)[0] == "A" if dataset == "xd" else value == "Normal"


def default_output_root(dataset: str) -> Path:
    if dataset not in {"xd", "ucf"}:
        raise ValueError(f"unsupported dataset={dataset!r}")
    return Path(f"../vadclipDNA_data/{dataset}_ctsc_vad")


def ensure_output_root(path: str | Path) -> Path:
    target = Path(path).resolve()
    package_root = Path(__file__).resolve().parents[1]
    data_root = (package_root.parent / "vadclipDNA_data").resolve()
    if target != data_root and data_root not in target.parents:
        raise ValueError("--output-root must stay under sibling ../vadclipDNA_data")
    return target


def stage_dir(root: str | Path, name: str, clean: bool = False) -> Path:
    base = ensure_output_root(root)
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError(f"invalid stage name {name!r}")
    if clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_target(path: str | Path, suffix: str):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=suffix, delete=False)
    return target, handle


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target, handle = _atomic_target(path, ".tmp")
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        Path(handle.name).replace(target)
    finally:
        temporary = Path(handle.name)
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: str | Path, content: object) -> None:
    import torch

    target, handle = _atomic_target(path, ".tmp")
    handle.close()
    try:
        torch.save(content, handle.name)
        Path(handle.name).replace(target)
    finally:
        temporary = Path(handle.name)
        if temporary.exists():
            temporary.unlink()


def save_json(path: str | Path, content: dict) -> None:
    target, handle = _atomic_target(path, ".tmp")
    try:
        with handle:
            handle.write(json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8"))
            handle.write(b"\n")
        Path(handle.name).replace(target)
    finally:
        temporary = Path(handle.name)
        if temporary.exists():
            temporary.unlink()


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    temporary.replace(target)


def relpath(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def base_key(path_or_key: str | Path) -> str:
    return CHUNK_SUFFIX.sub("", Path(str(path_or_key)).stem)


def chunk_index(path_or_key: str | Path) -> int:
    found = CHUNK_SUFFIX.search(Path(str(path_or_key)).stem)
    return int(found.group(1)) if found else 0


def resolve_path(recorded: str | Path, base: str | Path) -> Path:
    candidate = Path(recorded)
    return candidate if candidate.is_absolute() else (Path(base) / candidate).resolve()


def read_source_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{path}: source CSV must be non-empty with path,label columns")
    return frame


def grouped_source_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["_ctsc_key"] = work["path"].map(base_key)
    work["_ctsc_chunk"] = work["path"].map(chunk_index)
    result: dict[str, pd.DataFrame] = {}
    for key, group in work.groupby("_ctsc_key", sort=False):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError(f"video {key!r}: inconsistent labels {sorted(labels)}")
        result[str(key)] = group.sort_values("_ctsc_chunk").reset_index(drop=True)
    return result


def hidden_manifest_paths(path: str | Path, path_base: str | Path, prefix_from: str = "", prefix_to: str = "") -> dict[str, Path]:
    frame = pd.read_csv(path)
    if {"key", "hidden_path"} - set(frame.columns):
        raise ValueError(f"{path}: hidden manifest needs key,hidden_path columns")
    if "token_pool" in frame.columns and set(frame["token_pool"].astype(str)) != {"cls"}:
        raise ValueError("CTSC-VAD requires CLS hidden states")
    result: dict[str, Path] = {}
    for row in frame.itertuples(index=False):
        recorded = str(row.hidden_path)
        if prefix_from and recorded.startswith(prefix_from):
            recorded = prefix_to + recorded[len(prefix_from):]
        key, resolved = str(row.key), resolve_path(recorded, path_base)
        if key in result and result[key] != resolved:
            raise ValueError(f"{path}: duplicate hidden key {key!r}")
        result[key] = resolved
    if not result:
        raise ValueError(f"{path}: no hidden artifacts")
    return result


def load_hidden(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    if not isinstance(artifact, np.lib.npyio.NpzFile) or "hidden" not in artifact.files:
        raise ValueError(f"{path}: expected .npz containing hidden")
    try:
        value = np.asarray(artifact["hidden"], dtype=np.float32)
    finally:
        artifact.close()
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1:] != (12, 768) or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected finite [T,12,768] hidden state, got {value.shape}")
    return value


def load_clip_feature(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        try:
            value = artifact["features"] if "features" in artifact.files else artifact[artifact.files[0]]
        finally:
            artifact.close()
    else:
        value = artifact
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 512 or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected finite [T,512] feature, got {value.shape}")
    return value


def align_hidden(hidden: np.ndarray, target_length: int, policy: str) -> np.ndarray:
    if len(hidden) == target_length:
        return hidden
    if len(hidden) > target_length and policy == "crop_hidden":
        return hidden[:target_length]
    if len(hidden) < target_length and policy == "pad_hidden":
        return np.concatenate([hidden, np.repeat(hidden[-1:], target_length - len(hidden), axis=0)], axis=0)
    raise ValueError(f"hidden/feature length mismatch ({len(hidden)} vs {target_length}); check extraction or --alignment")


def resample_for_train(values: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP process_feat bin-mean/padding exactly for time-major data."""
    length = len(values)
    if values.ndim < 2 or length == 0:
        raise ValueError(f"expected non-empty time-major values, got {values.shape}")
    if length > visual_length:
        edges = np.linspace(0, length, visual_length + 1, dtype=np.int64)
        result = np.empty((visual_length, *values.shape[1:]), dtype=np.float32)
        for index in range(visual_length):
            left, right = int(edges[index]), int(edges[index + 1])
            result[index] = values[left:right].mean(axis=0) if right > left else values[left]
        return result, visual_length
    if length < visual_length:
        return np.concatenate([values.astype(np.float32), np.zeros((visual_length - length, *values.shape[1:]), dtype=np.float32)]), length
    return values.astype(np.float32), length


def video_label_vector(dataset: str, label: str) -> np.ndarray:
    names = list(labels_for_dataset(dataset))
    result = np.zeros(len(names), dtype=np.float32)
    value = str(label).strip()
    if dataset == "xd":
        for part in value.split("-"):
            if part in names:
                result[names.index(part)] = 1.0
    elif value in names:
        result[names.index(value)] = 1.0
    if not result.any():
        raise ValueError(f"cannot map {dataset} label {label!r}")
    return result


def normalize_rows(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def kmeans_unit_vectors(values: np.ndarray, clusters: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = normalize_rows(np.asarray(values, dtype=np.float32))
    if data.ndim != 2 or not len(data):
        raise ValueError("k-means expects non-empty [N,D] vectors")
    count = min(max(1, int(clusters)), len(data))
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(len(data), count, replace=False)].copy()
    assignments = np.full(len(data), -1, dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        next_assignments = np.argmax(data @ centers.T, axis=1).astype(np.int64)
        if np.array_equal(next_assignments, assignments):
            break
        assignments = next_assignments
        for index in range(count):
            members = data[assignments == index]
            centers[index] = members.mean(axis=0) if len(members) else data[int(rng.integers(len(data)))]
        centers = normalize_rows(centers)
    return centers.astype(np.float32), assignments


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

"""Shared, repository-local helpers for CTNC-VAD.

No code is imported from sibling VAD projects or from ``xd_dna``.  Existing
CLIP hidden states and 512D VadCLIP features are treated as data artifacts.
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
UCF_LABELS = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "road accidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}
CHUNK_SUFFIX = re.compile(r"__(\d+)$")


def default_output_root(dataset: str) -> Path:
    if dataset not in {"xd", "ucf"}:
        raise ValueError(f"unsupported dataset={dataset!r}")
    return Path(f"../vadclipDNA_data/{dataset}_ctnc_vad")


def ensure_output_root(path: str | Path) -> Path:
    """Keep generated files under the sibling ``vadclipDNA_data`` directory."""
    target = Path(path).resolve()
    code_root = Path(__file__).resolve().parents[1]
    data_root = (code_root.parent / "vadclipDNA_data").resolve()
    if target != data_root and data_root not in target.parents:
        raise ValueError("--output-root must stay under the sibling ../vadclipDNA_data directory")
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


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(target)


def atomic_torch_save(path: str | Path, content: object) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False
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
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp", delete=False
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(target)


def relpath(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(start).resolve())


def resolve_path(recorded: str | Path, base: str | Path) -> Path:
    candidate = Path(recorded)
    return candidate if candidate.is_absolute() else (Path(base) / candidate).resolve()


def base_key(path_or_key: str | Path) -> str:
    return CHUNK_SUFFIX.sub("", Path(str(path_or_key)).stem)


def chunk_index(path_or_key: str | Path) -> int:
    found = CHUNK_SUFFIX.search(Path(str(path_or_key)).stem)
    return int(found.group(1)) if found else 0


def normal_label(dataset: str) -> str:
    if dataset == "xd":
        return "A"
    if dataset == "ucf":
        return "Normal"
    raise ValueError(f"unsupported dataset={dataset!r}")


def is_normal_video(dataset: str, label: str) -> bool:
    """Accept both local XD label ``A`` and official chunked label ``A-0-0``."""
    value = str(label).strip()
    if dataset == "xd":
        return value == "A" or value.split("-", 1)[0] == "A"
    if dataset == "ucf":
        return value == "Normal"
    raise ValueError(f"unsupported dataset={dataset!r}")


def labels_for_dataset(dataset: str) -> dict[str, str]:
    if dataset == "xd":
        return XD_LABELS
    if dataset == "ucf":
        return UCF_LABELS
    raise ValueError(f"unsupported dataset={dataset!r}")


def video_label_vector(dataset: str, label: str) -> np.ndarray:
    """Return the official video-level multi-class target in prompt order.

    XD labels may contain multiple anomaly types (for example ``G-B2-B6``).
    This is supervision supplied by the dataset, not a label inferred from a
    VAD baseline prediction.
    """
    names = list(labels_for_dataset(dataset))
    result = np.zeros(len(names), dtype=np.float32)
    value = str(label).strip()
    if dataset == "xd":
        parts = value.split("-")
        if parts and parts[0] == "A":
            result[0] = 1.0
        else:
            for part in parts:
                if part in names:
                    result[names.index(part)] = 1.0
    elif dataset == "ucf":
        if value in names:
            result[names.index(value)] = 1.0
    else:
        raise ValueError(f"unsupported dataset={dataset!r}")
    if not result.any():
        raise ValueError(f"{dataset}: cannot map video label {label!r} to a prompt target")
    return result


def read_source_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{csv_path}: source CSV is empty")
    return frame


def grouped_source_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["_ctnc_key"] = work["path"].map(base_key)
    work["_ctnc_chunk"] = work["path"].map(chunk_index)
    groups: dict[str, pd.DataFrame] = {}
    # Evaluation labels in VadCLIP's gt.npy are concatenated in the source
    # test CSV order.  Never sort by video key here: doing so silently
    # misaligns every prediction with its frame-level ground truth.
    # ``sort=False`` still groups XD's __0 ... __9 training augmentations,
    # while preserving the first-occurrence order of individual test videos.
    for key, group in work.groupby("_ctnc_key", sort=False):
        labels = set(group["label"].astype(str))
        if len(labels) != 1:
            raise ValueError(f"video {key!r} has inconsistent labels {sorted(labels)}")
        groups[str(key)] = group.sort_values("_ctnc_chunk").reset_index(drop=True)
    return groups


def load_clip_feature(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        try:
            value = artifact["features"] if "features" in artifact.files else artifact[artifact.files[0]]
        finally:
            artifact.close()
    else:
        value = artifact
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 512:
        raise ValueError(f"{path}: expected non-empty [T,512] CLIP feature, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: CLIP feature contains non-finite values")
    return result


def load_hidden(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=False)
    if not isinstance(artifact, np.lib.npyio.NpzFile) or "hidden" not in artifact.files:
        raise ValueError(f"{path}: expected an .npz artifact containing 'hidden'")
    try:
        hidden = np.asarray(artifact["hidden"], dtype=np.float32)
    finally:
        artifact.close()
    if hidden.ndim != 3 or hidden.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [T,L,D] hidden state, got {hidden.shape}")
    if not np.isfinite(hidden).all():
        raise ValueError(f"{path}: hidden state contains non-finite values")
    return hidden


def hidden_manifest_paths(
    path: str | Path,
    path_base: str | Path,
    prefix_from: str = "",
    prefix_to: str = "",
) -> dict[str, Path]:
    manifest = Path(path)
    frame = pd.read_csv(manifest)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest}: missing required columns {sorted(missing)}")
    if "token_pool" in frame.columns and set(frame["token_pool"].astype(str)) != {"cls"}:
        raise ValueError(f"{manifest}: CTNC-VAD currently requires CLS hidden states")
    result: dict[str, Path] = {}
    for row in frame.itertuples(index=False):
        recorded = str(row.hidden_path)
        if prefix_from and recorded.startswith(prefix_from):
            recorded = prefix_to + recorded[len(prefix_from):]
        resolved = resolve_path(recorded, path_base)
        key = str(row.key)
        if key in result and result[key] != resolved:
            raise ValueError(f"{manifest}: duplicate key {key!r} with different hidden paths")
        result[key] = resolved
    if not result:
        raise ValueError(f"{manifest}: no hidden artifacts")
    return result


def align_hidden(hidden: np.ndarray, target_length: int, policy: str) -> tuple[np.ndarray, str]:
    if len(hidden) == target_length:
        return hidden, "exact"
    if len(hidden) > target_length and policy == "crop_hidden":
        return hidden[:target_length], "crop_hidden"
    if len(hidden) < target_length and policy == "pad_hidden":
        return np.concatenate([hidden, np.repeat(hidden[-1:], target_length - len(hidden), axis=0)], axis=0), "pad_hidden"
    raise ValueError(
        f"temporal length mismatch: hidden={len(hidden)}, feature={target_length}; "
        "use --alignment crop_hidden or pad_hidden only after checking the extraction protocol"
    )


def resample_for_train(values: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP ``process_feat``: bin-average long videos and zero-pad short ones."""
    if values.ndim < 2 or len(values) == 0:
        raise ValueError(f"expected non-empty time-major data, got {values.shape}")
    length = len(values)
    if length > visual_length:
        edges = np.linspace(0, length, visual_length + 1, dtype=np.int64)
        output = np.empty((visual_length, *values.shape[1:]), dtype=np.float32)
        for index in range(visual_length):
            left, right = int(edges[index]), int(edges[index + 1])
            output[index] = values[left:right].mean(axis=0) if right > left else values[left]
        return output, visual_length
    if length < visual_length:
        padding = np.zeros((visual_length - length, *values.shape[1:]), dtype=np.float32)
        return np.concatenate([values.astype(np.float32), padding], axis=0), length
    return values.astype(np.float32), length


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kmeans_unit_vectors(vectors: np.ndarray, clusters: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic cosine k-means without another project dependency."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("k-means requires a non-empty [N,D] matrix")
    clusters = min(max(1, int(clusters)), len(values))
    rng = np.random.default_rng(seed)
    initial = rng.choice(len(values), size=clusters, replace=False)
    centers = normalize_rows(values[initial])
    assignment = np.full(len(values), -1, dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        next_assignment = np.argmax(values @ centers.T, axis=1).astype(np.int64)
        if np.array_equal(next_assignment, assignment):
            break
        assignment = next_assignment
        for cluster in range(clusters):
            members = values[assignment == cluster]
            if len(members) == 0:
                centers[cluster] = values[int(rng.integers(len(values)))]
            else:
                centers[cluster] = members.mean(axis=0)
        centers = normalize_rows(centers)
    return centers.astype(np.float32), assignment


def normalize_rows(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)

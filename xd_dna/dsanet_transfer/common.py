"""Contracts and safe I/O shared by the DSANet-to-VadCLIP transfer stages.

This package consumes DSANet artifacts strictly as data.  It never imports a
DSANet model or source module, which keeps the experiment independent from the
DSANet training implementation while retaining an auditable artifact trail.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..common import base_key, output_root, relpath, rewrite_prefix


EXPECTED_CLIP_MODEL = "ViT-B/16"
EXPECTED_TOKEN_POOL = "cls"
EXPECTED_LAYER_COUNT = 12
EXPECTED_HIDDEN_DIM = 768
CLIP_FEATURE_DIM = 512
NORMAL_XD_LABEL = "A"


@dataclass(frozen=True)
class FDURecord:
    """One DSANet-exported FDU sequence matched to a VadCLIP list row."""

    video_key: str
    variant: str
    label: str
    fdu_path: Path


def default_output_root() -> Path:
    """Default output location, relative to the vadclipDNA_code launch root."""
    return Path("../vadclipDNA_data/xd_dsanet_neuron_transfer")


def sha256_file(path: str | Path) -> str:
    """Return a content fingerprint without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fdu_fingerprint(fdus: list[dict]) -> str:
    """Match DSANet's FDU fingerprint definition exactly."""
    return canonical_json_sha256(fdus)


def feature_variant(path_or_variant: str) -> str:
    return Path(str(path_or_variant)).stem


def resolve_recorded_path(
    recorded_path: str | Path,
    path_base: str | Path,
    prefix_from: str = "",
    prefix_to: str = "",
) -> Path:
    """Resolve a portable or legacy recorded path without modifying its source file."""
    rewritten = rewrite_prefix(str(recorded_path), prefix_from, prefix_to)
    candidate = Path(rewritten)
    return candidate if candidate.is_absolute() else (Path(path_base) / candidate).resolve()


def load_json_object(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value


def validate_dsanet_fdu_spec(specification: dict, source: str | Path) -> list[dict]:
    """Validate that a DSANet FDU list can address VadCLIP's ViT-B/16 hidden state."""
    required = {"clip_model", "token_pool", "num_fdus", "fdus"}
    missing = required - set(specification)
    if missing:
        raise ValueError(f"{source}: DSANet FDU JSON is missing {sorted(missing)}")
    if str(specification["clip_model"]) != EXPECTED_CLIP_MODEL:
        raise ValueError(
            f"{source}: requires clip_model={EXPECTED_CLIP_MODEL!r}, got {specification['clip_model']!r}"
        )
    if str(specification["token_pool"]) != EXPECTED_TOKEN_POOL:
        raise ValueError(
            f"{source}: only {EXPECTED_TOKEN_POOL!r} FDU features are compatible, got {specification['token_pool']!r}"
        )
    fdus = specification["fdus"]
    if not isinstance(fdus, list) or not fdus:
        raise ValueError(f"{source}: fdus must be a non-empty list")
    if int(specification["num_fdus"]) != len(fdus):
        raise ValueError(f"{source}: num_fdus does not match the FDU list length")

    seen: set[tuple[int, int]] = set()
    checked: list[dict] = []
    for index, item in enumerate(fdus):
        if not isinstance(item, dict) or {"layer", "dimension"} - set(item):
            raise ValueError(f"{source}: FDU #{index} must contain layer and dimension")
        layer, dimension = int(item["layer"]), int(item["dimension"])
        if not 1 <= layer <= EXPECTED_LAYER_COUNT:
            raise ValueError(f"{source}: FDU #{index} has invalid one-based layer={layer}")
        if not 0 <= dimension < EXPECTED_HIDDEN_DIM:
            raise ValueError(f"{source}: FDU #{index} has invalid dimension={dimension}")
        if (layer, dimension) in seen:
            raise ValueError(f"{source}: duplicate FDU (layer={layer}, dimension={dimension})")
        seen.add((layer, dimension))
        checked.append(dict(item))
    return checked


def validate_transfer_contract(contract: dict, source: str | Path) -> None:
    required = {
        "format_version", "method", "clip_model", "token_pool", "source_fdu_fingerprint",
        "fdus", "neuron_width", "clip_dim", "input_width",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"{source}: transfer contract is missing {sorted(missing)}")
    fdus = validate_dsanet_fdu_spec(
        {
            "clip_model": contract["clip_model"],
            "token_pool": contract["token_pool"],
            "num_fdus": len(contract["fdus"]),
            "fdus": contract["fdus"],
        },
        source,
    )
    if str(contract["source_fdu_fingerprint"]) != fdu_fingerprint(fdus):
        raise ValueError(f"{source}: source_fdu_fingerprint does not match fdus")
    width = len(fdus)
    if int(contract["neuron_width"]) != width:
        raise ValueError(f"{source}: neuron_width={contract['neuron_width']} but FDU width is {width}")
    if int(contract["clip_dim"]) != CLIP_FEATURE_DIM:
        raise ValueError(f"{source}: VadCLIP clip_dim must be {CLIP_FEATURE_DIM}")
    if int(contract["input_width"]) != width + CLIP_FEATURE_DIM:
        raise ValueError(f"{source}: input_width must equal neuron_width + {CLIP_FEATURE_DIM}")


def load_transfer_contract(path: str | Path) -> dict:
    contract = load_json_object(path)
    validate_transfer_contract(contract, path)
    return contract


def load_fdu_feature(path: str | Path, expected_width: int) -> np.ndarray:
    """Load one DSANet FDU feature and reject incomplete or incompatible files."""
    source = Path(path)
    value = np.load(source, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        value.close()
        raise ValueError(f"{source}: expected DSANet .npy FDU feature, not .npz")
    feature = np.asarray(value, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] == 0 or feature.shape[1] != expected_width:
        raise ValueError(f"{source}: expected non-empty [T,{expected_width}] FDU feature, got {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError(f"{source}: FDU feature contains NaN or infinity")
    return feature


def feature_file_signature(path: str | Path) -> tuple[int, int]:
    info = Path(path).stat()
    return int(info.st_size), int(info.st_mtime_ns)


def source_fingerprint(records: Iterable[FDURecord]) -> str:
    """Fingerprint file identities without placing machine-specific paths in metadata."""
    entries = []
    for record in sorted(records, key=lambda item: (item.video_key, item.variant)):
        size, mtime_ns = feature_file_signature(record.fdu_path)
        entries.append({
            "video_key": record.video_key,
            "variant": record.variant,
            "label": record.label,
            "size": size,
            "mtime_ns": mtime_ns,
        })
    return canonical_json_sha256(entries)


def output_path(path: str | Path) -> Path:
    """Validate the user-provided root against the project-wide data-output rule."""
    return output_root(path)


def relative_metadata_path(path: str | Path, metadata_parent: str | Path) -> str:
    return relpath(path, metadata_parent)


def read_vadclip_list(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: VadCLIP list is missing {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: VadCLIP list is empty")
    frame = frame[["path", "label"]].copy()
    frame["path"] = frame["path"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame["variant"] = frame["path"].map(feature_variant)
    if frame["variant"].duplicated().any():
        duplicates = frame.loc[frame["variant"].duplicated(), "variant"].head(3).tolist()
        raise ValueError(f"{path}: duplicate feature variants, e.g. {duplicates}")
    frame["video_key"] = frame["path"].map(base_key)
    return frame


def read_dsanet_manifest(
    manifest_path: str | Path,
    fdu_path_base: str | Path,
    fdu_dir: str | Path = "",
    fdu_prefix_from: str = "",
    fdu_prefix_to: str = "",
) -> dict[str, FDURecord]:
    """Read DSANet's aligned FDU manifest without importing DSANet code."""
    manifest = Path(manifest_path).resolve()
    frame = pd.read_csv(manifest)
    required = {"video_key", "variant", "label", "fdu_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest}: DSANet FDU manifest is missing {sorted(missing)}")
    explicit_dir = Path(fdu_dir).resolve() if str(fdu_dir) else None
    records: dict[str, FDURecord] = {}
    for row in frame.itertuples(index=False):
        key, variant, label = str(row.video_key), str(row.variant), str(row.label)
        path = (
            explicit_dir / f"{key}.npy"
            if explicit_dir is not None
            else resolve_recorded_path(str(row.fdu_path), fdu_path_base, fdu_prefix_from, fdu_prefix_to)
        )
        record = FDURecord(video_key=key, variant=variant, label=label, fdu_path=path)
        if variant in records:
            previous = records[variant]
            if previous != record:
                raise ValueError(f"{manifest}: duplicate variant with different FDU records: {variant}")
            continue
        records[variant] = record
    if not records:
        raise ValueError(f"{manifest}: no FDU records")
    return records


def match_records(source: pd.DataFrame, fdu_records: dict[str, FDURecord]) -> tuple[list[FDURecord], list[tuple[str, str, str]]]:
    """Match by full chunk variant, never merely by a lossy video stem."""
    matched: list[FDURecord] = []
    missing: list[tuple[str, str, str]] = []
    for row in source.itertuples(index=False):
        record = fdu_records.get(str(row.variant))
        if record is None:
            missing.append((str(row.path), str(row.label), str(row.variant)))
            continue
        if record.video_key != str(row.video_key):
            raise ValueError(
                f"variant {row.variant}: DSANet key {record.video_key!r} differs from VadCLIP key {row.video_key!r}"
            )
        if record.label != str(row.label):
            raise ValueError(
                f"variant {row.variant}: DSANet label {record.label!r} differs from VadCLIP label {row.label!r}"
            )
        matched.append(record)
    return matched, missing


def validate_dsanet_export_spec(manifest_path: str | Path, contract: dict) -> None:
    """Require the FDU manifest to have been exported from the exact transferred list."""
    spec_path = Path(manifest_path).resolve().parent / "export_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{spec_path}: DSANet export_spec.json is required to prove the FDU feature contract"
        )
    spec = load_json_object(spec_path)
    expected_width = int(contract["neuron_width"])
    if int(spec.get("fdu_dim", -1)) != expected_width:
        raise ValueError(f"{spec_path}: fdu_dim does not match transfer contract")
    if str(spec.get("fdu_fingerprint", "")) != str(contract["source_fdu_fingerprint"]):
        raise ValueError(f"{spec_path}: fdu_fingerprint does not match transferred DSANet neurons")
    if str(spec.get("token_pool", EXPECTED_TOKEN_POOL)) != EXPECTED_TOKEN_POOL:
        raise ValueError(f"{spec_path}: token_pool is not {EXPECTED_TOKEN_POOL!r}")


def remove_tree(path: Path) -> None:
    """Delete only a caller-validated, stage-local output directory."""
    if not path.exists():
        return
    import shutil

    shutil.rmtree(path)



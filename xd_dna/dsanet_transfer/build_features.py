"""Fuse reusable DSANet FDUs with the original 512D VadCLIP feature lists.

No VadCLIP score is used in this stage.  The only learned selection is the
one already recorded in DSANet's ``fdu_indices.json``.  DSANet's exported FDU
arrays are checked against that exact selection before they are reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..common import atomic_save_npy, atomic_save_npz, load_json, save_json, write_csv
from .common import (
    CLIP_FEATURE_DIM,
    NORMAL_XD_LABEL,
    FDURecord,
    default_output_root,
    feature_file_signature,
    load_fdu_feature,
    load_transfer_contract,
    match_records,
    output_path,
    read_dsanet_manifest,
    read_vadclip_list,
    relative_metadata_path,
    remove_tree,
    resolve_recorded_path,
    sha256_file,
    source_fingerprint,
    validate_dsanet_export_spec,
)


def _valid_feature(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            value.close()
            return False
        return value.shape == expected_shape and bool(np.isfinite(value).all())
    except (OSError, ValueError, EOFError):
        return False


def _shard_path(shard_root: Path, record: FDURecord) -> Path:
    # FDU keys can contain characters that are inconvenient in file names; the
    # feature identity is also stored in the shard payload for collision-proof validation.
    import hashlib

    digest = hashlib.sha256(record.video_key.encode("utf-8")).hexdigest()
    return shard_root / f"{digest}.npz"


def _valid_stat_shard(path: Path, record: FDURecord, width: int) -> bool:
    if not path.is_file():
        return False
    try:
        size, mtime_ns = feature_file_signature(record.fdu_path)
        archive = np.load(path, allow_pickle=False)
        try:
            required = {"count", "sum", "sum_sq", "source_size", "source_mtime_ns", "fdu_width"}
            if not required.issubset(set(archive.files)):
                return False
            count = int(np.asarray(archive["count"]).item())
            saved_width = int(np.asarray(archive["fdu_width"]).item())
            saved_size = int(np.asarray(archive["source_size"]).item())
            saved_mtime = int(np.asarray(archive["source_mtime_ns"]).item())
            total = np.asarray(archive["sum"], dtype=np.float64)
            total_sq = np.asarray(archive["sum_sq"], dtype=np.float64)
        finally:
            archive.close()
        return (
            count > 0
            and saved_width == width
            and saved_size == size
            and saved_mtime == mtime_ns
            and total.shape == (width,)
            and total_sq.shape == (width,)
            and np.isfinite(total).all()
            and np.isfinite(total_sq).all()
        )
    except (OSError, ValueError, EOFError):
        return False


def _normal_records(items: list[tuple[object, FDURecord]], normal_label: str) -> list[FDURecord]:
    """Use each pure-normal DSANet sequence once, regardless of CSV variants."""
    unique: dict[str, FDURecord] = {}
    for row, record in items:
        if str(row.label) != normal_label:
            continue
        prior = unique.get(record.video_key)
        if prior is not None and prior.fdu_path != record.fdu_path:
            raise ValueError(f"normal video {record.video_key!r} maps to multiple DSANet FDU paths")
        unique[record.video_key] = record
    if not unique:
        raise ValueError(f"no pure-normal training FDU records with label={normal_label!r}")
    return [unique[key] for key in sorted(unique)]


def _stats_metadata(
    contract_path: Path,
    source_list: Path,
    fdu_manifest: Path,
    normal_records: list[FDURecord],
    normal_label: str,
) -> dict:
    return {
        "format_version": 1,
        "normalization": "per-neuron z-score from XD train pure-normal FDU sequences",
        "transfer_contract_sha256": sha256_file(contract_path),
        "source_list_sha256": sha256_file(source_list),
        "dsanet_fdu_manifest_sha256": sha256_file(fdu_manifest),
        "normal_label": normal_label,
        "normal_video_count": len(normal_records),
        "normal_source_fingerprint": source_fingerprint(normal_records),
    }


def _load_normal_stats(stats_path: Path, metadata_path: Path, contract_path: Path, width: int) -> tuple[np.ndarray, np.ndarray]:
    if not stats_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"normal statistics are absent; first run build_features.py with --split train: {stats_path.parent}"
        )
    metadata = load_json(metadata_path)
    if str(metadata.get("transfer_contract_sha256", "")) != sha256_file(contract_path):
        raise ValueError(f"{metadata_path}: normal statistics were built for another transfer contract")
    archive = np.load(stats_path, allow_pickle=False)
    try:
        mean = np.asarray(archive["mean"], dtype=np.float32)
        std = np.asarray(archive["std"], dtype=np.float32)
    finally:
        archive.close()
    if mean.shape != (width,) or std.shape != (width,) or not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError(f"{stats_path}: invalid normal-statistic shapes or values")
    if np.any(std < 1e-6):
        raise ValueError(f"{stats_path}: standard deviation floor was not applied")
    return mean, std


def _build_normal_stats(
    stats_root: Path,
    normal_records: list[FDURecord],
    width: int,
    expected_metadata: dict,
    no_resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    stats_path, metadata_path = stats_root / "normal_stats.npz", stats_root / "normal_stats.json"
    persisted_metadata = dict(expected_metadata)
    persisted_metadata.pop("_contract_path", None)
    if not no_resume and stats_path.is_file() and metadata_path.is_file() and load_json(metadata_path) == persisted_metadata:
        return _load_normal_stats(stats_path, metadata_path, Path(expected_metadata["_contract_path"]), width)

    shard_root = stats_root / "normal_stat_shards"
    if no_resume:
        remove_tree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    for record in tqdm(normal_records, desc="cache pure-normal FDU statistics", unit="video"):
        shard = _shard_path(shard_root, record)
        if _valid_stat_shard(shard, record, width):
            continue
        feature = load_fdu_feature(record.fdu_path, width).astype(np.float64, copy=False)
        size, mtime_ns = feature_file_signature(record.fdu_path)
        atomic_save_npz(
            shard,
            count=np.asarray(len(feature), dtype=np.int64),
            sum=feature.sum(axis=0, dtype=np.float64),
            sum_sq=np.square(feature).sum(axis=0, dtype=np.float64),
            source_size=np.asarray(size, dtype=np.int64),
            source_mtime_ns=np.asarray(mtime_ns, dtype=np.int64),
            fdu_width=np.asarray(width, dtype=np.int64),
        )

    total_count = 0
    total_sum = np.zeros(width, dtype=np.float64)
    total_sum_sq = np.zeros(width, dtype=np.float64)
    for record in normal_records:
        shard = _shard_path(shard_root, record)
        if not _valid_stat_shard(shard, record, width):
            raise RuntimeError(f"{shard}: invalid normal-statistics shard after build")
        archive = np.load(shard, allow_pickle=False)
        try:
            total_count += int(np.asarray(archive["count"]).item())
            total_sum += np.asarray(archive["sum"], dtype=np.float64)
            total_sum_sq += np.asarray(archive["sum_sq"], dtype=np.float64)
        finally:
            archive.close()
    if total_count < 2:
        raise RuntimeError("at least two pure-normal FDU time positions are required for z-score statistics")
    mean = total_sum / total_count
    variance = (total_sum_sq - np.square(total_sum) / total_count) / (total_count - 1)
    std = np.sqrt(np.maximum(variance, 1e-12))
    atomic_save_npz(
        stats_path,
        mean=mean.astype(np.float32),
        std=np.maximum(std, 1e-6).astype(np.float32),
        count=np.asarray(total_count, dtype=np.int64),
    )
    save_json(metadata_path, persisted_metadata)
    return _load_normal_stats(stats_path, metadata_path, Path(expected_metadata["_contract_path"]), width)


def _stage_spec(
    split: str,
    contract_path: Path,
    source_list: Path,
    fdu_manifest: Path,
    records: list[FDURecord],
    stats_path: Path,
    input_width: int,
    fdu_validation: dict,
) -> dict:
    return {
        "format_version": 1,
        "dataset": "xd",
        "split": split,
        "transfer_contract_sha256": sha256_file(contract_path),
        "source_list_sha256": sha256_file(source_list),
        "dsanet_fdu_manifest_sha256": sha256_file(fdu_manifest),
        "fdu_source_fingerprint": source_fingerprint(records),
        "normal_stats_sha256": sha256_file(stats_path),
        "dsanet_export_validation": fdu_validation,
        "temporal_alignment": "strict",
        "input_width": input_width,
    }


def _prepare_stage_spec(stage: Path, expected: dict) -> None:
    path = stage / "build_spec.json"
    if path.is_file():
        current = load_json(path)
        if current != expected:
            raise RuntimeError(f"{path}: inputs changed; use --clean or a new --output-root")
    else:
        save_json(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resumable strict-time-aligned [DSANet FDU | VadCLIP 512D] XD features."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--source-list", required=True, help="Reusable local XD 512D VadCLIP path,label CSV.")
    parser.add_argument("--source-path-base", default=".", help="Base for relative paths in --source-list.")
    parser.add_argument("--clip-prefix-from", default="", help="Optional stale source-feature prefix.")
    parser.add_argument("--clip-prefix-to", default="", help="Replacement for --clip-prefix-from.")
    parser.add_argument("--dsanet-fdu-manifest", required=True, help="DSANet fdu_features/<split>/aligned_features.csv.")
    parser.add_argument("--fdu-path-base", default=".", help="Base for relative fdu_path entries in the DSANet manifest.")
    parser.add_argument("--fdu-dir", default="", help="Optional DSANet fdu_features/<split>/features override.")
    parser.add_argument("--fdu-prefix-from", default="", help="Optional stale DSANet FDU prefix.")
    parser.add_argument("--fdu-prefix-to", default="", help="Replacement for --fdu-prefix-from.")
    parser.add_argument("--neuron-contract", required=True, help="contract/dsanet_transfer_neurons.json from prepare.py.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--normal-label", default=NORMAL_XD_LABEL, help="Pure-normal XD label used only for train statistics.")
    parser.add_argument("--allow-missing-fdu", action="store_true", help="Skip missing FDU rows only in train, for approved DSANet export omissions.")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only this split's fused-feature stage.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute this split and normal-statistic shards instead of reusing valid artifacts.")
    args = parser.parse_args()
    if args.allow_missing_fdu and args.split != "train":
        parser.error("--allow-missing-fdu is valid only with --split train")

    root = output_path(args.output_root)
    feature_root, list_root = root / "features", root / "lists"
    stage = feature_root / args.split
    contract_path = Path(args.neuron_contract).resolve()
    source_list, fdu_manifest = Path(args.source_list).resolve(), Path(args.dsanet_fdu_manifest).resolve()
    if args.clean:
        remove_tree(stage)
        if args.split == "train":
            for target in (feature_root / "normal_stats.npz", feature_root / "normal_stats.json", feature_root / "normal_stat_shards"):
                if target.is_dir():
                    remove_tree(target)
                elif target.exists():
                    target.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    list_root.mkdir(parents=True, exist_ok=True)

    contract = load_transfer_contract(contract_path)
    fdu_validation = validate_dsanet_export_spec(fdu_manifest, contract)
    width, input_width = int(contract["neuron_width"]), int(contract["input_width"])
    source = read_vadclip_list(source_list)
    fdu_by_variant = read_dsanet_manifest(
        fdu_manifest, args.fdu_path_base, args.fdu_dir, args.fdu_prefix_from, args.fdu_prefix_to,
    )
    _matched, missing_manifest = match_records(source, fdu_by_variant)
    items: list[tuple[object, FDURecord]] = []
    skipped = [(path, label, variant, "missing_from_dsanet_manifest") for path, label, variant in missing_manifest]
    for row in source.itertuples(index=False):
        record = fdu_by_variant.get(str(row.variant))
        if record is None:
            continue
        if not record.fdu_path.is_file():
            skipped.append((str(row.path), str(row.label), str(row.variant), "missing_fdu_feature"))
            continue
        items.append((row, record))
    if skipped and not args.allow_missing_fdu:
        preview = "; ".join(f"{variant} ({reason})" for _path, _label, variant, reason in skipped[:3])
        raise FileNotFoundError(f"{len(skipped)} VadCLIP rows lack reusable DSANet FDUs: {preview}")
    if not items:
        raise RuntimeError("no matched VadCLIP/DSANet FDU rows remain")

    stats_path, stats_metadata_path = feature_root / "normal_stats.npz", feature_root / "normal_stats.json"
    if args.split == "train":
        normal_records = _normal_records(items, args.normal_label)
        metadata = _stats_metadata(contract_path, source_list, fdu_manifest, normal_records, args.normal_label)
        metadata["_contract_path"] = str(contract_path)
        mean, std = _build_normal_stats(feature_root, normal_records, width, metadata, args.no_resume)
    else:
        mean, std = _load_normal_stats(stats_path, stats_metadata_path, contract_path, width)

    spec = _stage_spec(
        args.split, contract_path, source_list, fdu_manifest,
        [record for _row, record in items], stats_path, input_width, fdu_validation,
    )
    _prepare_stage_spec(stage, spec)
    output_csv = list_root / f"xd_dsanet_transfer_{args.split}.csv"
    rows, alignment_rows = [], []
    for row, record in tqdm(items, desc=f"build {args.split} transferred features", unit="feature"):
        target = stage / f"{row.variant}.npy"
        clip_path = resolve_recorded_path(row.path, args.source_path_base, args.clip_prefix_from, args.clip_prefix_to)
        try:
            clip = load_clip_feature(clip_path)
            if clip.shape[1] != CLIP_FEATURE_DIM:
                raise ValueError(f"{clip_path}: expected {CLIP_FEATURE_DIM}D VadCLIP input, got {clip.shape}")
            expected_shape = (len(clip), input_width)
            if not args.no_resume and _valid_feature(target, expected_shape):
                action = "reused"
            else:
                fdu = load_fdu_feature(record.fdu_path, width)
                if len(fdu) != len(clip):
                    raise ValueError(
                        f"{row.variant}: strict temporal alignment failed, DSANet FDU T={len(fdu)} but VadCLIP T={len(clip)}"
                    )
                normalized = (fdu - mean) / std
                fused = np.concatenate([normalized.astype(np.float32), clip], axis=1).astype(np.float32, copy=False)
                if fused.shape != expected_shape or not np.isfinite(fused).all():
                    raise RuntimeError(f"{row.variant}: invalid fused feature shape or values")
                atomic_save_npy(target, fused)
                action = "wrote"
            rows.append((relative_metadata_path(target, output_csv.parent), str(row.label)))
            alignment_rows.append((
                str(row.path), str(row.variant), str(row.video_key), len(clip), width, input_width, action,
            ))
        except Exception:
            # --allow-missing-fdu applies only to known absent DSANet export
            # rows, which were filtered before this loop.  A mismatch or a
            # corrupt artifact here must fail loudly instead of silently
            # changing the VadCLIP train/test distribution.
            raise

    if args.split == "test" and skipped:
        raise RuntimeError("test split cannot skip rows because VadCLIP ground-truth alignment would be invalid")
    write_csv(output_csv, ["path", "label"], rows)
    write_csv(
        stage / "alignment.csv",
        ["source_path", "variant", "video_key", "clip_length", "neuron_width", "input_width", "action"],
        alignment_rows,
    )
    write_csv(stage / "skipped_rows.csv", ["source_path", "label", "variant", "reason"], skipped)
    save_json(stage / "summary.json", {
        "dataset": "xd", "split": args.split,
        "source_list": relative_metadata_path(source_list, stage),
        "dsanet_fdu_manifest": relative_metadata_path(fdu_manifest, stage),
        "neuron_contract": relative_metadata_path(contract_path, stage),
        "normal_stats": relative_metadata_path(stats_path, stage),
        "dsanet_export_validation": fdu_validation,
        "strict_temporal_alignment": True,
        "rows_written": len(rows), "rows_skipped": len(skipped),
        "neuron_width": width, "input_width": input_width,
        "list_csv": relative_metadata_path(output_csv, stage),
    })
    print(f"wrote {output_csv}: {len(rows)} rows of [T,{input_width}] features; skipped={len(skipped)}", flush=True)


if __name__ == "__main__":
    main()

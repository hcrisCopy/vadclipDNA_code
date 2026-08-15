"""Materialize the selected DNA probe snippets from reusable CLIP hidden files.

The stage writes deterministic video shards first. A cancelled run reuses every
valid shard, then merges only after the full cache is available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    atomic_save_npz,
    default_output_root,
    load_hidden,
    resolve_recorded_path,
    save_json,
    stage_dir,
)


REQUIRED_COLUMNS = {
    "sample_id", "split", "target", "video_key", "source_label", "source_role", "hidden_path", "time_index",
}


def read_shard(path: Path) -> dict[str, np.ndarray] | None:
    try:
        archive = np.load(path, allow_pickle=False)
        try:
            needed = {"hidden", "target", "split", "sample_id", "video_key"}
            if not needed.issubset(archive.files):
                return None
            result = {name: archive[name] for name in needed}
        finally:
            archive.close()
    except Exception:
        return None
    count = len(result["target"])
    if count == 0 or any(len(result[name]) != count for name in ("hidden", "split", "sample_id", "video_key")):
        return None
    if result["hidden"].ndim != 3 or not np.isfinite(result["hidden"]).all():
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create resumable [N,12,768] DNA probe cache from hidden artifacts.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], default="xd")
    parser.add_argument("--samples-csv", default="", help="Defaults to samples/samples.csv under --output-root.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--videos-per-shard", type=int, default=25)
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only cache under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute each cache shard, retaining the stage directory.")
    args = parser.parse_args()
    if args.videos_per_shard <= 0:
        parser.error("--videos-per-shard must be positive")

    output = stage_dir(args.output_root, "cache", clean=args.clean)
    samples_csv = Path(args.samples_csv).resolve() if args.samples_csv else (output.parent / "samples" / "samples.csv").resolve()
    samples = pd.read_csv(samples_csv)
    missing = REQUIRED_COLUMNS - set(samples.columns)
    if missing:
        raise ValueError(f"{samples_csv}: missing probe sample columns {sorted(missing)}")
    negatives = samples[samples["target"].astype(int) == 0]
    expected_normal = "A" if args.dataset == "xd" else "Normal"
    if set(negatives["source_label"].astype(str)) != {expected_normal} or set(negatives["source_role"].astype(str)) != {"pure_normal_video"}:
        raise ValueError(f"{samples_csv}: all target=0 rows must come from {expected_normal}-labelled pure normal videos")
    samples = samples.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(samples["sample_id"].to_numpy(), np.arange(len(samples))):
        raise ValueError(f"{samples_csv}: sample_id must be contiguous from zero")
    shards_dir = (output / "shards").resolve()
    shards_dir.mkdir(parents=True, exist_ok=True)
    video_keys = sorted(samples["video_key"].astype(str).unique())
    plans = [video_keys[index:index + args.videos_per_shard] for index in range(0, len(video_keys), args.videos_per_shard)]

    shard_paths: list[Path] = []
    for shard_index, keys in enumerate(tqdm(plans, desc="build probe-cache shards", unit="shard")):
        shard_path = shards_dir / f"shard_{shard_index:05d}.npz"
        shard_paths.append(shard_path)
        subset = samples[samples["video_key"].astype(str).isin(keys)]
        expected_ids = subset["sample_id"].to_numpy(dtype=np.int64)
        existing = None if args.no_resume else read_shard(shard_path)
        if existing is not None and np.array_equal(existing["sample_id"], expected_ids):
            continue

        values: list[np.ndarray] = []
        targets: list[int] = []
        splits: list[str] = []
        sample_ids: list[int] = []
        keys_out: list[str] = []
        for row in subset.itertuples(index=False):
            hidden_path = resolve_recorded_path(str(row.hidden_path), samples_csv.parent)
            hidden, _metadata = load_hidden(hidden_path)
            time_index = int(row.time_index)
            if not 0 <= time_index < len(hidden):
                raise IndexError(f"sample {row.sample_id}: time_index={time_index} outside {hidden_path} length={len(hidden)}")
            values.append(hidden[time_index].astype(np.float16))
            targets.append(int(row.target))
            splits.append(str(row.split))
            sample_ids.append(int(row.sample_id))
            keys_out.append(str(row.video_key))
        atomic_save_npz(
            shard_path,
            hidden=np.stack(values, axis=0),
            target=np.asarray(targets, dtype=np.int8),
            split=np.asarray(splits, dtype="U10"),
            sample_id=np.asarray(sample_ids, dtype=np.int64),
            video_key=np.asarray(keys_out, dtype="U512"),
        )

    parts = [read_shard(path) for path in tqdm(shard_paths, desc="merge probe-cache shards", unit="shard")]
    if any(part is None for part in parts):
        invalid = [str(path) for path, part in zip(shard_paths, parts) if part is None]
        raise RuntimeError(f"cannot merge invalid cache shards: {invalid[:3]}")
    merged = {name: np.concatenate([part[name] for part in parts if part is not None], axis=0) for name in parts[0] if parts[0] is not None}
    order = np.argsort(merged["sample_id"], kind="mergesort")
    merged = {name: value[order] for name, value in merged.items()}
    if not np.array_equal(merged["sample_id"], np.arange(len(samples))):
        raise RuntimeError("merged cache sample IDs do not exactly match samples.csv")
    if set(merged["split"].tolist()) != {"train", "validation"} or set(merged["target"].tolist()) != {0, 1}:
        raise RuntimeError("merged cache must retain both split values and both binary targets")
    layers = np.arange(1, merged["hidden"].shape[1] + 1, dtype=np.int16)
    atomic_save_npz(output / "probe_cache.npz", layers=layers, **merged)
    save_json(output / "summary.json", {
        "dataset": args.dataset, "samples_csv": str(samples_csv),
        "samples": int(len(samples)), "videos": len(video_keys), "shards": len(shard_paths),
        "hidden_shape": list(merged["hidden"].shape), "layers": layers.tolist(),
        "resume_contract": "sample_id exact match plus finite [N,L,D] shard tensors",
    })
    print(f"wrote {output / 'probe_cache.npz'} with hidden shape {tuple(merged['hidden'].shape)}", flush=True)


if __name__ == "__main__":
    main()

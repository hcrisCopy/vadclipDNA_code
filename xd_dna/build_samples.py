"""Build DNA probe samples: abnormal high-score snippets versus pure-normal video snippets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    base_key,
    default_output_root,
    deterministic_split,
    grouped_rows,
    is_pure_normal_xd,
    load_hidden,
    manifest_hidden_paths,
    read_path_label_csv,
    relpath,
    resample_scores,
    resolve_recorded_path,
    save_json,
    stage_dir,
    write_csv,
)


def read_pseudo_scores(path: str | Path) -> dict[str, tuple[str, Path]]:
    source = Path(path).resolve()
    frame = pd.read_csv(source)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: pseudo-score CSV is missing {sorted(missing)}")
    result: dict[str, tuple[str, Path]] = {}
    for row in frame.itertuples(index=False):
        key = str(row.key)
        score_path = resolve_recorded_path(str(row.score_path), source.parent)
        result[key] = (str(row.label), score_path)
    return result


def normal_rows(
    keys: list[str],
    hidden_by_key: dict[str, str],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    """Balance a split with snippets exclusively sampled from A-labelled videos.

    Contributions are balanced by video before a final deterministic sample, so
    a long normal video cannot dominate the probe's negative class.
    """
    if count <= 0:
        return []
    if not keys:
        raise RuntimeError("cannot construct pure-normal negatives without normal videos")
    quota = int(np.ceil(count / len(keys)))
    candidates: list[tuple[str, int]] = []
    lengths: dict[str, int] = {}
    for key in tqdm(sorted(keys), desc="sample pure-normal negatives", unit="video", leave=False):
        hidden, _metadata = load_hidden(hidden_by_key[key])
        lengths[key] = len(hidden)
        take = min(quota, len(hidden))
        indices = rng.choice(len(hidden), size=take, replace=False)
        candidates.extend((key, int(index)) for index in indices)
    if len(candidates) >= count:
        chosen = rng.choice(len(candidates), size=count, replace=False)
        return [candidates[int(index)] for index in chosen]

    # This fallback is unusual for XD but remains pure-normal and records no
    # abnormal-video snippet as a negative when the normal pool is short.
    extras = [(key, int(rng.integers(lengths[key]))) for key in sorted(keys) for _ in range(quota)]
    combined = candidates + extras
    chosen = rng.choice(len(combined), size=count, replace=True)
    return [combined[int(index)] for index in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/validation probe samples with pure-normal XD negatives."
    )
    parser.add_argument("--source-train-csv", required=True, help="Reusable XD 512D train CSV with path,label columns.")
    parser.add_argument("--hidden-manifest", required=True, help="Reusable [T,12,768] CLS hidden manifest.")
    parser.add_argument(
        "--hidden-path-base", default=".",
        help="Base directory for relative hidden_path entries in the DSANet manifest; use '.' from vadclipDNA_code.",
    )
    parser.add_argument("--pseudo-csv", required=True, help="group_scores.csv made by xd_dna.score_pseudo.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--hidden-prefix-from", default="", help="Optional stale hidden-path prefix to rewrite.")
    parser.add_argument("--hidden-prefix-to", default="", help="Replacement for --hidden-prefix-from.")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--top-p", type=float, default=0.10, help="High pseudo-score fraction selected from each abnormal video.")
    parser.add_argument("--min-positive-per-video", type=int, default=3)
    parser.add_argument("--max-positive-per-video", type=int, default=32)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only samples under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Rebuild samples.csv even if it is already present.")
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5)")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.min_positive_per_video <= 0 or args.max_positive_per_video < args.min_positive_per_video:
        parser.error("positive sample bounds must satisfy 0 < min <= max")

    output = stage_dir(args.output_root, "samples", clean=args.clean)
    target = output / "samples.csv"
    if target.exists() and not args.no_resume:
        print(f"reuse existing sample manifest: {target}", flush=True)
        return

    groups = grouped_rows(read_path_label_csv(args.source_train_csv))
    hidden_by_key, token_pool = manifest_hidden_paths(
        args.hidden_manifest, args.hidden_prefix_from, args.hidden_prefix_to, args.hidden_path_base,
    )
    pseudo_by_key = read_pseudo_scores(args.pseudo_csv)

    normals, abnormals, skipped = [], [], []
    for key, group in groups.items():
        label = str(group.iloc[0]["label"])
        if key not in hidden_by_key:
            skipped.append([key, label, "missing_hidden"])
        elif is_pure_normal_xd(label):
            normals.append(key)
        elif key not in pseudo_by_key:
            skipped.append([key, label, "missing_pseudo_score"])
        elif pseudo_by_key[key][0] != label:
            skipped.append([key, label, "pseudo_label_mismatch"])
        else:
            abnormals.append(key)
    if len(normals) < 2 or len(abnormals) < 2:
        raise RuntimeError(
            f"need at least two readable normal and abnormal videos; got normal={len(normals)} abnormal={len(abnormals)}"
        )

    rng = np.random.default_rng(args.seed)
    normal_split = deterministic_split(normals, args.validation_fraction, rng)
    abnormal_split = deterministic_split(abnormals, args.validation_fraction, rng)
    sample_rows: list[dict[str, object]] = []

    for key in tqdm(sorted(abnormals), desc="select abnormal pseudo-positive snippets", unit="video"):
        label = str(groups[key].iloc[0]["label"])
        hidden, _metadata = load_hidden(hidden_by_key[key])
        _pseudo_label, score_path = pseudo_by_key[key]
        scores = resample_scores(np.load(score_path, allow_pickle=False), len(hidden))
        wanted = int(np.ceil(len(scores) * args.top_p))
        wanted = min(max(args.min_positive_per_video, wanted), args.max_positive_per_video, len(scores))
        order = np.argsort(scores, kind="mergesort")
        for time_index in order[-wanted:][::-1]:
            sample_rows.append({
                "split": abnormal_split[key], "target": 1, "video_key": key,
                "source_label": label, "source_role": "abnormal_high_pseudo_score",
                "hidden_path": relpath(hidden_by_key[key], output), "time_index": int(time_index),
                "baseline_score": float(scores[int(time_index)]),
            })

    for split in ("train", "validation"):
        positives = sum(row["split"] == split and row["target"] == 1 for row in sample_rows)
        eligible_normals = [key for key in normals if normal_split[key] == split]
        for key, time_index in normal_rows(eligible_normals, hidden_by_key, positives, rng):
            sample_rows.append({
                "split": split, "target": 0, "video_key": key,
                "source_label": "A", "source_role": "pure_normal_video",
                "hidden_path": relpath(hidden_by_key[key], output), "time_index": int(time_index),
                "baseline_score": "",
            })

    frame = pd.DataFrame(sample_rows)
    if frame.empty:
        raise RuntimeError("sample manifest is empty")
    for split in ("train", "validation"):
        counts = frame.loc[frame["split"] == split, "target"].value_counts().to_dict()
        if counts.get(0, 0) != counts.get(1, 0) or counts.get(0, 0) == 0:
            raise RuntimeError(f"{split}: expected balanced positive/pure-normal negative samples, got {counts}")
        negative_labels = set(frame.loc[(frame["split"] == split) & (frame["target"] == 0), "source_label"])
        if negative_labels != {"A"}:
            raise RuntimeError(f"{split}: negative samples are not exclusively pure normal A videos: {negative_labels}")
    frame = frame.sort_values(["split", "target", "video_key", "time_index"], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
    write_csv(target, frame.columns.tolist(), frame.itertuples(index=False, name=None))
    write_csv(output / "skipped_videos.csv", ["video_key", "label", "reason"], skipped)
    save_json(output / "summary.json", {
        "dataset": "xd",
        "token_pool": token_pool,
        "negative_source": "pure_normal_video_only",
        "negative_label": "A",
        "positive_source": "highest_frozen_vadclip_logits1_scores_from_abnormal_videos",
        "validation_fraction": args.validation_fraction,
        "top_p": args.top_p,
        "min_positive_per_video": args.min_positive_per_video,
        "max_positive_per_video": args.max_positive_per_video,
        "samples_by_split_target": {
            f"{split}_{target_value}": int(((frame["split"] == split) & (frame["target"] == target_value)).sum())
            for split in ("train", "validation") for target_value in (0, 1)
        },
        "normal_videos": len(normals), "abnormal_videos": len(abnormals), "skipped_videos": len(skipped),
        "source_train_csv": args.source_train_csv, "hidden_manifest": args.hidden_manifest,
        "pseudo_csv": args.pseudo_csv,
    })
    print(f"wrote {target} with {len(frame)} balanced samples; negatives are pure normal A videos only", flush=True)


if __name__ == "__main__":
    main()

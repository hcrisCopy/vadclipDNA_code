"""Generate reproducible XD pseudo scores with the frozen local VadCLIP model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import (
    atomic_save_npy,
    default_output_root,
    grouped_rows,
    load_clip_feature,
    read_path_label_csv,
    relpath,
    save_json,
    stage_dir,
    write_csv,
    resolve_recorded_path,
)
from .vadclip import build_baseline, load_options, score_sequence


def valid_score(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return values if len(values) and np.isfinite(values).all() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score XD training videos with frozen local VadCLIP logits1.")
    parser.add_argument("--source-train-csv", required=True, help="Reusable 512D XD train CSV with path,label columns.")
    parser.add_argument(
        "--source-path-base", default=".",
        help="Base directory for relative paths inside the original VadCLIP CSV; use '.' when running from vadclipDNA_code.",
    )
    parser.add_argument("--baseline-model", required=True, help="VadCLIP XD 512D checkpoint, not a DSANet checkpoint.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only pseudo_scores under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video scores without deleting the stage.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = stage_dir(args.output_root, "pseudo_scores", clean=args.clean)
    score_dir = (root / "scores").resolve()
    score_dir.mkdir(parents=True, exist_ok=True)
    options = load_options()
    if options.visual_width != 512:
        raise ValueError(f"unexpected local XD VadCLIP input width: {options.visual_width}")
    device = torch.device(args.device)
    model = build_baseline(options, args.baseline_model, device)

    source_csv = Path(args.source_train_csv).resolve()
    source_path_base = Path(args.source_path_base).resolve()
    groups = grouped_rows(read_path_label_csv(source_csv))
    rows: list[list[object]] = []
    for key, group in tqdm(groups.items(), desc="VadCLIP pseudo scores", unit="video"):
        output = score_dir / f"{key}.npy"
        scores = None if args.no_resume else valid_score(output)
        if scores is None:
            # Original VadCLIP's XDDataset calls np.load(row['path']) directly,
            # so its relative paths are relative to the process working
            # directory, not the CSV directory.  Keep that contract exactly.
            variants = [load_clip_feature(resolve_recorded_path(str(row.path), source_path_base)) for row in group.itertuples(index=False)]
            for feature in variants:
                if feature.shape[1] != options.visual_width:
                    raise ValueError(f"{key}: expected 512D final CLIP feature, got {feature.shape}")
            lengths = {len(feature) for feature in variants}
            if len(lengths) != 1:
                raise ValueError(f"{key}: 512D feature variants have inconsistent temporal lengths {sorted(lengths)}")
            variant_scores = [score_sequence(model, feature, options.visual_length, device) for feature in variants]
            scores = np.stack(variant_scores, axis=0).mean(axis=0).astype(np.float32)
            atomic_save_npy(output, scores)
        rows.append([key, str(group.iloc[0]["label"]), relpath(output, root), len(scores), len(group)])

    write_csv(root / "group_scores.csv", ["key", "label", "score_path", "score_len", "num_variants"], rows)
    save_json(root / "run_config.json", {
        "method": "frozen_vadclip_logits1_pseudo_score",
        "source_train_csv": args.source_train_csv,
        "source_path_base": args.source_path_base,
        "baseline_model": args.baseline_model,
        "score_aggregation": "mean over same-video CLIP variants",
        "visual_length": int(options.visual_length),
        "visual_width": int(options.visual_width),
        "videos": len(rows),
    })
    print(f"wrote {root / 'group_scores.csv'} for {len(rows)} XD video groups", flush=True)


if __name__ == "__main__":
    main()

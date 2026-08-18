"""Evaluate a trained CTNC sidecar on an unchanged, frozen VadCLIP baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .assets import load_assets
from .baseline import build_frozen_baseline
from .circuit import ChannelRankVerifier, load_verifier_state
from .common import default_output_root, hidden_manifest_paths, save_json, stage_dir
from .dataset import HiddenBagDataset
from .evaluate import collect_predictions, summarize_predictions, write_prediction_index


def model_state(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"{path}: expected a CTNC state dictionary or checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable official-metric CTNC hidden-channel verification with an unchanged frozen VadCLIP baseline.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model-path", required=True, help="training/model_best.pth")
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only evaluation/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video prediction artifacts.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "evaluation", clean=args.clean)
    device = torch.device(args.device)
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError(f"assets dataset={assets['dataset']!r} does not match --dataset={args.dataset!r}")
    selected_layers = assets["selected_layers"].cpu().numpy()
    selected_dimensions = assets["selected_dimensions"].cpu().numpy()
    hidden = hidden_manifest_paths(
        args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to
    )
    baseline, options = build_frozen_baseline(args.dataset, args.init_baseline_model, device)
    test_set = HiddenBagDataset(
        args.dataset, args.source_test_csv, args.source_path_base, hidden, selected_layers, selected_dimensions,
        options.visual_length, False, args.alignment, False,
    )
    verifier = ChannelRankVerifier(assets).to(device)
    load_verifier_state(verifier, model_state(args.model_path))
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    predictions, labels, rows = collect_predictions(
        verifier, test_set, baseline, options.visual_length, args.dataset, device,
        output / "predictions", not args.no_resume, "CTNC test",
    )
    metrics = summarize_predictions(predictions, labels, gt, gtsegments, gtlabels, args.dataset)
    write_prediction_index(output / "prediction_index.csv", rows)
    save_json(output / "metrics.json", {
        **metrics,
        "dataset": args.dataset,
        "assets": args.assets,
        "model_path": args.model_path,
        "baseline_checkpoint": args.init_baseline_model,
        "verification": "hard-selected hidden-channel witness evidence re-ranks frozen baseline anomaly odds",
    })
    final = metrics["rank_verified"]
    print(
        f"baseline AUC2/AP2={metrics['baseline']['auc2']:.6f}/{metrics['baseline']['ap2']:.6f} | "
        f"channel AUC/AP={metrics['channel_evidence_only']['auc']:.6f}/{metrics['channel_evidence_only']['ap']:.6f} | "
        f"selected-channel AP={metrics['channel_evidence_only']['ap']:.6f} | "
        f"verified AUC2/AP2={final['auc2']:.6f}/{final['ap2']:.6f} dMAP={final['detection_map_average']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()

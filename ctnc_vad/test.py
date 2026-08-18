"""Evaluate a trained CTNC sidecar on an unchanged, frozen VadCLIP baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .assets import load_assets
from .baseline import build_frozen_baseline
from .circuit import NormalityCircuit
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
    parser = argparse.ArgumentParser(description="Resumable official-metric CTNC test with an unchanged frozen VadCLIP baseline.")
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
    parser.add_argument("--rank-anchor-fraction", type=float, default=0.125)
    parser.add_argument("--rank-margin", type=float, default=0.10)
    parser.add_argument("--rank-strength", type=float, default=0.25)
    parser.add_argument("--rank-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only evaluation/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video prediction artifacts.")
    args = parser.parse_args()
    if not 0 < args.rank_anchor_fraction <= 0.5 or min(args.rank_margin, args.rank_strength) < 0 or args.rank_steps < 0:
        parser.error("invalid rank-rectification arguments")
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
    circuit = NormalityCircuit(assets).to(device)
    circuit.load_state_dict(model_state(args.model_path), strict=True)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    predictions, labels, rows = collect_predictions(
        circuit, test_set, baseline, options.visual_length, args.dataset, device,
        args.rank_anchor_fraction, args.rank_margin, args.rank_strength, args.rank_steps,
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
        "rank_rectification": {
            "anchor_fraction": args.rank_anchor_fraction, "margin": args.rank_margin,
            "strength": args.rank_strength, "steps": args.rank_steps,
        },
    })
    final = metrics["rank_rectified"]
    print(
        f"baseline AUC2/AP2={metrics['baseline']['auc2']:.6f}/{metrics['baseline']['ap2']:.6f} | "
        f"circuit AUC/AP={metrics['circuit_only']['auc']:.6f}/{metrics['circuit_only']['ap']:.6f} | "
        f"rectified AUC2/AP2={final['auc2']:.6f}/{final['ap2']:.6f} dMAP={final['detection_map_average']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Official-metric test for a trained CTSC sidecar and frozen VadCLIP."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .assets import load_assets
from .baseline import build_frozen_baseline
from .circuit import SparseClassCircuit
from .common import default_output_root, hidden_manifest_paths, save_json, stage_dir
from .data import ChannelBagDataset
from .evaluate import collect_predictions, summarize, write_prediction_index
from .metrics import print_vadclip_metrics


def model_state(path: str | Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and "model_state_dict" in value:
        return value["model_state_dict"]
    if isinstance(value, dict):
        return value
    raise ValueError(f"{path}: expected CTSC state dictionary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTSC certified raw-channel promotion with exact VadCLIP metrics.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only evaluation/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "evaluation", clean=args.clean)
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError("asset dataset does not match --dataset")
    paths = hidden_manifest_paths(args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to)
    device = torch.device(args.device)
    baseline, options = build_frozen_baseline(args.dataset, args.init_baseline_model, device)
    dataset = ChannelBagDataset(args.dataset, args.source_test_csv, args.source_path_base, paths, assets["selected_layers"].numpy(), assets["selected_dimensions"].numpy(), options.visual_length, False, args.alignment)
    model = SparseClassCircuit(assets).to(device)
    model.load_state_dict(model_state(args.model_path), strict=True)
    gt, gtsegments, gtlabels = np.load(args.gt_path), np.load(args.gt_segment_path, allow_pickle=True), np.load(args.gt_label_path, allow_pickle=True)
    predictions, _labels, rows = collect_predictions(model, dataset, baseline, options.visual_length, args.dataset, device, output / "predictions", not args.no_resume, "CTSC test")
    metrics = summarize(predictions, gt, gtsegments, gtlabels, args.dataset)
    write_prediction_index(output / "prediction_index.csv", rows)
    save_json(output / "metrics.json", {**metrics, "dataset": args.dataset, "assets": args.assets, "model_path": args.model_path, "baseline_checkpoint": args.init_baseline_model, "method": "class-specific temporal raw-channel circuit plus external certified one-way promotion"})
    print_vadclip_metrics("[Frozen VadCLIP]", metrics["baseline"])
    print_vadclip_metrics("[CTSC circuit only]", metrics["channel_circuit_only"])
    print_vadclip_metrics("[CTSC certified promotion]", metrics["classwise_certified_promotion"])


if __name__ == "__main__":
    main()

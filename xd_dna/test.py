"""Resumable final XD test for the DNA-on-VadCLIP residual model."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import (
    XD_LABELS,
    atomic_save_npz,
    default_output_root,
    load_json,
    save_json,
    stage_dir,
)
from .dataset import XDDNAFeatureDataset
from .metrics import infer_item, metrics_from_predictions
from .model import DNAResidualVadCLIP
from .vadclip import load_options


def cached_prediction(path: Path, expected_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        try:
            prob1, prob2, logits2 = archive["prob1"], archive["prob2"], archive["logits2"]
        finally:
            archive.close()
    except Exception:
        return None
    if prob1.shape != (expected_length,) or prob2.shape != (expected_length,) or logits2.shape != (expected_length, 7):
        return None
    if not (np.isfinite(prob1).all() and np.isfinite(prob2).all() and np.isfinite(logits2).all()):
        return None
    return prob1.astype(np.float32), prob2.astype(np.float32), logits2.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected best XD DNA-VadCLIP model with VadCLIP metrics.")
    parser.add_argument("--test-list", required=True, help="xd_concat_test.csv made by xd_dna.build_features.")
    parser.add_argument("--neuron-json", default="")
    parser.add_argument("--model-path", default="", help="Defaults to training/model_best.pth under --output-root.")
    parser.add_argument("--init-baseline-model", required=True, help="Official 512D VadCLIP XD checkpoint used to construct the wrapper.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--gt-path", default="VadCLIP/list/gt.npy")
    parser.add_argument("--gt-segment-path", default="VadCLIP/list/gt_segment.npy")
    parser.add_argument("--gt-label-path", default="VadCLIP/list/gt_label.npy")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and recompute only evaluation predictions/results.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore valid per-video predictions and recompute them.")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "evaluation", clean=args.clean)
    predictions = (output / "predictions").resolve()
    if args.no_resume and predictions.exists():
        shutil.rmtree(predictions)
    predictions.mkdir(parents=True, exist_ok=True)
    root = output.parent
    neuron_json = Path(args.neuron_json).resolve() if args.neuron_json else (root / "localization" / "selected_neurons.json").resolve()
    contract = load_json(neuron_json)
    neuron_width, input_width = int(contract["neuron_width"]), int(contract["input_width"])
    model_path = Path(args.model_path).resolve() if args.model_path else (root / "training" / "model_best.pth").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model is absent: {model_path}")
    device = torch.device(args.device)
    options = load_options()
    dataset = XDDNAFeatureDataset(args.test_list, options.visual_length, input_width, test_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = DNAResidualVadCLIP(options, args.init_baseline_model, device, neuron_width, args.residual_hidden_dim, args.residual_depth).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    probabilities1: list[np.ndarray] = []
    probabilities2: list[np.ndarray] = []
    logits2_probability: list[np.ndarray] = []
    for item in tqdm(loader, desc="evaluate XD test videos", unit="video"):
        listed_path, length = str(item[3][0]), int(item[2])
        target = predictions / f"{Path(listed_path).stem}.npz"
        result = None if args.no_resume else cached_prediction(target, length)
        if result is None:
            result = infer_item(model, item, options.visual_length, list(XD_LABELS.values()), device)
            atomic_save_npz(target, prob1=result[0], prob2=result[1], logits2=result[2])
        probabilities1.append(result[0])
        probabilities2.append(result[1])
        logits2_probability.append(result[2])
    metrics = metrics_from_predictions(probabilities1, probabilities2, logits2_probability, gt, gtsegments, gtlabels)
    save_json(output / "metrics.json", {
        **metrics.to_dict(), "selection_metric": "AP2", "model_path": str(model_path),
        "neuron_json": str(neuron_json), "test_list": args.test_list,
        "prediction_resume_contract": "per-video finite prob1/prob2/logits2 arrays with matching temporal length",
    })
    print(f"AUC1={metrics.auc1:.6f} AP1={metrics.ap1:.6f} | AUC2={metrics.auc2:.6f} AP2={metrics.ap2:.6f}", flush=True)
    for iou, value in metrics.detection_map_by_iou.items():
        print(f"mAP@{iou}={value:.2f}%", flush=True)
    print(f"average detection mAP={metrics.detection_map_average:.2f}% | wrote {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

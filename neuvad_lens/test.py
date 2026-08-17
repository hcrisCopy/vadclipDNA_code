"""Resumable XD NeuVAD-Lens evaluation with the unchanged VadCLIP metrics."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xd_dna.dataset import XDDNAFeatureDataset
from xd_dna.metrics import infer_item, metrics_from_predictions
from xd_dna.vadclip import load_options

from .common import (
    atomic_save_npz,
    default_output_root,
    labels_for_dataset,
    load_json,
    save_json,
    stage_dir,
)
from .model import NeuVADLensVadCLIP


def cached_prediction(path: Path, expected_length: int, class_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Accept only complete finite per-video predictions from an interrupted run."""
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
    if prob1.shape != (expected_length,) or prob2.shape != (expected_length,) or logits2.shape != (expected_length, class_count):
        return None
    if not (np.isfinite(prob1).all() and np.isfinite(prob2).all() and np.isfinite(logits2).all()):
        return None
    return prob1.astype(np.float32), prob2.astype(np.float32), logits2.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuVAD-Lens using the official XD VadCLIP metric protocol.")
    parser.add_argument("--test-list", required=True, help="xd_neuvad_lens_test.csv made by neuvad_lens.build_features.")
    parser.add_argument("--neuron-json", required=True, help="Reusable xd_dna selected_neurons.json.")
    parser.add_argument("--lens-assets", required=True)
    parser.add_argument("--model-path", default="", help="Defaults to training/model_best.pth under --output-root.")
    parser.add_argument("--init-baseline-model", required=True, help="Official 512D VadCLIP XD checkpoint.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--gt-path", default="VadCLIP/list/gt.npy")
    parser.add_argument("--gt-segment-path", default="VadCLIP/list/gt_segment.npy")
    parser.add_argument("--gt-label-path", default="VadCLIP/list/gt_label.npy")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dna-hidden-dim", type=int, default=1024)
    parser.add_argument("--dna-depth", type=int, default=3)
    parser.add_argument("--text-hidden-dim", type=int, default=512)
    parser.add_argument("--text-depth", type=int, default=2)
    parser.add_argument("--text-temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and recompute only Lens evaluation artifacts.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore valid per-video predictions and recompute them.")
    args = parser.parse_args()
    if args.num_workers < 0 or args.text_temperature <= 0:
        parser.error("workers must be non-negative and text temperature must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "evaluation", clean=args.clean)
    predictions = (output / "predictions").resolve()
    if args.no_resume and predictions.exists():
        shutil.rmtree(predictions)
    predictions.mkdir(parents=True, exist_ok=True)
    root = output.parent
    model_path = Path(args.model_path).resolve() if args.model_path else (root / "training" / "model_best.pth").resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model is absent: {model_path}")
    neuron_width = int(load_json(Path(args.neuron_json).resolve())["neuron_width"])
    input_width = neuron_width + 512 + 768
    device = torch.device(args.device)
    options = load_options("xd")
    dataset = XDDNAFeatureDataset(args.test_list, options.visual_length, input_width, test_mode=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = NeuVADLensVadCLIP(
        options, args.init_baseline_model, args.lens_assets, device, neuron_width,
        args.dna_hidden_dim, args.dna_depth, args.text_hidden_dim, args.text_depth, args.text_temperature,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    prompt_text = list(labels_for_dataset("xd").values())
    probabilities1: list[np.ndarray] = []
    probabilities2: list[np.ndarray] = []
    logits2_probability: list[np.ndarray] = []
    for item in tqdm(loader, desc="evaluate XD Lens test videos", unit="video"):
        listed_path, length = str(item[3][0]), int(item[2])
        target = predictions / f"{Path(listed_path).stem}.npz"
        result = None if args.no_resume else cached_prediction(target, length, options.classes_num)
        if result is None:
            result = infer_item(model, item, options.visual_length, prompt_text, device)
            atomic_save_npz(target, prob1=result[0], prob2=result[1], logits2=result[2])
        probabilities1.append(result[0])
        probabilities2.append(result[1])
        logits2_probability.append(result[2])
    metrics = metrics_from_predictions(probabilities1, probabilities2, logits2_probability, gt, gtsegments, gtlabels)
    save_json(output / "metrics.json", {
        **metrics.to_dict(),
        "selection_metric": "AP2",
        "model_path": str(model_path),
        "lens_assets": args.lens_assets,
        "test_list": args.test_list,
        "prediction_resume_contract": "per-video finite prob1/prob2/logits2 arrays with matching temporal length",
    })
    print(f"AUC1={metrics.auc1:.6f} AP1={metrics.ap1:.6f} | AUC2={metrics.auc2:.6f} AP2={metrics.ap2:.6f}", flush=True)
    for iou, value in metrics.detection_map_by_iou.items():
        print(f"mAP@{iou}={value:.2f}%", flush=True)
    print(f"average detection mAP={metrics.detection_map_average:.2f}% | wrote {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Resumable UCF-Crime final evaluation using the official VadCLIP metrics."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import UCF_TEST_LABELS, atomic_save_npz, load_json, save_json, stage_dir
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
    if prob1.shape != (expected_length,) or prob2.shape != (expected_length,) or logits2.shape != (expected_length, 14):
        return None
    return (prob1.astype(np.float32), prob2.astype(np.float32), logits2.astype(np.float32)) if np.isfinite(logits2).all() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best UCF DNA-VadCLIP model with VadCLIP's official protocol.")
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--output-root", default="../vadclipDNA_data/ucf_normal_negative_top64")
    parser.add_argument("--gt-path", default="VadCLIP/list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="VadCLIP/list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="VadCLIP/list/gt_label_ucf.npy")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "evaluation", clean=args.clean)
    predictions = output / "predictions"
    if args.no_resume and predictions.exists():
        shutil.rmtree(predictions)
    predictions.mkdir(parents=True, exist_ok=True)
    root = output.parent
    neuron_json = Path(args.neuron_json).resolve() if args.neuron_json else (root / "localization" / "selected_neurons.json").resolve()
    contract = load_json(neuron_json)
    neuron_width, input_width = int(contract["neuron_width"]), int(contract["input_width"])
    model_path = Path(args.model_path).resolve() if args.model_path else (root / "training" / "model_best.pth").resolve()
    device = torch.device(args.device)
    options = load_options("ucf")
    dataset = XDDNAFeatureDataset(args.test_list, options.visual_length, input_width, test_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = DNAResidualVadCLIP(options, args.init_baseline_model, device, neuron_width, args.residual_hidden_dim, args.residual_depth).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    gt, gtsegments, gtlabels = np.load(args.gt_path), np.load(args.gt_segment_path, allow_pickle=True), np.load(args.gt_label_path, allow_pickle=True)
    p1: list[np.ndarray] = []
    p2: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[str] = []
    for item in tqdm(loader, desc="evaluate UCF test videos", unit="video"):
        listed_path, length = str(item[3][0]), int(item[2])
        target = predictions / f"{Path(listed_path).stem}.npz"
        result = None if args.no_resume else cached_prediction(target, length)
        if result is None:
            result = infer_item(model, item, options.visual_length, list(UCF_TEST_LABELS.values()), device)
            atomic_save_npz(target, prob1=result[0], prob2=result[1], logits2=result[2])
        p1.append(result[0]); p2.append(result[1]); logits.append(result[2]); labels.append(str(item[1][0]))
    metrics = metrics_from_predictions(p1, p2, logits, gt, gtsegments, gtlabels, dataset="ucf", video_labels=labels)
    save_json(output / "metrics.json", {**metrics.to_dict(), "selection_metric": "AUC1", "model_path": str(model_path), "neuron_json": str(neuron_json), "test_list": args.test_list})
    print(
        f"AUC1={metrics.auc1:.6f} AP1={metrics.ap1:.6f} | AUC2={metrics.auc2:.6f} AP2={metrics.ap2:.6f} | "
        f"Ano-AUC1={metrics.ano_auc1:.6f} Ano-AUC2={metrics.ano_auc2:.6f}",
        flush=True,
    )
    for iou, value in metrics.detection_map_by_iou.items():
        print(f"mAP@{iou}={value:.2f}%", flush=True)
    print(f"average detection mAP={metrics.detection_map_average:.2f}% | wrote {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

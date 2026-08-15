"""DNA triadic neuron localization over all reusable CLIP ViT-B/16 layers."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from tqdm import tqdm

from .common import (
    atomic_save_npy,
    atomic_save_npz,
    default_output_root,
    relpath,
    save_json,
    set_seed,
    stage_dir,
    write_csv,
)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator <= 1e-12 else float(1.0 - np.dot(first, second) / denominator)


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Use DSANet-DNA's probe/activation/gradient/weight triadic ingredients."""
    set_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = torch.from_numpy(train_x.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32, copy=False)).to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).squeeze(-1), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        valid_tensor = torch.from_numpy(validation_x.astype(np.float32, copy=False)).to(device)
        probabilities = torch.sigmoid(model(valid_tensor).squeeze(-1)).cpu().numpy()
        prediction = (probabilities >= 0.5).astype(np.int64)
        train_probabilities = torch.sigmoid(model(x).squeeze(-1))
        residual = (train_probabilities - y).abs().unsqueeze(1)
        weight_tensor = model.weight.detach().reshape(-1)
        mean_gradient = (residual * weight_tensor.abs().unsqueeze(0)).mean(dim=0).cpu().numpy()
    return {
        "weight": model.weight.detach().cpu().numpy().reshape(-1),
        "mean_gradient": mean_gradient,
        "validation_accuracy": float(accuracy_score(validation_y, prediction)),
        "validation_auc": float(roc_auc_score(validation_y, probabilities)),
        "validation_ap": float(average_precision_score(validation_y, probabilities)),
        "final_train_loss": float(loss.detach().cpu().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize DNA neurons with pure-normal negatives and top-k per CLIP layer selection."
    )
    parser.add_argument("--cache", default="", help="Defaults to cache/probe_cache.npz under --output-root.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk-per-layer", type=int, default=64, help="Select this many highest triadic-score neurons independently in every layer.")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only localization under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute localization even if selected_neurons.json already exists.")
    args = parser.parse_args()
    if args.probe_epochs <= 0 or args.probe_lr <= 0 or args.probe_weight_decay < 0:
        parser.error("probe hyperparameters must be positive (weight decay may be zero)")
    if args.topk_per_layer <= 0:
        parser.error("--topk-per-layer must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "localization", clean=args.clean)
    selected_json = output / "selected_neurons.json"
    if selected_json.exists() and not args.no_resume:
        print(f"reuse existing localization: {selected_json}", flush=True)
        return
    cache_path = Path(args.cache).resolve() if args.cache else (output.parent / "cache" / "probe_cache.npz").resolve()
    cache = np.load(cache_path, allow_pickle=False)
    try:
        hidden = np.asarray(cache["hidden"], dtype=np.float32)
        target = np.asarray(cache["target"], dtype=np.int64)
        split = np.asarray(cache["split"]).astype(str)
        layers = np.asarray(cache["layers"], dtype=np.int64)
    finally:
        cache.close()
    if hidden.ndim != 3 or len(layers) != hidden.shape[1]:
        raise ValueError(f"{cache_path}: expected [N,L,D] cache with matching layers, got {hidden.shape} and {layers.shape}")
    if args.topk_per_layer > hidden.shape[2]:
        raise ValueError(f"--topk-per-layer={args.topk_per_layer} exceeds hidden dimension {hidden.shape[2]}")
    train_mask, validation_mask = split == "train", split == "validation"
    if not train_mask.any() or not validation_mask.any():
        raise RuntimeError("probe cache must have train and validation samples")
    for name, mask in (("train", train_mask), ("validation", validation_mask)):
        if set(target[mask].tolist()) != {0, 1}:
            raise RuntimeError(f"{name} split must contain both pseudo-positive and pure-normal-negative samples")

    normal_train = hidden[train_mask & (target == 0)]
    if len(normal_train) < 2:
        raise RuntimeError("need at least two pure-normal training samples for z-score statistics")
    normal_mean = normal_train.mean(axis=0).astype(np.float32)
    normal_std = normal_train.std(axis=0, ddof=1).astype(np.float32)
    atomic_save_npy(output / "normal_mean.npy", normal_mean)
    atomic_save_npy(output / "normal_std.npy", normal_std)

    device = torch.device(args.device)
    metrics: list[dict[str, float | int]] = []
    score_rows: list[dict[str, float | int | bool]] = []
    probe_weights, probe_gradients = [], []
    for layer_index, layer_number in enumerate(tqdm(layers, desc="fit DNA linear probes", unit="layer")):
        train_x, validation_x = hidden[train_mask, layer_index], hidden[validation_mask, layer_index]
        train_y, validation_y = target[train_mask], target[validation_mask]
        probe = fit_linear_probe(
            train_x, train_y, validation_x, validation_y,
            args.probe_epochs, args.probe_lr, args.probe_weight_decay,
            device, args.seed + int(layer_number),
        )
        positive_center, negative_center = train_x[train_y == 1].mean(axis=0), train_x[train_y == 0].mean(axis=0)
        activations = np.abs(train_x).mean(axis=0)
        weights = np.asarray(probe["weight"])
        gradients = np.asarray(probe["mean_gradient"])
        triadic_score = np.abs(activations * gradients * weights)
        order = np.argsort(-triadic_score, kind="mergesort")
        selected = set(order[:args.topk_per_layer].tolist())
        metrics.append({
            "layer": int(layer_number), "layer_index": layer_index,
            "dcos": cosine_distance(positive_center, negative_center),
            "probe_accuracy": float(probe["validation_accuracy"]),
            "probe_auc": float(probe["validation_auc"]),
            "probe_ap": float(probe["validation_ap"]),
            "probe_train_loss": float(probe["final_train_loss"]),
        })
        for rank, dimension in enumerate(order, start=1):
            score_rows.append({
                "layer": int(layer_number), "layer_index": layer_index, "dimension": int(dimension),
                "rank_in_layer": rank, "mean_activation": float(activations[dimension]),
                "mean_gradient": float(gradients[dimension]), "probe_weight": float(weights[dimension]),
                "triadic_score": float(triadic_score[dimension]), "is_selected": int(dimension) in selected,
            })
        probe_weights.append(weights.astype(np.float32))
        probe_gradients.append(gradients.astype(np.float32))

    metrics_frame = pd.DataFrame(metrics)
    scores_frame = pd.DataFrame(score_rows).sort_values(["layer", "rank_in_layer"], kind="mergesort")
    write_csv(output / "layer_metrics.csv", metrics_frame.columns.tolist(), metrics_frame.itertuples(index=False, name=None))
    write_csv(output / "neuron_scores.csv", scores_frame.columns.tolist(), scores_frame.itertuples(index=False, name=None))
    selected_records = []
    for layer_index, layer_number in enumerate(layers):
        subset = scores_frame[(scores_frame["layer_index"] == layer_index) & scores_frame["is_selected"]]
        subset = subset.sort_values("rank_in_layer", kind="mergesort")
        selected_records.append({
            "layer": int(layer_number), "layer_index": layer_index,
            "dims": subset["dimension"].astype(int).tolist(),
            "scores": subset["triadic_score"].astype(float).tolist(),
            "mean_gradients": subset["mean_gradient"].astype(float).tolist(),
            "probe_weights": subset["probe_weight"].astype(float).tolist(),
        })
    selected_width = len(layers) * args.topk_per_layer
    atomic_save_npz(output / "linear_probes.npz", layers=layers, weight=np.stack(probe_weights), mean_gradient=np.stack(probe_gradients))
    save_json(selected_json, {
        "method": "dsanet_dna_triadic_probe_vadclip_xd_normal_negative",
        "description": "Per-layer linear probes use abnormal high VadCLIP pseudo-score snippets as positives and A-labelled pure-normal-video snippets as negatives. A neuron score is |mean activation × mean gradient × probe weight|.",
        "dataset": "xd", "clip_model": "ViT-B/16", "token_pool": "cls",
        "negative_source": "pure_normal_video_only", "negative_label": "A",
        "selection_mode": "topk_per_layer", "topk_per_layer": args.topk_per_layer,
        "num_layers": len(layers), "hidden_dim": int(hidden.shape[2]), "neuron_width": selected_width,
        "clip_dim": 512, "input_width": selected_width + 512,
        "feature_order": "zscored_selected_neurons_then_official_512d_clip",
        "normal_mean_path": relpath(output / "normal_mean.npy", output),
        "normal_std_path": relpath(output / "normal_std.npy", output),
        "probe_epochs": args.probe_epochs, "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay, "cache": relpath(cache_path, output),
        "selected": selected_records,
    })
    print(f"wrote {selected_json}: {args.topk_per_layer} neurons/layer × {len(layers)} layers = {selected_width}", flush=True)


if __name__ == "__main__":
    main()

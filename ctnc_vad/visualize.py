"""Render auditable CTNC-VAD evidence from saved test and audit artifacts.

The figure deliberately follows unit-dissection practice: it shows the
time-series decision, the individual hidden channels responsible for its most
important frame, and where those channels sit in the normal variance/PCA map.
It does not need to run or alter VadCLIP again.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .assets import load_assets
from .common import default_output_root, save_json, stage_dir, write_csv


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    try:
        return {key: np.asarray(value[key]) for key in value.files}
    finally:
        value.close()


def prediction_keys(prediction_dir: Path) -> list[str]:
    return sorted(path.stem for path in prediction_dir.glob("*.npz"))


def choose_most_reordered(prediction_dir: Path) -> str:
    best_key, best_change = "", float("-inf")
    for key in prediction_keys(prediction_dir):
        values = load_npz(prediction_dir / f"{key}.npz")
        if not {"prob2", "verified"} <= set(values):
            continue
        change = float(np.max(np.abs(values["verified"] - values["prob2"])))
        if change > best_change:
            best_key, best_change = key, change
    if not best_key:
        raise FileNotFoundError(f"no valid prediction artifacts in {prediction_dir}")
    return best_key


def channel_labels(assets: dict, indices: np.ndarray) -> list[str]:
    prompts = list(assets["prompts"])
    layers = assets["selected_layers"].cpu().numpy()
    dimensions = assets["selected_dimensions"].cpu().numpy()
    classes = assets["selected_text_class"].cpu().numpy()
    return [f"L{int(layers[i]) + 1} D{int(dimensions[i])} · {prompts[int(classes[i])]}" for i in indices]


def text_conditioned_contribution(
    audit: dict[str, np.ndarray], assets: dict, anomaly_class: int
) -> np.ndarray:
    """Reconstruct one declared anomaly text's contribution per raw channel."""
    delta = np.asarray(audit["channel_delta"], dtype=np.float32)
    gates = np.asarray(audit["channel_gates"], dtype=np.float32)
    affinity = assets["selected_text_affinity"].cpu().numpy().astype(np.float32)
    directions = np.sign(affinity[:, anomaly_class])
    directions[directions == 0] = 1.0
    affinity = np.abs(affinity)
    affinity /= np.maximum(affinity.sum(axis=-1, keepdims=True), 1e-6)
    directional_change = np.maximum(delta * directions.reshape(1, -1), 0.0)
    return directional_change * gates.reshape(1, -1) * affinity[:, anomaly_class].reshape(1, -1)


def make_figure(
    key: str,
    prediction: dict[str, np.ndarray],
    audit: dict[str, np.ndarray],
    assets: dict,
    topk: int,
    target: Path,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    baseline = np.asarray(prediction["prob2"], dtype=np.float32)
    final = np.asarray(prediction["verified"], dtype=np.float32)
    evidence = np.asarray(prediction.get("evidence", audit["hidden_anomaly"]), dtype=np.float32)
    semantic = np.asarray(audit["semantic_score"], dtype=np.float32)
    frame = int(np.argmax(np.abs(final - baseline)))
    class_evidence = np.asarray(audit["class_evidence"], dtype=np.float32)
    if class_evidence.ndim != 2 or len(class_evidence) != len(final):
        raise ValueError(f"{key}: audit class_evidence must be [T,C] aligned with predictions")
    anomaly_class = int(np.argmax(class_evidence[frame]))
    contribution = text_conditioned_contribution(audit, assets, anomaly_class)
    if contribution.ndim != 2 or len(contribution) != len(final):
        raise ValueError(f"{key}: audit channel evidence must be [T,K] aligned with predictions")
    global_top = np.argsort(-contribution.mean(axis=0))[:topk]
    if "class_top_channel_index" in audit:
        frame_top = np.asarray(audit["class_top_channel_index"], dtype=np.int64)[frame, :, anomaly_class]
    else:
        frame_top = np.argsort(-contribution[frame])[:topk]
    frame_top = frame_top[:topk]
    time = np.arange(len(final))
    layers = assets["selected_layers"].cpu().numpy()
    dimensions = assets["selected_dimensions"].cpu().numpy()
    variance = assets["normal_variance"].cpu().numpy()
    pca_energy = assets["normal_pca_coordinate_energy"].cpu().numpy()
    selected_variance = variance[layers, dimensions]
    selected_pca = pca_energy[layers, dimensions]
    classes = assets["selected_text_class"].cpu().numpy()

    prompt = list(assets["prompts"])[anomaly_class + 1]
    figure = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.05])
    ax_score = figure.add_subplot(grid[0, :])
    ax_heat = figure.add_subplot(grid[1, 0])
    ax_map = figure.add_subplot(grid[1, 1])
    ax_bar = figure.add_subplot(grid[2, :])
    figure.suptitle(f"CTNC hidden-channel explanation · {key}", fontsize=15)

    ax_score.plot(time, baseline, label="frozen baseline", linewidth=1.7, color="#4c78a8")
    ax_score.plot(time, final, label="CTNC re-ranked", linewidth=1.7, color="#e45756")
    ax_score.plot(time, evidence, label="channel-witness score", linewidth=1.2, color="#54a24b", alpha=0.9)
    ax_score.plot(time, semantic, label="frozen CLIP text confirmation", linewidth=1.1, color="#f58518", alpha=0.8, linestyle="--")
    ax_score.axvline(frame, color="#333333", linestyle="--", linewidth=1.0, label="largest re-rank")
    ax_score.set_xlim(0, max(1, len(final) - 1))
    ax_score.set_ylim(-0.03, 1.03)
    ax_score.set_xlabel("segment index")
    ax_score.set_ylabel("anomaly score")
    ax_score.set_title("A. What changed in temporal ranking")
    ax_score.legend(loc="upper right", ncol=3, fontsize=9)
    ax_score.grid(alpha=0.2)

    heat = contribution[:, global_top].T
    image = ax_heat.imshow(heat, aspect="auto", interpolation="nearest", cmap="magma")
    ax_heat.axvline(frame, color="cyan", linestyle="--", linewidth=1.0)
    ax_heat.set_yticks(np.arange(len(global_top)))
    ax_heat.set_yticklabels(channel_labels(assets, global_top), fontsize=8)
    ax_heat.set_xlabel("segment index")
    ax_heat.set_title(f"B. Top hidden channels over time for: {prompt}")
    figure.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04, label="channel contribution")

    ax_map.scatter(
        np.maximum(variance.reshape(-1), 1e-12), np.maximum(pca_energy.reshape(-1), 1e-12),
        s=5, alpha=0.12, color="#9e9e9e", label="all CLIP channels",
    )
    scatter = ax_map.scatter(
        np.maximum(selected_variance, 1e-12), np.maximum(selected_pca, 1e-12),
        c=classes, cmap="tab10", s=13, alpha=0.78, label="selected original channels",
    )
    ax_map.scatter(
        np.maximum(selected_variance[frame_top], 1e-12), np.maximum(selected_pca[frame_top], 1e-12),
        s=48, facecolors="none", edgecolors="#111111", linewidths=0.8, label="top at marked frame",
    )
    ax_map.set_xscale("log")
    ax_map.set_yscale("log")
    ax_map.set_xlabel("normal variance of original channel")
    ax_map.set_ylabel("truncated-PCA coordinate energy")
    ax_map.set_title("C. Why these channels were selected")
    ax_map.legend(loc="best", fontsize=8)
    colorbar = figure.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04)
    colorbar.set_label("assigned anomaly-text index")

    bar_values = contribution[frame, frame_top]
    bar_labels = channel_labels(assets, frame_top)
    positions = np.arange(len(frame_top))
    ax_bar.barh(positions, bar_values[::-1], color="#e45756", alpha=0.88)
    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels(bar_labels[::-1], fontsize=9)
    ax_bar.set_xlabel("text-directional normal departure × channel gate × fixed CLIP text affinity")
    ax_bar.set_title(f"D. Exact original-channel witnesses at segment {frame} for: {prompt}")
    ax_bar.grid(axis="x", alpha=0.2)

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return frame, anomaly_class, global_top, frame_top


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CTNC temporal and original-channel explanations from resumable test/audit artifacts.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--assets", required=True, help="discovery/circuit_assets.pt")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--audit-split-name", default="test")
    parser.add_argument("--video-key", action="append", default=[], help="Video key to render; repeat this flag for several videos.")
    parser.add_argument("--auto-top", type=int, default=0, help="When no --video-key is given, render this many videos with the largest final-vs-baseline change.")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only visualization/<split-name>/.")
    args = parser.parse_args()
    if args.topk <= 0 or args.auto_top < 0:
        parser.error("--topk must be positive and --auto-top must be non-negative")
    output_root = args.output_root or str(default_output_root(args.dataset))
    root = stage_dir(output_root, "visualization")
    output = root / args.split_name
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = Path(output_root) / "evaluation" / "predictions"
    audit_root = Path(output_root) / "audit" / args.audit_split_name
    if not evaluation.is_dir():
        raise FileNotFoundError(f"{evaluation} is absent; first run `python -m ctnc_vad.test ...`")
    if not audit_root.is_dir():
        raise FileNotFoundError(f"{audit_root} is absent; first run `python -m ctnc_vad.audit ...`")
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError(f"assets dataset={assets['dataset']!r} does not match --dataset={args.dataset!r}")
    keys = list(args.video_key)
    if not keys:
        values: list[tuple[float, str]] = []
        for key in prediction_keys(evaluation):
            prediction = load_npz(evaluation / f"{key}.npz")
            if {"prob2", "verified"} <= set(prediction):
                values.append((float(np.max(np.abs(prediction["verified"] - prediction["prob2"]))), key))
        keys = [key for _change, key in sorted(values, reverse=True)[:args.auto_top]]
    if not keys:
        raise ValueError("give --video-key or a positive --auto-top")

    rows: list[list[object]] = []
    for key in tqdm(keys, desc=f"CTNC visualization {args.split_name}", unit="video"):
        target = output / f"{key}.png"
        report = output / f"{key}_top_channels.csv"
        if target.is_file() and report.is_file():
            rows.append([key, target.name, report.name, "reused"])
            continue
        prediction = load_npz(evaluation / f"{key}.npz")
        audit = load_npz(audit_root / f"{key}.npz")
        frame, anomaly_class, global_top, frame_top = make_figure(key, prediction, audit, assets, args.topk, target)
        prompts = list(assets["prompts"])
        layers = assets["selected_layers"].cpu().numpy()
        dimensions = assets["selected_dimensions"].cpu().numpy()
        classes = assets["selected_text_class"].cpu().numpy()
        directions = assets["selected_text_direction"].cpu().numpy()
        gates = np.asarray(audit["channel_gates"], dtype=np.float32)
        contribution = text_conditioned_contribution(audit, assets, anomaly_class)
        write_csv(
            report,
            ["rank_at_explained_frame", "explained_anomaly_text", "circuit_index", "layer_1based", "dimension", "channel_assigned_anomaly_text", "text_direction", "learned_gate", "contribution", "mean_contribution"],
            [
                [rank + 1, prompts[anomaly_class + 1], int(index), int(layers[index]) + 1, int(dimensions[index]), prompts[int(classes[index])], float(directions[index]), float(gates[index]), float(contribution[frame, index]), float(contribution[:, index].mean())]
                for rank, index in enumerate(frame_top)
            ],
        )
        rows.append([key, target.name, report.name, "new"])
        save_json(output / f"{key}_summary.json", {
            "video_key": key,
            "explained_segment_index": frame,
            "explained_anomaly_text": prompts[anomaly_class + 1],
            "top_channels_over_video": [int(item) for item in global_top],
            "top_channels_at_explained_frame": [int(item) for item in frame_top],
            "baseline_score": float(prediction["prob2"][frame]),
            "final_score": float(prediction["verified"][frame]),
            "channel_witness_score": float(prediction["evidence"][frame]),
            "frozen_clip_text_confirmation": float(audit["semantic_score"][frame]),
        })
    write_csv(output / "index.csv", ["video_key", "figure", "top_channel_csv", "action"], rows)
    print(f"wrote {len(rows)} channel-explanation figures under {output}", flush=True)


if __name__ == "__main__":
    main()

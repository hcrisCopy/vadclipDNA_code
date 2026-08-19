"""Render intrinsic class-specific raw-channel explanations from audit artifacts."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .assets import load_assets
from .common import default_output_root, save_json, stage_dir, write_csv


def load_npz(path: Path) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=False)
    try:
        return {name: np.asarray(value[name]) for name in value.files}
    finally:
        value.close()


def labels(assets: dict, indices: np.ndarray, class_index: int) -> list[str]:
    layers, dimensions = assets["selected_layers"].cpu().numpy(), assets["selected_dimensions"].cpu().numpy()
    response = assets["semantic_response"].cpu().numpy()
    sign = ["+" if response[int(index), class_index] >= 0 else "-" for index in indices]
    return [f"L{int(layers[index]) + 1} D{int(dimensions[index])} ({direction})" for index, direction in zip(indices, sign)]


def choose_keys(audit_root: Path, count: int) -> list[str]:
    changes: list[tuple[float, str]] = []
    for path in audit_root.glob("*.npz"):
        value = load_npz(path)
        changes.append((float(np.max(np.abs(value["fused"] - value["baseline"]))), path.stem))
    return [key for _change, key in sorted(changes, reverse=True)[:count]]


def make_figure(key: str, audit: dict[str, np.ndarray], assets: dict, topk: int, target: Path) -> tuple[int, int, np.ndarray]:
    baseline, fused = audit["baseline"], audit["fused"]
    frame = int(np.argmax(np.abs(fused - baseline)))
    class_index = int(np.argmax(audit["circuit_class_probability"][frame]))
    indices = audit["top_channel_index"][:, class_index]
    values = audit["top_channel_contribution"][:, class_index]
    all_indices = indices.reshape(-1)
    all_values = values.reshape(-1)
    sums = np.bincount(all_indices, weights=all_values, minlength=len(assets["selected_layers"]))
    chosen = np.argsort(-sums)[:topk]
    heat = np.zeros((len(chosen), len(fused)), dtype=np.float32)
    for time in range(len(fused)):
        for rank, index in enumerate(indices[time]):
            matching = np.flatnonzero(chosen == index)
            if len(matching):
                heat[matching[0], time] = values[time, rank]
    prompt = list(assets["prompts"])[class_index + 1]
    figure = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.0])
    score_axis = figure.add_subplot(grid[0, :])
    heat_axis = figure.add_subplot(grid[1, 0])
    map_axis = figure.add_subplot(grid[1, 1])
    bar_axis = figure.add_subplot(grid[2, :])
    time = np.arange(len(fused))
    score_axis.plot(time, baseline, label="frozen baseline", color="#4c78a8", linewidth=1.7)
    score_axis.plot(time, fused, label="CTSC classwise PoE", color="#e45756", linewidth=1.7)
    score_axis.plot(time, audit["circuit_class_probability"][:, class_index], label=f"{prompt} raw-channel circuit", color="#54a24b", linewidth=1.2)
    score_axis.axvline(frame, color="#222", linestyle="--", linewidth=1, label="explained segment")
    score_axis.set_ylim(-0.03, 1.03)
    score_axis.set_xlim(0, max(1, len(fused) - 1))
    score_axis.set_xlabel("segment index")
    score_axis.set_ylabel("score")
    score_axis.set_title("A. Frozen baseline and independently computed channel circuit")
    score_axis.legend(ncol=4, fontsize=9)
    score_axis.grid(alpha=0.2)
    image = heat_axis.imshow(heat, aspect="auto", interpolation="nearest", cmap="magma")
    heat_axis.axvline(frame, color="cyan", linestyle="--", linewidth=1)
    heat_axis.set_yticks(np.arange(len(chosen)))
    heat_axis.set_yticklabels(labels(assets, chosen, class_index), fontsize=8)
    heat_axis.set_xlabel("segment index")
    heat_axis.set_title(f"B. Intrinsic Top raw channels for: {prompt}")
    figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04, label="direct channel contribution")
    layers, dimensions = assets["selected_layers"].cpu().numpy(), assets["selected_dimensions"].cpu().numpy()
    variance = assets["normal_variance"].cpu().numpy()[layers, dimensions]
    pca = assets["normal_pca_coordinate_energy"].cpu().numpy()[layers, dimensions]
    weights = audit["normalized_channel_weight"][:, class_index]
    scatter = map_axis.scatter(np.maximum(variance, 1e-12), np.maximum(pca, 1e-12), c=weights, cmap="viridis", s=18, alpha=0.8)
    map_axis.scatter(np.maximum(variance[chosen], 1e-12), np.maximum(pca[chosen], 1e-12), s=52, facecolors="none", edgecolors="#111", linewidths=0.8)
    map_axis.set_xscale("log")
    map_axis.set_yscale("log")
    map_axis.set_xlabel("normal variance of original channel")
    map_axis.set_ylabel("normal PCA coordinate energy")
    map_axis.set_title("C. Which discovered raw channels the circuit actually retained")
    figure.colorbar(scatter, ax=map_axis, fraction=0.046, pad=0.04, label="learned direct class weight")
    frame_indices, frame_values = indices[frame], values[frame]
    order = np.argsort(frame_values)[::-1]
    frame_indices, frame_values = frame_indices[order], frame_values[order]
    bar_axis.barh(np.arange(len(frame_indices)), frame_values[::-1], color="#e45756")
    bar_axis.set_yticks(np.arange(len(frame_indices)))
    bar_axis.set_yticklabels(labels(assets, frame_indices[::-1], class_index), fontsize=9)
    bar_axis.set_xlabel("raw z-score along text direction × direct class weight")
    bar_axis.set_title(f"D. Exact witnesses for {prompt} at segment {frame}")
    bar_axis.grid(axis="x", alpha=0.2)
    figure.suptitle(f"CTSC raw hidden-channel explanation · {key}", fontsize=15)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return frame, class_index, frame_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CTSC class-specific raw hidden-channel evidence from audit artifacts.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--audit-split-name", default="test")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--video-key", action="append", default=[])
    parser.add_argument("--auto-top", type=int, default=0)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only visualization/<split-name>/ under --output-root.")
    args = parser.parse_args()
    if args.topk <= 0 or args.auto_top < 0:
        parser.error("--topk must be positive and --auto-top non-negative")
    output_root = args.output_root or str(default_output_root(args.dataset))
    root = stage_dir(output_root, "visualization")
    output = root / args.split_name
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_root = Path(output_root) / "audit" / args.audit_split_name
    if not audit_root.is_dir():
        raise FileNotFoundError(f"{audit_root} is absent; run ctsc_vad.audit first")
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError("asset dataset does not match --dataset")
    keys = list(args.video_key) or choose_keys(audit_root, args.auto_top)
    if not keys:
        raise ValueError("give --video-key or a positive --auto-top")
    rows: list[list[object]] = []
    for key in tqdm(keys, desc=f"CTSC visualization {args.split_name}", unit="video"):
        target, report = output / f"{key}.png", output / f"{key}_top_channels.csv"
        if target.is_file() and report.is_file():
            rows.append([key, target.name, report.name, "reused"])
            continue
        audit = load_npz(audit_root / f"{key}.npz")
        frame, class_index, selected = make_figure(key, audit, assets, args.topk, target)
        prompts, layers, dimensions, response = list(assets["prompts"]), assets["selected_layers"].cpu().numpy(), assets["selected_dimensions"].cpu().numpy(), assets["semantic_response"].cpu().numpy()
        index_at_frame, contribution_at_frame, zscore_at_frame = audit["top_channel_index"][frame, class_index], audit["top_channel_contribution"][frame, class_index], audit["top_channel_zscore"][frame, class_index]
        order = np.argsort(contribution_at_frame)[::-1]
        write_csv(report, ["rank", "explained_anomaly_text", "circuit_index", "layer_1based", "dimension", "signed_text_response", "raw_zscore", "direct_weight", "contribution"], [[rank + 1, prompts[class_index + 1], int(index_at_frame[item]), int(layers[index_at_frame[item]]) + 1, int(dimensions[index_at_frame[item]]), float(response[index_at_frame[item], class_index]), float(zscore_at_frame[item]), float(audit["normalized_channel_weight"][index_at_frame[item], class_index]), float(contribution_at_frame[item])] for rank, item in enumerate(order)])
        save_json(output / f"{key}_summary.json", {"video_key": key, "explained_segment_index": frame, "explained_anomaly_text": prompts[class_index + 1], "top_circuit_indices": [int(value) for value in selected], "baseline_score": float(audit["baseline"][frame]), "fused_score": float(audit["fused"][frame])})
        rows.append([key, target.name, report.name, "new"])
    write_csv(output / "index.csv", ["video_key", "figure", "top_channel_csv", "action"], rows)
    print(f"wrote {len(rows)} raw-channel explanation figures under {output}", flush=True)


if __name__ == "__main__":
    main()

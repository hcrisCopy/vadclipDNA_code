"""Discover text-grounded raw CLIP channel candidates and normal contexts.

Only normal videos determine normal statistics.  Neither this script nor its
assets reads a VadCLIP score.  PCA/SVD is used solely to rank *original*
coordinates; no PCA component is ever passed to the circuit reader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import add_vadclip_source
from .common import (
    atomic_torch_save, default_output_root, grouped_source_rows, hidden_manifest_paths,
    is_normal_video, kmeans_unit_vectors, labels_for_dataset, load_hidden, normal_label,
    normalize_rows, read_source_csv, relpath, save_json, stage_dir, write_csv,
)
from .temporal import check_odd_window, numpy_temporal_dynamics


def uniform_indices(length: int, maximum: int) -> np.ndarray:
    return np.linspace(0, length - 1, min(length, maximum), dtype=np.int64)


def rank_per_layer(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for layer in range(values.shape[0]):
        order = np.argsort(values[layer], kind="stable")
        result[layer, order] = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    return result


def load_text_route(model_name: str, prompts: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    add_vadclip_source()
    from clip import clip

    model, _ = clip.load(model_name, device=str(device))
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        embeddings = model.encode_token(tokens)
        text = torch.nn.functional.normalize(model.encode_text(embeddings, tokens).float(), dim=-1).cpu().numpy()
    visual = model.visual
    weight = visual.ln_post.weight.detach().float().cpu().numpy()
    bias = visual.ln_post.bias.detach().float().cpu().numpy()
    projection, eps = visual.proj.detach().float().cpu().numpy(), float(visual.ln_post.eps)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weight, bias, projection, eps, text


def final_visual(last_hidden: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float, projection: np.ndarray) -> np.ndarray:
    mean = last_hidden.mean(axis=-1, keepdims=True)
    variance = np.square(last_hidden - mean).mean(axis=-1, keepdims=True)
    return normalize_rows(((last_hidden - mean) / np.sqrt(variance + eps) * weight + bias) @ projection)


def randomized_coordinate_energy(normal_rows, global_mean: np.ndarray, global_std: np.ndarray, rank: int, frames: int, seed: int) -> np.ndarray:
    """Normal truncated-SVD energy per original coordinate, without full covariance."""
    layers, width = global_mean.shape
    sketch_width = min(width, rank + 8)
    probe = np.random.default_rng(seed).standard_normal((width, sketch_width)).astype(np.float32)
    sketch = np.zeros((layers, width, sketch_width), dtype=np.float64)
    for _key, _label, path in tqdm(normal_rows, desc="normal SVD range", unit="video"):
        hidden = load_hidden(path)
        hidden = hidden[uniform_indices(len(hidden), frames)]
        standardized = (hidden - global_mean[None]) / global_std[None]
        for layer in range(layers):
            values = standardized[:, layer]
            sketch[layer] += values.T @ (values @ probe)
    basis = np.empty_like(sketch)
    for layer in range(layers):
        basis[layer] = np.linalg.qr(sketch[layer], mode="reduced")[0]
    covariance = np.zeros((layers, sketch_width, sketch_width), dtype=np.float64)
    for _key, _label, path in tqdm(normal_rows, desc="normal SVD spectrum", unit="video"):
        hidden = load_hidden(path)
        hidden = hidden[uniform_indices(len(hidden), frames)]
        standardized = (hidden - global_mean[None]) / global_std[None]
        for layer in range(layers):
            projected = standardized[:, layer] @ basis[layer]
            covariance[layer] += projected.T @ projected
    energy = np.empty((layers, width), dtype=np.float32)
    for layer in range(layers):
        values, vectors = np.linalg.eigh(covariance[layer])
        selected_values, selected_vectors = values[-rank:], vectors[:, -rank:]
        principal = basis[layer] @ selected_vectors
        energy[layer] = np.sum(np.square(principal) * selected_values[None], axis=1).astype(np.float32)
    return energy


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover CTSC raw channel candidates and context-normalized reference statistics.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--candidate-per-layer", type=int, default=128)
    parser.add_argument("--context-count", type=int, default=16)
    parser.add_argument("--context-iters", type=int, default=50)
    parser.add_argument("--global-subspace-rank", type=int, default=16)
    parser.add_argument("--global-subspace-frames", type=int, default=4)
    parser.add_argument("--semantic-frames-per-video", type=int, default=128)
    parser.add_argument("--temporal-short-window", type=int, default=5, help="Odd local window for short-vs-long channel change.")
    parser.add_argument("--temporal-long-window", type=int, default=21, help="Odd normal-context window; must exceed --temporal-short-window.")
    parser.add_argument("--temporal-persistence-window", type=int, default=5, help="Odd local support window used by the fixed circuit certificate.")
    parser.add_argument("--std-floor", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-hidden-manifest", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only discovery/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if min(args.candidate_per_layer, args.context_count, args.context_iters, args.global_subspace_rank, args.global_subspace_frames, args.semantic_frames_per_video) <= 0:
        parser.error("candidate/context/SVD/frame counts must be positive")
    if args.std_floor <= 0 or args.ridge < 0:
        parser.error("--std-floor must be positive and --ridge non-negative")
    try:
        args.temporal_short_window = check_odd_window("--temporal-short-window", args.temporal_short_window)
        args.temporal_long_window = check_odd_window("--temporal-long-window", args.temporal_long_window)
        args.temporal_persistence_window = check_odd_window("--temporal-persistence-window", args.temporal_persistence_window)
    except ValueError as error:
        parser.error(str(error))
    if args.temporal_short_window >= args.temporal_long_window:
        parser.error("--temporal-short-window must be smaller than --temporal-long-window")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "discovery", clean=args.clean)
    asset_path = output / "ctsc_assets.pt"
    if asset_path.is_file() and not args.no_resume:
        print(f"reuse {asset_path}; pass --clean or --no-resume to rebuild", flush=True)
        return

    groups = grouped_source_rows(read_source_csv(args.source_train_csv))
    paths = hidden_manifest_paths(args.hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to)
    rows, missing = [], []
    for key, group in groups.items():
        if key not in paths:
            if args.strict_hidden_manifest:
                raise FileNotFoundError(f"{key}: absent from {args.hidden_manifest}")
            missing.append((key, str(group.iloc[0]["label"])))
        else:
            rows.append((key, str(group.iloc[0]["label"]), paths[key]))
    normal_rows = [row for row in rows if is_normal_video(args.dataset, row[1])]
    if not normal_rows:
        raise ValueError("discovery requires normal training videos")
    first = load_hidden(normal_rows[0][2])
    layers, width = first.shape[1:]
    if (layers, width) != (12, 768) or args.candidate_per_layer > width or args.global_subspace_rank >= width:
        raise ValueError("cached hidden contract must be [T,12,768] with valid candidate/SVD ranks")

    total, squared, count, signatures, normal_keys = np.zeros((layers, width), np.float64), np.zeros((layers, width), np.float64), 0, [], []
    for key, _label, path in tqdm(normal_rows, desc="normal moments", unit="video"):
        hidden = load_hidden(path)
        total += hidden.sum(axis=0, dtype=np.float64)
        squared += np.square(hidden, dtype=np.float64).sum(axis=0)
        count += len(hidden)
        signatures.append(normalize_rows(hidden[:, -1].mean(axis=0, keepdims=True))[0])
        normal_keys.append(key)
    global_mean = total / count
    global_std = np.sqrt(np.maximum(squared / count - np.square(global_mean), args.std_floor ** 2))
    pca_energy = randomized_coordinate_energy(normal_rows, global_mean, global_std, args.global_subspace_rank, args.global_subspace_frames, args.seed)
    centers, assignments = kmeans_unit_vectors(np.stack(signatures), args.context_count, args.context_iters, args.seed)

    prompts = list(labels_for_dataset(args.dataset).values())
    ln_weight, ln_bias, projection, ln_eps, text_features = load_text_route(args.clip_model, prompts, torch.device(args.device))
    cross, second = np.zeros((layers, width, 512), np.float64), np.zeros((layers, width), np.float64)
    for _key, _label, path in tqdm(normal_rows, desc="fit channel-text probes", unit="video"):
        hidden = load_hidden(path)
        sampled = hidden[uniform_indices(len(hidden), args.semantic_frames_per_video)]
        target = final_visual(sampled[:, -1], ln_weight, ln_bias, ln_eps, projection)
        centered = sampled - global_mean[None]
        for layer in range(layers):
            cross[layer] += centered[:, layer].T @ target
            second[layer] += np.square(centered[:, layer]).sum(axis=0)
    semantic_lens = cross / (second[..., None] + args.ridge)
    semantic_response = np.einsum("ldk,ck->ldc", semantic_lens, text_features[1:] - text_features[:1]).astype(np.float32)
    semantic_strength = np.abs(semantic_response).max(axis=-1)
    normal_variance = np.square(global_std).astype(np.float32)
    # Preserve raw-coordinate normal-subspace coverage, but no longer prefer
    # the noisiest normal scene coordinates. The dominant signal is the frozen
    # CLIP text response; normal stability is used only as a tie breaker.
    normal_stability = 1.0 - rank_per_layer(normal_variance)
    score = 0.65 * rank_per_layer(semantic_strength) + 0.25 * rank_per_layer(pca_energy) + 0.10 * normal_stability
    chosen = np.argsort(-score, axis=1, kind="stable")[:, :args.candidate_per_layer]
    selected_layers = np.repeat(np.arange(layers, dtype=np.int64), args.candidate_per_layer)
    selected_dimensions = chosen.reshape(-1).astype(np.int64)
    selected_response = semantic_response[selected_layers, selected_dimensions]

    contexts, selected_width = len(centers), len(selected_layers)
    context_sum = np.zeros((contexts, selected_width), np.float64)
    context_square = np.zeros_like(context_sum)
    context_count = np.zeros(contexts, np.int64)
    key_context = dict(zip(normal_keys, assignments))
    for key, _label, path in tqdm(normal_rows, desc="context channel moments", unit="video"):
        values = load_hidden(path)[:, selected_layers, selected_dimensions]
        context = int(key_context[key])
        context_sum[context] += values.sum(axis=0, dtype=np.float64)
        context_square[context] += np.square(values, dtype=np.float64).sum(axis=0)
        context_count[context] += len(values)
    fallback_mean = global_mean[selected_layers, selected_dimensions]
    fallback_std = global_std[selected_layers, selected_dimensions]
    context_mean, context_std = np.empty_like(context_sum, np.float32), np.empty_like(context_sum, np.float32)
    for context in range(contexts):
        if context_count[context] == 0:
            context_mean[context], context_std[context] = fallback_mean, fallback_std
        else:
            mean = context_sum[context] / context_count[context]
            context_mean[context] = mean
            context_std[context] = np.sqrt(np.maximum(context_square[context] / context_count[context] - np.square(mean), args.std_floor ** 2))
    # Static normality is insufficient for VAD.  Record the ordinary velocity
    # and short-vs-long trajectory of every selected original coordinate,
    # separately for each normal scene context.  No anomaly label or VadCLIP
    # output participates in these statistics.
    temporal_sum = np.zeros((contexts, selected_width, 2), np.float64)
    temporal_square = np.zeros_like(temporal_sum)
    temporal_count = np.zeros(contexts, np.int64)
    global_temporal_sum = np.zeros((selected_width, 2), np.float64)
    global_temporal_square = np.zeros_like(global_temporal_sum)
    global_temporal_count = 0
    for key, _label, path in tqdm(normal_rows, desc="normal temporal channel moments", unit="video"):
        values = load_hidden(path)[:, selected_layers, selected_dimensions]
        context = int(key_context[key])
        zscore = (values - context_mean[context]) / context_std[context]
        dynamics = numpy_temporal_dynamics(zscore, args.temporal_short_window, args.temporal_long_window)
        temporal_sum[context] += dynamics.sum(axis=0, dtype=np.float64)
        temporal_square[context] += np.square(dynamics, dtype=np.float64).sum(axis=0)
        temporal_count[context] += len(dynamics)
        global_temporal_sum += dynamics.sum(axis=0, dtype=np.float64)
        global_temporal_square += np.square(dynamics, dtype=np.float64).sum(axis=0)
        global_temporal_count += len(dynamics)
    global_temporal_mean = global_temporal_sum / max(1, global_temporal_count)
    global_temporal_std = np.sqrt(np.maximum(global_temporal_square / max(1, global_temporal_count) - np.square(global_temporal_mean), args.std_floor ** 2))
    context_temporal_mean = np.empty_like(temporal_sum, np.float32)
    context_temporal_std = np.empty_like(temporal_sum, np.float32)
    for context in range(contexts):
        if temporal_count[context] == 0:
            context_temporal_mean[context], context_temporal_std[context] = global_temporal_mean, global_temporal_std
        else:
            mean = temporal_sum[context] / temporal_count[context]
            context_temporal_mean[context] = mean
            context_temporal_std[context] = np.sqrt(np.maximum(temporal_square[context] / temporal_count[context] - np.square(mean), args.std_floor ** 2))
    assets = {
        "version": 2, "dataset": args.dataset, "prompts": prompts, "clip_model": args.clip_model,
        "hidden_layers": layers, "hidden_width": width,
        "selected_layers": torch.from_numpy(selected_layers), "selected_dimensions": torch.from_numpy(selected_dimensions),
        "semantic_response": torch.from_numpy(selected_response),
        "context_centers": torch.from_numpy(centers), "context_mean": torch.from_numpy(context_mean), "context_std": torch.from_numpy(context_std),
        "context_temporal_mean": torch.from_numpy(context_temporal_mean), "context_temporal_std": torch.from_numpy(context_temporal_std),
        "temporal_short_window": args.temporal_short_window, "temporal_long_window": args.temporal_long_window,
        "temporal_persistence_window": args.temporal_persistence_window,
        "normal_variance": torch.from_numpy(normal_variance), "normal_pca_coordinate_energy": torch.from_numpy(pca_energy),
    }
    atomic_torch_save(asset_path, assets)
    rows_for_csv = []
    for index, (layer, dimension) in enumerate(zip(selected_layers, selected_dimensions)):
        text_index = int(np.abs(selected_response[index]).argmax())
        rows_for_csv.append([index, int(layer) + 1, int(dimension), float(normal_variance[layer, dimension]), float(pca_energy[layer, dimension]), prompts[text_index + 1], float(selected_response[index, text_index])])
    write_csv(output / "selected_raw_channels.csv", ["circuit_index", "layer_1based", "dimension", "normal_variance", "normal_pca_energy", "dominant_text", "signed_text_response"], rows_for_csv)
    write_csv(output / "missing_hidden.csv", ["video_key", "label"], missing)
    save_json(output / "summary.json", {"dataset": args.dataset, "normal_label": normal_label(args.dataset), "normal_videos": len(normal_rows), "missing_videos": len(missing), "candidate_per_layer": args.candidate_per_layer, "selected_width": selected_width, "context_count": contexts, "temporal_operators": ["text_aligned_level", "text_aligned_velocity", "opposite_text_velocity", "text_aligned_short_long_shift", "persistent_text_aligned_level"], "temporal_windows": {"short": args.temporal_short_window, "long": args.temporal_long_window, "persistence": args.temporal_persistence_window}, "selection": "raw-coordinate CLIP text response plus normal-subspace coverage and stability; no VadCLIP score and no anomaly label", "assets": relpath(asset_path, output)})
    print(f"wrote {asset_path}: {selected_width} text-grounded original channels with normal temporal references, contexts={contexts}, missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()

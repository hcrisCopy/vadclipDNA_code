"""Discover a sparse, text-grounded CLIP normality circuit from cached hidden states.

This stage never reads a VadCLIP anomaly score.  It uses only pure-normal
videos, video-level weak labels, the frozen CLIP text encoder and the supplied
``[T,L,D]`` CLS hidden-state artifacts.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import add_vadclip_source
from .common import (
    base_key,
    default_output_root,
    grouped_source_rows,
    hidden_manifest_paths,
    kmeans_unit_vectors,
    labels_for_dataset,
    load_hidden,
    normal_label,
    is_normal_video,
    normalize_rows,
    read_source_csv,
    relpath,
    save_json,
    stage_dir,
    write_csv,
    atomic_torch_save,
)


def uniform_indices(length: int, maximum: int) -> np.ndarray:
    return np.linspace(0, length - 1, min(length, maximum), dtype=np.int64)


def project_last_hidden(hidden: np.ndarray, ln_weight: np.ndarray, ln_bias: np.ndarray, eps: float, projection: np.ndarray) -> np.ndarray:
    """Exact final CLS ``ln_post + proj`` route, implemented from cached layer-12 states."""
    mean = hidden.mean(axis=-1, keepdims=True)
    variance = ((hidden - mean) ** 2).mean(axis=-1, keepdims=True)
    post = (hidden - mean) / np.sqrt(variance + eps)
    post = post * ln_weight + ln_bias
    visual = post @ projection
    return normalize_rows(visual)


def percentile_tail_mean(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(values) * fraction)))
    tail = np.partition(values, len(values) - count, axis=0)[-count:]
    return tail.mean(axis=0)


def rank_within_layer(values: np.ndarray) -> np.ndarray:
    rank = np.empty_like(values, dtype=np.float32)
    for layer in range(values.shape[0]):
        order = np.argsort(values[layer], kind="stable")
        rank[layer, order] = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
    return rank


def load_clip_text_route(model_name: str, prompts: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Load only the frozen OpenAI CLIP parameters needed for text grounding."""
    add_vadclip_source()
    from clip import clip

    model, _ = clip.load(model_name, device=str(device))
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        # VadCLIP's bundled CLIP fork exposes the prompt-tuning interface
        # ``encode_text(text_embeddings, token_ids)`` rather than OpenAI
        # CLIP's one-argument convenience method.
        text_embeddings = model.encode_token(tokens)
        text = torch.nn.functional.normalize(model.encode_text(text_embeddings, tokens).float(), dim=-1).cpu().numpy()
    visual = model.visual
    weight = visual.ln_post.weight.detach().float().cpu().numpy()
    bias = visual.ln_post.bias.detach().float().cpu().numpy()
    projection = visual.proj.detach().float().cpu().numpy()
    eps = float(visual.ln_post.eps)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weight, bias, projection, eps, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover a baseline-independent CTNC normality circuit from reusable CLIP hidden states.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-train-csv", required=True, help="Original VadCLIP train path,label CSV.")
    parser.add_argument("--hidden-manifest", required=True, help="Reusable CLS hidden manifest with key,hidden_path columns.")
    parser.add_argument("--hidden-path-base", default=".", help="Base for relative hidden paths recorded in the manifest.")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--output-root", default="", help="Defaults to ../vadclipDNA_data/<dataset>_ctnc_vad.")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--candidate-per-layer", type=int, default=32)
    parser.add_argument("--context-count", type=int, default=16)
    parser.add_argument("--context-iters", type=int, default=50)
    parser.add_argument("--frames-per-video", type=int, default=128, help="Maximum cached frames per video used for semantic-lens fitting.")
    parser.add_argument("--tail-fraction", type=float, default=0.125, help="Bag-level upper-tail fraction; never uses a baseline score.")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Weight of frozen text-lens evidence in channel selection.")
    parser.add_argument("--ridge", type=float, default=1e-3, help="Diagonal ridge for the cached hidden-to-final semantic lens.")
    parser.add_argument("--std-floor", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--strict-hidden-manifest",
        action="store_true",
        help="Fail instead of intersecting the source CSV with the hidden manifest. By default, missing training hidden states are skipped and recorded.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only discovery/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute discovery even when discovery/circuit_assets.pt exists.")
    args = parser.parse_args()
    if not 0 < args.tail_fraction <= 1 or not 0 <= args.semantic_weight <= 1:
        parser.error("--tail-fraction must be in (0,1] and --semantic-weight must be in [0,1]")
    if min(args.candidate_per_layer, args.context_count, args.frames_per_video, args.context_iters) <= 0:
        parser.error("candidate/context/frame counts and context iterations must be positive")
    if args.std_floor <= 0 or args.ridge < 0:
        parser.error("--std-floor must be positive and --ridge must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "discovery", clean=args.clean)
    asset_path = output / "circuit_assets.pt"
    if asset_path.is_file() and not args.no_resume:
        print(f"reuse {asset_path}; pass --no-resume or --clean to rebuild discovery", flush=True)
        return

    groups = grouped_source_rows(read_source_csv(args.source_train_csv))
    hidden_by_key = hidden_manifest_paths(
        args.hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to
    )
    normal = normal_label(args.dataset)
    rows: list[tuple[str, str, Path]] = []
    missing: list[tuple[str, str]] = []
    for key, group in groups.items():
        path = hidden_by_key.get(key)
        if path is None:
            if args.strict_hidden_manifest:
                raise FileNotFoundError(f"{key}: absent from {args.hidden_manifest}")
            missing.append((key, str(group.iloc[0]["label"])))
            continue
        rows.append((key, str(group.iloc[0]["label"]), path))
    normal_rows = [row for row in rows if is_normal_video(args.dataset, row[1])]
    abnormal_rows = [row for row in rows if not is_normal_video(args.dataset, row[1])]
    if not normal_rows or not abnormal_rows:
        raise ValueError("discovery needs both pure-normal and abnormal video bags")

    first = load_hidden(normal_rows[0][2])
    layers, width = int(first.shape[1]), int(first.shape[2])
    if width != 768 or args.candidate_per_layer > width:
        raise ValueError(f"candidate-per-layer must be <= hidden width 768, got {args.candidate_per_layer}")

    # Pass 1: pure-normal global moments and video signatures.  No anomaly score is involved.
    state_sum = np.zeros((layers, width), dtype=np.float64)
    state_square_sum = np.zeros_like(state_sum)
    state_count = 0
    signatures: list[np.ndarray] = []
    normal_keys: list[str] = []
    for key, _label, path in tqdm(normal_rows, desc="normal moments", unit="video"):
        hidden = load_hidden(path)
        if hidden.shape[1:] != (layers, width):
            raise ValueError(f"{key}: hidden shape {hidden.shape} differs from [{layers},{width}]")
        state_sum += hidden.sum(axis=0, dtype=np.float64)
        state_square_sum += np.square(hidden, dtype=np.float64).sum(axis=0)
        state_count += len(hidden)
        signatures.append(normalize_rows(hidden[:, -1, :].mean(axis=0, keepdims=True))[0])
        normal_keys.append(key)
    global_mean = state_sum / max(1, state_count)
    global_std = np.sqrt(np.maximum(state_square_sum / max(1, state_count) - np.square(global_mean), args.std_floor ** 2))
    context_centers, context_assignments = kmeans_unit_vectors(
        np.stack(signatures), args.context_count, args.context_iters, args.seed
    )
    context_of_normal = {key: int(context) for key, context in zip(normal_keys, context_assignments)}

    prompts = list(labels_for_dataset(args.dataset).values())
    ln_weight, ln_bias, projection, ln_eps, text_features = load_clip_text_route(args.clip_model, prompts, torch.device(args.device))
    if projection.shape != (width, 512) or text_features.shape[1] != 512:
        raise ValueError("the selected CLIP model is incompatible with cached ViT-B/16 hidden states")

    # Pass 2: fit a diagonal semantic lens from each cached layer to CLIP's final image-text space.
    cross = np.zeros((layers, width, 512), dtype=np.float64)
    second = np.zeros((layers, width), dtype=np.float64)
    for key, _label, path in tqdm(normal_rows, desc="fit semantic lenses", unit="video"):
        hidden = load_hidden(path)
        index = uniform_indices(len(hidden), args.frames_per_video)
        sampled = hidden[index]
        target = project_last_hidden(sampled[:, -1, :], ln_weight, ln_bias, ln_eps, projection)
        centered = sampled - global_mean[None]
        for layer in range(layers):
            cross[layer] += centered[:, layer, :].T @ target
            second[layer] += np.square(centered[:, layer, :]).sum(axis=0)
    semantic_lens = cross / (second[..., None] + float(args.ridge))
    # ``response[l,d,c]`` is signed: a positive (negative) activation change
    # in hidden coordinate ``d`` moves layer ``l`` toward (away from) anomaly
    # text concept ``c`` relative to normal text.  Keeping this sign is the
    # key distinction from the old direction-free absolute deviation score.
    text_directions = text_features[1:] - text_features[:1]
    semantic_response = np.einsum("ldk,ck->ldc", semantic_lens, text_directions).astype(np.float32)
    semantic_score = np.abs(semantic_response).max(axis=-1).astype(np.float32)

    # Pass 3: weak bag statistics.  The tail is over a hidden coordinate, not a baseline prediction.
    normal_tail_sum = np.zeros((layers, width), dtype=np.float64)
    abnormal_tail_sum = np.zeros_like(normal_tail_sum)
    normal_bags = abnormal_bags = 0
    for key, label, path in tqdm(rows, desc="bag-level circuit evidence", unit="video"):
        hidden = load_hidden(path)
        if hidden.shape[1:] != (layers, width):
            raise ValueError(f"{key}: hidden shape {hidden.shape} differs from [{layers},{width}]")
        tail = percentile_tail_mean(np.abs((hidden - global_mean) / global_std), args.tail_fraction)
        if is_normal_video(args.dataset, label):
            normal_tail_sum += tail
            normal_bags += 1
        else:
            abnormal_tail_sum += tail
            abnormal_bags += 1
    discriminative_score = (abnormal_tail_sum / abnormal_bags - normal_tail_sum / normal_bags).astype(np.float32)
    discriminative_rank = rank_within_layer(discriminative_score)
    semantic_rank = rank_within_layer(semantic_score)
    combined_score = (1.0 - args.semantic_weight) * discriminative_rank + args.semantic_weight * semantic_rank
    chosen_dims = np.argsort(-combined_score, axis=1, kind="stable")[:, :args.candidate_per_layer]
    selected_layers = np.repeat(np.arange(layers, dtype=np.int64), args.candidate_per_layer)
    selected_dimensions = chosen_dims.reshape(-1).astype(np.int64)
    selected_width = len(selected_layers)
    selected_affinity = semantic_response[selected_layers, selected_dimensions]
    selected_text_class = np.abs(selected_affinity).argmax(axis=-1).astype(np.int64) + 1
    selected_text_direction = np.sign(
        selected_affinity[np.arange(selected_width), selected_text_class - 1]
    ).astype(np.float32)
    selected_text_direction[selected_text_direction == 0] = 1.0

    # Pass 4: scene-conditioned normal state and transition statistics in the sparse circuit only.
    contexts = len(context_centers)
    selected_sum = np.zeros((contexts, selected_width), dtype=np.float64)
    selected_square_sum = np.zeros_like(selected_sum)
    selected_count = np.zeros(contexts, dtype=np.int64)
    transition_sum = np.zeros_like(selected_sum)
    transition_square_sum = np.zeros_like(selected_sum)
    transition_count = np.zeros(contexts, dtype=np.int64)
    for key, _label, path in tqdm(normal_rows, desc="build context normal banks", unit="video"):
        hidden = load_hidden(path)
        context = context_of_normal[key]
        chosen = hidden[:, selected_layers, selected_dimensions]
        selected_sum[context] += chosen.sum(axis=0, dtype=np.float64)
        selected_square_sum[context] += np.square(chosen, dtype=np.float64).sum(axis=0)
        selected_count[context] += len(chosen)
        if len(chosen) > 1:
            transition = chosen[1:] - chosen[:-1]
            transition_sum[context] += transition.sum(axis=0, dtype=np.float64)
            transition_square_sum[context] += np.square(transition, dtype=np.float64).sum(axis=0)
            transition_count[context] += len(transition)
    if np.any(selected_count == 0):
        raise RuntimeError("a normal context has no state observations")
    state_mean = selected_sum / selected_count[:, None]
    state_std = np.sqrt(np.maximum(selected_square_sum / selected_count[:, None] - np.square(state_mean), args.std_floor ** 2))
    transition_mean = np.divide(
        transition_sum, np.maximum(transition_count[:, None], 1), out=np.zeros_like(transition_sum), where=True
    )
    transition_std = np.sqrt(np.maximum(
        np.divide(transition_square_sum, np.maximum(transition_count[:, None], 1), out=np.zeros_like(transition_square_sum), where=True)
        - np.square(transition_mean),
        args.std_floor ** 2,
    ))

    artifact = {
        "version": 2,
        "dataset": args.dataset,
        "hidden_layers": layers,
        "hidden_width": width,
        "normal_label": normal,
        "prompts": prompts,
        "clip_model": args.clip_model,
        "selected_layers": torch.from_numpy(selected_layers),
        "selected_dimensions": torch.from_numpy(selected_dimensions),
        "selected_text_direction": torch.from_numpy(selected_text_direction),
        "selected_text_class": torch.from_numpy(selected_text_class),
        "selected_text_affinity": torch.from_numpy(selected_affinity),
        "context_centers": torch.from_numpy(context_centers),
        "state_mean": torch.from_numpy(state_mean.astype(np.float32)),
        "state_std": torch.from_numpy(state_std.astype(np.float32)),
        "transition_mean": torch.from_numpy(transition_mean.astype(np.float32)),
        "transition_std": torch.from_numpy(transition_std.astype(np.float32)),
        "ln_post_weight": torch.from_numpy(ln_weight),
        "ln_post_bias": torch.from_numpy(ln_bias),
        "ln_post_eps": ln_eps,
        "visual_projection": torch.from_numpy(projection),
        "text_features": torch.from_numpy(text_features),
    }
    atomic_torch_save(asset_path, artifact)

    table_rows = []
    chosen_set = {(int(layer), int(dim)) for layer, dim in zip(selected_layers, selected_dimensions)}
    for layer in range(layers):
        for dim in range(width):
            class_index = int(np.abs(semantic_response[layer, dim]).argmax()) + 1
            direction = float(np.sign(semantic_response[layer, dim, class_index - 1]) or 1.0)
            table_rows.append([
                layer + 1, dim, float(discriminative_score[layer, dim]), float(semantic_score[layer, dim]),
                float(combined_score[layer, dim]), prompts[class_index], direction, int((layer, dim) in chosen_set),
            ])
    write_csv(
        output / "channel_scores.csv",
        [
            "layer_1based", "dimension", "bag_discriminative", "semantic_lens", "combined",
            "dominant_anomaly_text", "signed_text_direction", "selected",
        ],
        table_rows,
    )
    write_csv(output / "missing_hidden.csv", ["video_key", "label"], missing)
    if missing:
        print(
            f"warning: skipped {len(missing)} source videos without cached hidden states; "
            f"see {output / 'missing_hidden.csv'}",
            flush=True,
        )
    save_json(output / "summary.json", {
        "dataset": args.dataset,
        "source_train_csv": args.source_train_csv,
        "hidden_manifest": args.hidden_manifest,
        "normal_label": normal,
        "normal_videos": len(normal_rows),
        "abnormal_videos": len(abnormal_rows),
        "missing_videos": len(missing),
        "hidden_shape": [layers, width],
        "selected_width": selected_width,
        "candidate_per_layer": args.candidate_per_layer,
        "context_count": contexts,
        "selection": "weak bag tail contrast plus signed frozen hidden-to-text semantic directions; no VadCLIP score is used",
        "assets": relpath(asset_path, output),
    })
    print(
        f"wrote {asset_path}: {selected_width} sparse hidden dimensions across {layers} layers; "
        f"normal contexts={contexts}; missing={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

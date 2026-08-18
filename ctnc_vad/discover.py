"""Discover a sparse, text-grounded CLIP normality circuit from cached hidden states.

This stage never reads a VadCLIP anomaly score. It uses pure-normal videos,
the frozen CLIP text encoder and supplied ``[T,L,D]`` CLS hidden artifacts to
fix a reusable normal channel gallery.
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
    video_label_vector,
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
    parser.add_argument(
        "--candidate-per-layer", type=int, default=64,
        help="Hard-selected text-grounded hidden witnesses retained per CLIP layer.",
    )
    parser.add_argument("--context-count", type=int, default=16)
    parser.add_argument("--context-iters", type=int, default=50)
    parser.add_argument(
        "--normal-prototype-count", type=int, default=64,
        help="Real normal hidden states retained per scene context for nearest-normal counterfactual retrieval.",
    )
    parser.add_argument(
        "--prototype-frames-per-video", type=int, default=32,
        help="Uniform normal frames contributed by each video to the context prototype bank.",
    )
    parser.add_argument(
        "--global-subspace-rank", type=int, default=16,
        help="Rank of the all-channel normal truncated PCA used to select original hidden witnesses.",
    )
    parser.add_argument(
        "--global-subspace-frames", type=int, default=4,
        help="Uniform normal frames per video for randomized all-channel SVD discovery.",
    )
    parser.add_argument("--frames-per-video", type=int, default=128, help="Maximum cached frames per video used for semantic-lens fitting.")
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
    if min(
        args.candidate_per_layer, args.context_count, args.frames_per_video, args.context_iters,
        args.normal_prototype_count, args.prototype_frames_per_video,
        args.global_subspace_rank, args.global_subspace_frames,
    ) <= 0:
        parser.error("candidate/context/prototype/frame counts and context iterations must be positive")
    if args.std_floor <= 0 or args.ridge < 0:
        parser.error("--std-floor must be positive and --ridge must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "discovery", clean=args.clean)
    asset_path = output / "circuit_assets.pt"
    if asset_path.is_file() and not args.no_resume:
        old = torch.load(asset_path, map_location="cpu", weights_only=False)
        if isinstance(old, dict) and int(old.get("version", -1)) == 8:
            print(f"reuse {asset_path}; pass --no-resume or --clean to rebuild discovery", flush=True)
            return
        print(f"rebuild {asset_path}: old discovery version is incompatible with channel witnesses", flush=True)

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
    if args.global_subspace_rank >= width:
        raise ValueError("--global-subspace-rank must be smaller than the CLIP hidden width")

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

    # A compact randomized SVD finds the shared *all-channel* normal
    # subspace.  It is deliberately a discovery-only operation: its role is
    # to select dimensions whose weak abnormal evidence survives removal of
    # ordinary correlated changes (camera, pose, background), not to replace
    # original dimensions with opaque PCA components at inference.
    global_sketch_width = min(width, args.global_subspace_rank + 8)
    global_rng = np.random.default_rng(args.seed)
    random_probe = global_rng.standard_normal((width, global_sketch_width)).astype(np.float32)
    range_sketch = np.zeros((layers, width, global_sketch_width), dtype=np.float64)
    for _key, _label, path in tqdm(normal_rows, desc="fit normal SVD range", unit="video"):
        hidden = load_hidden(path)
        sampled = hidden[uniform_indices(len(hidden), args.global_subspace_frames)]
        standardized = ((sampled - global_mean[None]) / global_std[None]).astype(np.float32)
        for layer in range(layers):
            projected = standardized[:, layer] @ random_probe
            range_sketch[layer] += standardized[:, layer].T @ projected
    range_basis = np.empty((layers, width, global_sketch_width), dtype=np.float64)
    for layer in range(layers):
        range_basis[layer] = np.linalg.qr(range_sketch[layer], mode="reduced")[0]

    small_covariance = np.zeros((layers, global_sketch_width, global_sketch_width), dtype=np.float64)
    for _key, _label, path in tqdm(normal_rows, desc="fit normal SVD spectrum", unit="video"):
        hidden = load_hidden(path)
        sampled = hidden[uniform_indices(len(hidden), args.global_subspace_frames)]
        standardized = ((sampled - global_mean[None]) / global_std[None]).astype(np.float32)
        for layer in range(layers):
            projected = standardized[:, layer] @ range_basis[layer]
            small_covariance[layer] += projected.T @ projected
    global_normal_subspace_basis = np.empty((layers, width, args.global_subspace_rank), dtype=np.float32)
    # Coordinate energy is the diagonal of the normal truncated-PCA
    # reconstruction. It lets us select *original dimensions* that actively
    # participate in the normal manifold, rather than using PCA components as
    # the final representation.
    normal_pca_coordinate_energy = np.empty((layers, width), dtype=np.float32)
    for layer in range(layers):
        values, vectors = np.linalg.eigh(small_covariance[layer])
        principal_values = values[-args.global_subspace_rank:]
        principal_basis = range_basis[layer] @ vectors[:, -args.global_subspace_rank:]
        global_normal_subspace_basis[layer] = principal_basis.astype(np.float32)
        normal_pca_coordinate_energy[layer] = np.sum(
            np.square(principal_basis) * principal_values[None], axis=1
        ).astype(np.float32)

    context_centers, context_assignments = kmeans_unit_vectors(
        np.stack(signatures), args.context_count, args.context_iters, args.seed
    )
    # kmeans may reduce the requested count when the normal set has fewer
    # distinct samples.  Use the actual number throughout gallery creation.
    contexts = int(context_centers.shape[0])
    context_of_normal = {key: int(context) for key, context in zip(normal_keys, context_assignments)}

    prompts = list(labels_for_dataset(args.dataset).values())
    anomaly_text_count = len(prompts) - 1
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
    semantic_text_score = np.abs(semantic_response).astype(np.float32)
    semantic_score = semantic_text_score.max(axis=-1)

    # Pass 3: variance/PCA sensitive-neuron localization. Following LAKE's
    # central observation, normal channels with structured high variance are
    # the active coordinates of the normal manifold; anomalies disturb those
    # coordinates more reliably than dormant low-variance dimensions. The
    # truncated-PCA coordinate energy stabilizes this variance criterion under
    # correlated normal scene changes. No abnormal video label or baseline
    # prediction enters this selection.
    normal_variance = np.square(global_std).astype(np.float32)
    variance_rank = rank_within_layer(normal_variance)
    pca_energy_rank = rank_within_layer(normal_pca_coordinate_energy)
    semantic_rank = rank_within_layer(semantic_score)
    # Text grounding is a small tie-breaker, not a pseudo label: it prefers
    # active normal-manifold coordinates that also have a frozen CLIP route
    # to an anomaly concept, while structural normal sensitivity remains the
    # dominant reason a coordinate is selected.
    sensitivity_score = 0.45 * variance_rank + 0.45 * pca_energy_rank + 0.10 * semantic_rank
    chosen_dims = np.argsort(-sensitivity_score, axis=1, kind="stable")[:, :args.candidate_per_layer]
    selected_layers = np.repeat(np.arange(layers, dtype=np.int64), args.candidate_per_layer)
    selected_dimensions = chosen_dims.reshape(-1).astype(np.int64)
    selected_width = len(selected_layers)
    selected_affinity = semantic_response[selected_layers, selected_dimensions]
    selected_text_class = np.abs(selected_affinity).argmax(axis=-1).astype(np.int64) + 1
    selected_text_direction = np.sign(
        selected_affinity[np.arange(selected_width), selected_text_class - 1]
    ).astype(np.float32)
    selected_text_direction[selected_text_direction == 0] = 1.0
    selection_text_classes = selected_text_class.reshape(layers, args.candidate_per_layer) - 1
    class_bags = np.zeros(anomaly_text_count, dtype=np.int64)
    for _key, label, _path in abnormal_rows:
        class_bags += video_label_vector(args.dataset, label)[1:].astype(np.int64)

    # Pass 4: retain a compact gallery of *real* normal states for every
    # scene context. At test time the reader retrieves its nearest normal
    # vector by cosine similarity in the selected original-channel subspace.
    prototype_candidates: list[list[np.ndarray]] = [[] for _ in range(contexts)]
    for key, _label, path in tqdm(normal_rows, desc="build normal prototype banks", unit="video"):
        hidden = load_hidden(path)
        context = context_of_normal[key]
        chosen = hidden[:, selected_layers, selected_dimensions]
        index = uniform_indices(len(chosen), args.prototype_frames_per_video)
        # Keep the actual selected hidden vector for cosine gallery probing.
        # No PCA coordinate replaces it at inference.
        prototype_candidates[context].append(chosen[index].astype(np.float32))
    rng = np.random.default_rng(args.seed)
    normal_prototypes = np.empty((contexts, args.normal_prototype_count, selected_width), dtype=np.float32)
    for context, groups_for_context in enumerate(prototype_candidates):
        candidates = np.concatenate(groups_for_context, axis=0)
        choose = rng.choice(len(candidates), size=args.normal_prototype_count, replace=len(candidates) < args.normal_prototype_count)
        normal_prototypes[context] = candidates[choose]
    artifact = {
        "version": 8,
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
        "selected_by_text_class": torch.from_numpy(selection_text_classes.reshape(-1).astype(np.int64) + 1),
        "selected_text_affinity": torch.from_numpy(selected_affinity),
        "context_centers": torch.from_numpy(context_centers),
        "normal_prototypes": torch.from_numpy(normal_prototypes),
        "global_normal_subspace_basis": torch.from_numpy(global_normal_subspace_basis),
        "global_subspace_rank": args.global_subspace_rank,
        "normal_variance": torch.from_numpy(normal_variance),
        "normal_pca_coordinate_energy": torch.from_numpy(normal_pca_coordinate_energy),
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
                layer + 1, dim, float(normal_variance[layer, dim]), float(normal_pca_coordinate_energy[layer, dim]),
                float(sensitivity_score[layer, dim]), float(semantic_score[layer, dim]),
                prompts[class_index], direction, int((layer, dim) in chosen_set),
            ])
    write_csv(
        output / "channel_scores.csv",
        [
            "layer_1based", "dimension", "normal_variance", "normal_pca_coordinate_energy", "sensitivity",
            "semantic_lens",
            "dominant_anomaly_text", "signed_text_direction", "selected",
        ],
        table_rows,
    )
    class_table_rows = []
    for layer in range(layers):
        for dimension in range(width):
            for text_class in range(anomaly_text_count):
                class_table_rows.append([
                    layer + 1,
                    dimension,
                    prompts[text_class + 1],
                    float(normal_variance[layer, dimension]),
                    float(normal_pca_coordinate_energy[layer, dimension]),
                    float(semantic_response[layer, dimension, text_class]),
                    float(sensitivity_score[layer, dimension]),
                ])
    write_csv(
        output / "channel_text_scores.csv",
        ["layer_1based", "dimension", "anomaly_text", "normal_variance", "normal_pca_coordinate_energy", "signed_semantic_lens", "sensitivity"],
        class_table_rows,
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
        "class_training_videos": {prompts[index + 1]: int(count) for index, count in enumerate(class_bags)},
        "candidate_per_layer": args.candidate_per_layer,
        "context_count": contexts,
        "normal_prototype_count": args.normal_prototype_count,
        "prototype_frames_per_video": args.prototype_frames_per_video,
        "global_subspace_rank": args.global_subspace_rank,
        "global_subspace_frames": args.global_subspace_frames,
        "selection": "normal variance plus truncated-PCA coordinate energy, with frozen hidden-to-text affinity as a tie-breaker; no anomaly label or VadCLIP score is used",
        "assets": relpath(asset_path, output),
    })
    print(
        f"wrote {asset_path}: {selected_width} sparse hidden dimensions across {layers} layers; "
        f"normal contexts={contexts}; missing={len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

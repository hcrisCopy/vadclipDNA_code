"""Build resumable frozen CLIP text-lens assets from reusable normal hidden data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from xd_dna.common import is_pure_normal
from xd_dna.vadclip import add_local_vadclip_source

from .common import (
    atomic_torch_save,
    base_key,
    default_output_root,
    labels_for_dataset,
    load_clip_feature,
    load_hidden,
    manifest_hidden_paths,
    read_path_label_csv,
    relpath,
    resolve_recorded_path,
    save_json,
    set_seed,
    stage_dir,
)


class RunningMoments:
    """Numerically stable vector mean and sample variance without retaining all frames."""

    def __init__(self, width: int) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("running moments require non-empty [N,D] values")
        values64 = np.asarray(values, dtype=np.float64)
        batch_count = len(values64)
        batch_mean = values64.mean(axis=0)
        batch_m2 = np.square(values64 - batch_mean).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total

    def standard_deviation(self) -> np.ndarray:
        if self.count < 2:
            raise RuntimeError("at least two normal frames are required for normal statistics")
        return np.sqrt(np.maximum(self.m2 / (self.count - 1), 1e-12)).astype(np.float32)


def resolved_layer_index(configured: int, layer_count: int) -> int:
    result = layer_count + configured if configured < 0 else configured
    if result < 0 or result >= layer_count:
        raise ValueError(f"last-layer-index={configured} is invalid for {layer_count} extracted layers")
    return result


def post_layer_norm(
    hidden: np.ndarray,
    layer_index: int,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    device: torch.device,
) -> np.ndarray:
    """Apply the native CLIP LN-post to raw final CLS states."""
    raw = torch.from_numpy(np.asarray(hidden[:, layer_index], dtype=np.float32)).to(device)
    with torch.no_grad():
        result = F.layer_norm(raw, (raw.shape[-1],), ln_weight, ln_bias, float(ln_eps))
    return result.cpu().numpy().astype(np.float32, copy=False)


def update_reservoir(
    reservoir: np.ndarray,
    filled: int,
    seen: int,
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Keep a deterministic uniform sample of normal LN-post vectors."""
    for value in values:
        seen += 1
        if filled < len(reservoir):
            reservoir[filled] = value
            filled += 1
        else:
            target = int(rng.integers(seen))
            if target < len(reservoir):
                reservoir[target] = value
    return filled, seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen CLIP text directions and a compact normality bank for NeuVAD-Lens."
    )
    parser.add_argument("--dataset", choices=["xd", "ucf"], default="xd")
    parser.add_argument("--source-train-csv", required=True, help="Original reusable 512D train path,label CSV.")
    parser.add_argument("--source-path-base", default=".", help="Base for relative original feature paths; use '.' from vadclipDNA_code.")
    parser.add_argument("--hidden-manifest", required=True, help="Reusable [T,12,768] CLS hidden manifest.")
    parser.add_argument("--hidden-path-base", default=".", help="Base for relative hidden_path values; use '.' from vadclipDNA_code.")
    parser.add_argument(
        "--allow-missing-hidden",
        action="store_true",
        help="Skip pure-normal train videos absent from the hidden manifest and record them in summary.json.",
    )
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--last-layer-index", type=int, default=-1, help="-1 means the final extracted Transformer layer.")
    parser.add_argument("--normal-subspace-dim", type=int, default=64)
    parser.add_argument("--prototype-count", type=int, default=16)
    parser.add_argument("--verify-videos", type=int, default=16, help="Normal videos used to verify raw final hidden and 512D CLIP alignment.")
    parser.add_argument("--min-projection-cosine", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only the lens stage.")
    parser.add_argument("--no-resume", action="store_true", help="Rebuild lens_assets.pt even when it is valid.")
    args = parser.parse_args()
    if args.normal_subspace_dim <= 0 or args.normal_subspace_dim > 768:
        parser.error("--normal-subspace-dim must be in [1,768]")
    if args.prototype_count <= 0 or args.verify_videos <= 0:
        parser.error("--prototype-count and --verify-videos must be positive")
    if not 0.0 < args.min_projection_cosine <= 1.0:
        parser.error("--min-projection-cosine must be in (0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "lens", clean=args.clean)
    target = output / "lens_assets.pt"
    if target.is_file() and not args.no_resume:
        print(f"reuse existing lens assets: {target}", flush=True)
        return
    set_seed(args.seed)
    device = torch.device(args.device)
    add_local_vadclip_source()
    from clip import clip

    clip_model, _preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval().requires_grad_(False)
    visual = clip_model.visual
    if not hasattr(visual, "ln_post") or not hasattr(visual, "proj"):
        raise ValueError(f"{args.clip_model} is not a CLIP ViT with ln_post and projection")
    ln_weight = visual.ln_post.weight.detach().float().to(device)
    ln_bias = visual.ln_post.bias.detach().float().to(device)
    visual_projection = visual.proj.detach().float().to(device)
    if visual_projection.shape != (768, 512):
        raise ValueError(f"expected ViT-B/16 projection [768,512], got {tuple(visual_projection.shape)}")

    labels = labels_for_dataset(args.dataset, test=False)
    class_names = list(labels.values())
    if len(class_names) < 2 or class_names[0] != "normal":
        raise ValueError(f"{args.dataset}: expected the first fixed text class to be normal, got {class_names}")
    with torch.no_grad():
        # VadCLIP's local CLIP fork exposes the prompt-tuning interface:
        # encode_text(token_embeddings, token_ids), rather than OpenAI CLIP's
        # encode_text(token_ids).  Keep the unmodified fixed-text path used by
        # the baseline by obtaining its token embeddings first.
        text_tokens = clip.tokenize(class_names).to(device)
        text_embeddings = clip_model.encode_token(text_tokens)
        text_features = clip_model.encode_text(text_embeddings, text_tokens).float()
        text_features = F.normalize(text_features, dim=-1, eps=1e-6)
    normal_text = text_features[0]
    abnormal_text = text_features[1:]
    text_directions = (visual_projection @ (abnormal_text - normal_text).t()).t()

    source = read_path_label_csv(args.source_train_csv)
    hidden_by_key, token_pool = manifest_hidden_paths(
        args.hidden_manifest, args.hidden_prefix_from, args.hidden_prefix_to, args.hidden_path_base,
    )
    normal_keys: list[str] = []
    seen_keys: set[str] = set()
    missing_normal_keys: list[str] = []
    normal_source_path: dict[str, str] = {}
    for row in source.itertuples(index=False):
        key = base_key(str(row.path))
        if is_pure_normal(args.dataset, str(row.label)) and key not in seen_keys:
            seen_keys.add(key)
            if key not in hidden_by_key:
                if not args.allow_missing_hidden:
                    raise FileNotFoundError(
                        f"{key}: pure-normal train video is absent from hidden manifest; "
                        "pass --allow-missing-hidden only when this is an expected extraction omission"
                    )
                missing_normal_keys.append(key)
                continue
            normal_keys.append(key)
            normal_source_path[key] = str(row.path)
    if missing_normal_keys:
        print(
            f"warning: skipped {len(missing_normal_keys)} pure-normal train videos absent from hidden manifest; "
            "their keys will be recorded in summary.json",
            flush=True,
        )
    if len(normal_keys) < 2:
        raise RuntimeError(f"need at least two readable pure-normal videos, found {len(normal_keys)}")

    moments = RunningMoments(768)
    layer_index: int | None = None
    for key in tqdm(sorted(normal_keys), desc="collect normal LN-post statistics", unit="video"):
        hidden, _metadata = load_hidden(hidden_by_key[key])
        if hidden.shape[2] != 768:
            raise ValueError(f"{key}: expected hidden width 768, got {hidden.shape}")
        candidate_layer = resolved_layer_index(args.last_layer_index, hidden.shape[1])
        if layer_index is None:
            layer_index = candidate_layer
        elif layer_index != candidate_layer:
            raise ValueError("hidden manifest contains inconsistent layer counts")
        moments.update(post_layer_norm(hidden, layer_index, ln_weight, ln_bias, visual.ln_post.eps, device))
    if layer_index is None:
        raise RuntimeError("normal statistics did not process any hidden features")
    normal_mean = moments.mean.astype(np.float32)
    normal_std = moments.standard_deviation()
    normal_indices = np.argsort(-np.square(normal_std), kind="mergesort")[:args.normal_subspace_dim].astype(np.int64)

    reservoir = np.empty((args.prototype_count, 768), dtype=np.float32)
    filled = seen = 0
    rng = np.random.default_rng(args.seed)
    for key in tqdm(sorted(normal_keys), desc="sample normal prototypes", unit="video"):
        hidden, _metadata = load_hidden(hidden_by_key[key])
        post = post_layer_norm(hidden, layer_index, ln_weight, ln_bias, visual.ln_post.eps, device)
        filled, seen = update_reservoir(reservoir, filled, seen, post, rng)
    if filled == 0:
        raise RuntimeError("normal prototype reservoir is empty")
    prototypes = reservoir[:filled, normal_indices]
    prototypes = (prototypes - normal_mean[normal_indices]) / np.maximum(normal_std[normal_indices], 1e-6)

    # The Lens assumes the reusable final CLS state is immediately before the
    # frozen CLIP LN-post/projection. Verify this contract before any expensive
    # training begins; cosine comparison is invariant to whether the saved 512D
    # official feature was already L2-normalized.
    source_path_base = Path(args.source_path_base).resolve()
    projection_cosines: list[float] = []
    for key in tqdm(sorted(normal_keys)[:args.verify_videos], desc="verify CLIP projection contract", unit="video"):
        hidden, _metadata = load_hidden(hidden_by_key[key])
        post = post_layer_norm(hidden, layer_index, ln_weight, ln_bias, visual.ln_post.eps, device)
        with torch.no_grad():
            reconstructed = F.normalize(torch.from_numpy(post).to(device) @ visual_projection, dim=-1, eps=1e-6)
        official = load_clip_feature(resolve_recorded_path(normal_source_path[key], source_path_base))
        length = min(len(reconstructed), len(official))
        if length <= 0:
            raise RuntimeError(f"{key}: empty feature alignment during CLIP projection validation")
        official_tensor = F.normalize(torch.from_numpy(official[:length]).to(device), dim=-1, eps=1e-6)
        projection_cosines.append(float((reconstructed[:length] * official_tensor).sum(dim=-1).mean().cpu().item()))
    if not projection_cosines or min(projection_cosines) < args.min_projection_cosine:
        raise RuntimeError(
            "raw final CLS cannot reproduce the supplied 512D CLIP feature under native LN-post/projection; "
            f"cosines={projection_cosines}, required minimum={args.min_projection_cosine}. "
            "Check hidden extraction layer/order before training."
        )

    asset = {
        "schema_version": 1,
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "token_pool": token_pool,
        "last_layer_index": layer_index,
        "normal_class_name": class_names[0],
        "abnormal_class_names": class_names[1:],
        "ln_weight": ln_weight.cpu(),
        "ln_bias": ln_bias.cpu(),
        "ln_eps": float(visual.ln_post.eps),
        "visual_projection": visual_projection.cpu(),
        "normal_text": normal_text.cpu(),
        "abnormal_text": abnormal_text.cpu(),
        "text_directions": text_directions.cpu(),
        "normal_mean": torch.from_numpy(normal_mean),
        "normal_std": torch.from_numpy(normal_std),
        "normal_indices": torch.from_numpy(normal_indices),
        "normal_prototypes": torch.from_numpy(prototypes.astype(np.float32)),
    }
    atomic_torch_save(target, asset)
    save_json(output / "summary.json", {
        "schema_version": 1,
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "token_pool": token_pool,
        "last_layer_index": layer_index,
        "normal_class_name": class_names[0],
        "abnormal_class_names": class_names[1:],
        "normal_videos": len(normal_keys),
        "missing_normal_hidden_videos": missing_normal_keys,
        "normal_frames": moments.count,
        "normal_subspace_dim": int(args.normal_subspace_dim),
        "prototype_count": int(filled),
        "projection_cosine_mean": float(np.mean(projection_cosines)),
        "projection_cosine_min": float(np.min(projection_cosines)),
        "source_train_csv": relpath(args.source_train_csv, output),
        "hidden_manifest": relpath(args.hidden_manifest, output),
        "lens_assets": target.name,
    })
    print(
        f"wrote {target}: {len(class_names) - 1} abnormal text directions, "
        f"all 768 text dimensions, normal prototypes={filled}x{args.normal_subspace_dim}",
        flush=True,
    )


if __name__ == "__main__":
    main()

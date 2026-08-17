"""Export resumable text-lens evidence curves; top-k is presentation-only."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .common import (
    atomic_save_npz,
    default_output_root,
    load_clip_feature,
    load_json,
    read_path_label_csv,
    resolve_recorded_path,
    save_json,
    stage_dir,
)
from .lens import TextProjectionLens


def cached_audit(path: Path, length: int, class_count: int, topk: int) -> bool:
    """Reuse only a complete evidence artifact with the expected contract."""
    if not path.is_file():
        return False
    try:
        archive = np.load(path, allow_pickle=False)
        try:
            route = archive["class_route"]
            distance = archive["normal_distance"]
            dimensions = archive["top_dimension"]
            values = archive["top_contribution"]
        finally:
            archive.close()
    except Exception:
        return False
    return (
        route.shape == (length, class_count)
        and distance.shape == (length,)
        and dimensions.shape == (length, class_count, topk)
        and values.shape == (length, class_count, topk)
        and np.isfinite(route).all()
        and np.isfinite(distance).all()
        and np.isfinite(values).all()
    )


def sorted_top_positive(contributions: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Return indices/values of strongest positive evidence without changing inference."""
    if contributions.ndim != 3:
        raise ValueError(f"expected [T,C,768] contributions, got {contributions.shape}")
    width = contributions.shape[-1]
    if topk <= 0 or topk > width:
        raise ValueError(f"topk must be in [1,{width}]")
    order = np.argsort(-contributions, axis=-1, kind="mergesort")[..., :topk]
    values = np.take_along_axis(contributions, order, axis=-1)
    return order.astype(np.int16), values.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NeuVAD-Lens class routing and per-frame top text contributions without affecting inference."
    )
    parser.add_argument("--feature-list", required=True, help="Lens train or test CSV made by neuvad_lens.build_features.")
    parser.add_argument("--neuron-json", required=True)
    parser.add_argument("--lens-assets", required=True)
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--split-name", choices=["train", "test"], required=True)
    parser.add_argument("--topk", type=int, default=8, help="Display-only top evidence dimensions per frame/category.")
    parser.add_argument("--text-temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only the lens_audit stage.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video evidence files.")
    args = parser.parse_args()
    if args.topk <= 0 or args.topk > 768 or args.text_temperature <= 0:
        parser.error("--topk must be in [1,768] and text temperature must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "lens_audit", clean=args.clean)
    split_dir = (output / args.split_name).resolve()
    if args.no_resume and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    neuron_width = int(load_json(Path(args.neuron_json).resolve())["neuron_width"])
    expected_width = neuron_width + 512 + 768
    device = torch.device(args.device)
    lens = TextProjectionLens(args.lens_assets).to(device).eval()
    frame = read_path_label_csv(args.feature_list)
    list_path = Path(args.feature_list).resolve()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"audit Lens {args.split_name}", unit="video"):
        feature_path = resolve_recorded_path(str(row.path), list_path.parent)
        feature = load_clip_feature(feature_path)
        if feature.shape[1] != expected_width:
            raise ValueError(f"{feature_path}: expected {expected_width}D Lens feature, got {feature.shape}")
        target = split_dir / f"{feature_path.stem}.npz"
        if not args.no_resume and cached_audit(target, len(feature), lens.abnormal_class_count, args.topk):
            continue
        last_hidden = torch.from_numpy(feature[:, neuron_width + 512:]).unsqueeze(0).to(device)
        with torch.no_grad():
            _evidence, route, distance, contributions = lens(last_hidden, temperature=args.text_temperature)
        dimensions, values = sorted_top_positive(contributions.squeeze(0).cpu().numpy(), args.topk)
        atomic_save_npz(
            target,
            class_route=route.squeeze(0).cpu().numpy().astype(np.float32),
            normal_distance=distance.squeeze(0).cpu().numpy().astype(np.float32),
            top_dimension=dimensions,
            top_contribution=values,
        )
    save_json(output / f"{args.split_name}_summary.json", {
        "feature_list": args.feature_list,
        "lens_assets": args.lens_assets,
        "topk": args.topk,
        "text_temperature": args.text_temperature,
        "abnormal_class_names": lens.class_names,
        "normal_class_name": lens.normal_class_name,
        "meaning": "top_dimension/top_contribution are display-only; all 768 dimensions remain in model inference.",
        "output_directory": args.split_name,
    })
    print(f"wrote resumable {args.split_name} evidence curves under {split_dir}", flush=True)


if __name__ == "__main__":
    main()

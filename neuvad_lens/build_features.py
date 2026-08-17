"""Build resumable [DNA neurons | CLIP | raw final CLS] NeuVAD-Lens features."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from xd_dna.build_features import selected_contract

from .common import (
    atomic_save_npy,
    base_key,
    default_output_root,
    load_clip_feature,
    load_hidden,
    manifest_hidden_paths,
    read_path_label_csv,
    relpath,
    resolve_recorded_path,
    save_json,
    stage_dir,
    write_csv,
)
from .lens import load_lens_asset


def align_hidden(hidden: np.ndarray, clip_length: int, alignment: str) -> tuple[np.ndarray, str]:
    """Use the exact temporal alignment options of xd_dna.build_features."""
    if hidden.shape[0] == clip_length:
        return hidden, "exact"
    if hidden.shape[0] > clip_length and alignment == "crop_hidden":
        return hidden[:clip_length], "crop_hidden"
    if hidden.shape[0] < clip_length and alignment == "pad_hidden":
        padding = np.repeat(hidden[-1:], clip_length - len(hidden), axis=0)
        return np.concatenate([hidden, padding], axis=0), "pad_hidden"
    raise ValueError(
        f"temporal length mismatch: hidden={len(hidden)} clip={clip_length}; "
        "use --alignment crop_hidden or pad_hidden only when deliberately justified"
    )


def fuse_feature(
    hidden: np.ndarray,
    clip: np.ndarray,
    normal_mean: np.ndarray,
    normal_std: np.ndarray,
    selected: list[tuple[int, np.ndarray]],
    last_layer_index: int,
    alignment: str,
) -> tuple[np.ndarray, str]:
    """Preserve DNA feature order and append the raw final 768D CLS state."""
    if hidden.ndim != 3 or hidden.shape[1:] != normal_mean.shape:
        raise ValueError(f"hidden [T,L,D]={hidden.shape} differs from DNA normal statistics {normal_mean.shape}")
    hidden, action = align_hidden(hidden, len(clip), alignment)
    z_hidden = (hidden - normal_mean) / np.maximum(normal_std, 1e-6)
    neurons = np.concatenate([z_hidden[:, layer, dimensions] for layer, dimensions in selected], axis=1)
    last_hidden = hidden[:, last_layer_index]
    result = np.concatenate([neurons, clip, last_hidden], axis=1).astype(np.float32)
    return result, action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resumable NeuVAD-Lens features without modifying the VadCLIP feature protocol."
    )
    parser.add_argument("--dataset", choices=["xd", "ucf"], default="xd")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--source-csv", required=True, help="Original reusable 512D VadCLIP path,label CSV.")
    parser.add_argument("--source-path-base", default=".", help="Base for relative original feature paths; use '.' from vadclipDNA_code.")
    parser.add_argument("--hidden-manifest", required=True, help="Reusable [T,12,768] CLS hidden manifest.")
    parser.add_argument("--hidden-path-base", default=".", help="Base for relative hidden paths; use '.' from vadclipDNA_code.")
    parser.add_argument("--neuron-json", required=True, help="Reusable xd_dna localization/selected_neurons.json.")
    parser.add_argument("--lens-assets", required=True, help="lens/lens_assets.pt made by neuvad_lens.build_lens_assets.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--allow-missing-hidden", action="store_true", help="Skip missing train hidden data only.")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only this split's Lens features.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video feature artifacts.")
    args = parser.parse_args()

    features_root = stage_dir(args.output_root, "features")
    split_dir = (features_root / args.split).resolve()
    if args.clean and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    lists_dir = stage_dir(args.output_root, "lists")
    output_csv = lists_dir / f"{args.dataset}_neuvad_lens_{args.split}.csv"
    if args.clean and output_csv.exists():
        output_csv.unlink()

    neuron_json = Path(args.neuron_json).resolve()
    contract, normal_mean, normal_std, selected = selected_contract(neuron_json)
    lens_path = Path(args.lens_assets).resolve()
    lens = load_lens_asset(lens_path)
    last_layer_index = int(lens["last_layer_index"])
    neuron_width = int(contract["neuron_width"])
    input_width = neuron_width + 512 + 768
    hidden_by_key, token_pool = manifest_hidden_paths(
        args.hidden_manifest, args.hidden_prefix_from, args.hidden_prefix_to, args.hidden_path_base,
    )
    source_csv = Path(args.source_csv).resolve()
    source_path_base = Path(args.source_path_base).resolve()
    source = read_path_label_csv(source_csv)

    output_rows: list[list[str]] = []
    alignment_rows: list[list[object]] = []
    skipped_rows: list[list[str]] = []
    written: set[Path] = set()
    for row in tqdm(source.itertuples(index=False), total=len(source), desc=f"build {args.split} Lens features", unit="feature"):
        source_path, label = str(row.path), str(row.label)
        key, stem = base_key(source_path), Path(source_path).stem
        target = split_dir / f"{stem}.npy"
        if target in written:
            raise ValueError(f"duplicate output feature filename: {target.name}")
        written.add(target)
        hidden_path = hidden_by_key.get(key)
        if hidden_path is None:
            if not args.allow_missing_hidden:
                raise FileNotFoundError(f"{key}: no hidden artifact in {args.hidden_manifest}")
            skipped_rows.append([source_path, label, key, "missing_hidden"])
            continue
        clip = load_clip_feature(resolve_recorded_path(source_path, source_path_base))
        reused = False
        if target.is_file() and not args.no_resume:
            candidate = load_clip_feature(target)
            if candidate.shape == (len(clip), input_width):
                fused, action, reused = candidate, "reused", True
            else:
                fused, action = None, "invalid_existing"
        else:
            fused, action = None, "new"
        if not reused:
            hidden, _metadata = load_hidden(hidden_path)
            if last_layer_index < 0 or last_layer_index >= hidden.shape[1]:
                raise ValueError(f"{key}: lens last layer {last_layer_index} is invalid for hidden {hidden.shape}")
            fused, action = fuse_feature(
                hidden, clip, normal_mean, normal_std, selected, last_layer_index, args.alignment
            )
            if fused.shape != (len(clip), input_width):
                raise RuntimeError(f"{key}: got fused shape {fused.shape}, expected {(len(clip), input_width)}")
            atomic_save_npy(target, fused)
        output_rows.append([relpath(target, output_csv.parent), label])
        alignment_rows.append([source_path, key, len(clip), neuron_width, 768, input_width, action])

    if args.split == "test" and skipped_rows:
        raise RuntimeError("test rows cannot be skipped because official ground-truth alignment would be invalid")
    write_csv(output_csv, ["path", "label"], output_rows)
    write_csv(
        split_dir / "alignment.csv",
        ["source_path", "video_key", "clip_length", "dna_width", "last_hidden_width", "fused_width", "alignment"],
        alignment_rows,
    )
    write_csv(split_dir / "skipped_rows.csv", ["source_path", "label", "video_key", "reason"], skipped_rows)
    save_json(split_dir / "summary.json", {
        "dataset": args.dataset,
        "split": args.split,
        "source_csv": args.source_csv,
        "hidden_manifest": args.hidden_manifest,
        "neuron_json": relpath(neuron_json, split_dir),
        "lens_assets": relpath(lens_path, split_dir),
        "token_pool": token_pool,
        "feature_order": "zscored_dna_neurons_then_official_512d_clip_then_raw_final_cls_hidden",
        "dna_width": neuron_width,
        "last_hidden_width": 768,
        "input_width": input_width,
        "alignment": args.alignment,
        "allow_missing_hidden": args.allow_missing_hidden,
        "rows_written": len(output_rows),
        "rows_skipped": len(skipped_rows),
        "list_csv": relpath(output_csv, split_dir),
    })
    print(
        f"wrote {output_csv}: {len(output_rows)} rows of [T,{input_width}] "
        f"[DNA|CLIP|last-hidden] features; skipped={len(skipped_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Build VadCLIP-ready [selected DNA neurons | 512D CLIP] XD feature lists."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    atomic_save_npy,
    base_key,
    default_output_root,
    load_clip_feature,
    load_hidden,
    load_json,
    manifest_hidden_paths,
    read_path_label_csv,
    relpath,
    resolve_recorded_path,
    save_json,
    stage_dir,
    write_csv,
)


def selected_contract(path: Path) -> tuple[dict, np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    config = load_json(path)
    required = {"selected", "normal_mean_path", "normal_std_path", "neuron_width", "clip_dim", "input_width"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path}: selected-neuron JSON is missing {sorted(missing)}")
    mean = np.load(resolve_recorded_path(config["normal_mean_path"], path.parent), allow_pickle=False).astype(np.float32)
    std = np.load(resolve_recorded_path(config["normal_std_path"], path.parent), allow_pickle=False).astype(np.float32)
    selected = [(int(item["layer_index"]), np.asarray(item["dims"], dtype=np.int64)) for item in config["selected"]]
    width = sum(len(dims) for _layer, dims in selected)
    if mean.ndim != 2 or std.shape != mean.shape:
        raise ValueError(f"{path}: invalid normal statistic shapes {mean.shape} and {std.shape}")
    if width != int(config["neuron_width"]) or int(config["clip_dim"]) != 512 or int(config["input_width"]) != width + 512:
        raise ValueError(f"{path}: invalid selected-neuron input contract")
    if any(layer < 0 or layer >= mean.shape[0] or len(dims) == 0 or dims.min() < 0 or dims.max() >= mean.shape[1] for layer, dims in selected):
        raise ValueError(f"{path}: selected layer/dimension index is outside normal-statistics shape")
    return config, mean, std, selected


def fuse_feature(hidden: np.ndarray, clip: np.ndarray, mean: np.ndarray, std: np.ndarray, selected: list[tuple[int, np.ndarray]], alignment: str) -> tuple[np.ndarray, str]:
    if hidden.shape[1:] != mean.shape:
        raise ValueError(f"hidden [L,D]={hidden.shape[1:]} differs from normal statistics {mean.shape}")
    if hidden.shape[0] == clip.shape[0]:
        action = "exact"
    elif hidden.shape[0] > clip.shape[0] and alignment == "crop_hidden":
        hidden, action = hidden[:len(clip)], "crop_hidden"
    elif hidden.shape[0] < clip.shape[0] and alignment == "pad_hidden":
        hidden = np.concatenate([hidden, np.repeat(hidden[-1:], len(clip) - len(hidden), axis=0)], axis=0)
        action = "pad_hidden"
    else:
        raise ValueError(
            f"temporal length mismatch: hidden={len(hidden)} clip={len(clip)}; "
            "use --alignment crop_hidden or pad_hidden only when deliberately justified"
        )
    z_hidden = (hidden - mean) / np.maximum(std, 1e-6)
    neurons = np.concatenate([z_hidden[:, layer, dims] for layer, dims in selected], axis=1).astype(np.float32)
    fused = np.concatenate([neurons, clip], axis=1).astype(np.float32)
    return fused, action


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable XD [DNA neuron|CLIP] features for the local VadCLIP wrapper.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--source-csv", required=True, help="Original/reused XD 512D path,label CSV for this split.")
    parser.add_argument(
        "--source-path-base", default=".",
        help="Base directory for relative paths in the original VadCLIP CSV; use '.' when running from vadclipDNA_code.",
    )
    parser.add_argument("--hidden-manifest", required=True, help="Reusable CLS hidden manifest for this split.")
    parser.add_argument(
        "--hidden-path-base", default=".",
        help="Base directory for relative hidden_path entries in the DSANet manifest; use '.' from vadclipDNA_code.",
    )
    parser.add_argument("--neuron-json", default="", help="Defaults to localization/selected_neurons.json under --output-root.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--allow-missing-hidden", action="store_true", help="Skip rows without hidden data; intended only for the known XD train omissions.")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only this split's fused features.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid single-row fused features.")
    args = parser.parse_args()

    root = stage_dir(args.output_root, "features")
    split_dir = (root / args.split).resolve()
    if args.clean and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    lists_dir = stage_dir(args.output_root, "lists")
    output_csv = lists_dir / f"xd_concat_{args.split}.csv"
    if args.clean and output_csv.exists():
        output_csv.unlink()
    neuron_json = Path(args.neuron_json).resolve() if args.neuron_json else (root.parent / "localization" / "selected_neurons.json").resolve()
    contract, mean, std, selected = selected_contract(neuron_json)
    hidden_by_key, token_pool = manifest_hidden_paths(
        args.hidden_manifest, args.hidden_prefix_from, args.hidden_prefix_to, args.hidden_path_base,
    )
    source_csv = Path(args.source_csv).resolve()
    source_path_base = Path(args.source_path_base).resolve()
    source = read_path_label_csv(source_csv)

    output_rows: list[list[str]] = []
    alignment_rows: list[list[object]] = []
    skipped: list[list[str]] = []
    seen_outputs: set[Path] = set()
    for row in tqdm(source.itertuples(index=False), total=len(source), desc=f"build {args.split} fused features", unit="feature"):
        source_path, label = str(row.path), str(row.label)
        key, stem = base_key(source_path), Path(source_path).stem
        target = split_dir / f"{stem}.npy"
        if target in seen_outputs:
            raise ValueError(f"duplicate output feature filename: {target.name}")
        seen_outputs.add(target)
        hidden_path = hidden_by_key.get(key)
        if hidden_path is None:
            if not args.allow_missing_hidden:
                raise FileNotFoundError(f"{key}: no hidden artifact in {args.hidden_manifest}")
            skipped.append([source_path, label, key, "missing_hidden"])
            continue
        # Match the unmodified VadCLIP dataset: original feature CSV paths are
        # interpreted from the launch directory, not from the CSV's directory.
        clip = load_clip_feature(resolve_recorded_path(source_path, source_path_base))
        reused = False
        if target.is_file() and not args.no_resume:
            candidate = load_clip_feature(target)
            if candidate.shape == (len(clip), int(contract["input_width"])):
                fused, action, reused = candidate, "reused", True
            else:
                fused, action = None, "invalid_existing"
        else:
            fused, action = None, "new"
        if not reused:
            hidden, _metadata = load_hidden(hidden_path)
            fused, action = fuse_feature(hidden, clip, mean, std, selected, args.alignment)
            expected = (len(clip), int(contract["input_width"]))
            if fused.shape != expected:
                raise RuntimeError(f"{key}: fused shape {fused.shape}, expected {expected}")
            atomic_save_npy(target, fused)
        output_rows.append([relpath(target, output_csv.parent), label])
        alignment_rows.append([source_path, key, len(clip), int(contract["neuron_width"]), int(contract["input_width"]), action])

    if args.split == "test" and skipped:
        raise RuntimeError("test rows cannot be skipped because XD ground-truth alignment would be invalid")
    write_csv(output_csv, ["path", "label"], output_rows)
    write_csv(split_dir / "alignment.csv", ["source_path", "video_key", "clip_length", "neuron_width", "fused_width", "alignment"], alignment_rows)
    write_csv(split_dir / "skipped_rows.csv", ["source_path", "label", "video_key", "reason"], skipped)
    save_json(split_dir / "summary.json", {
        "split": args.split, "source_csv": args.source_csv, "source_path_base": args.source_path_base,
        "hidden_manifest": args.hidden_manifest,
        "neuron_json": relpath(neuron_json, split_dir), "token_pool": token_pool,
        "alignment": args.alignment, "allow_missing_hidden": args.allow_missing_hidden,
        "rows_written": len(output_rows), "rows_skipped": len(skipped),
        "neuron_width": int(contract["neuron_width"]), "input_width": int(contract["input_width"]),
        "list_csv": relpath(output_csv, split_dir),
    })
    print(f"wrote {output_csv}: {len(output_rows)} rows of [T,{contract['input_width']}] features; skipped={len(skipped)}", flush=True)


if __name__ == "__main__":
    main()

"""Export per-video CTNC evidence without rerunning or modifying VadCLIP."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .assets import load_assets
from .circuit import NormalityCircuit
from .common import atomic_save_npz, default_output_root, hidden_manifest_paths, stage_dir, write_csv
from .dataset import HiddenBagDataset


def model_state(path: str | Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and "model_state_dict" in value:
        return value["model_state_dict"]
    if isinstance(value, dict):
        return value
    raise ValueError(f"{path}: expected CTNC state dictionary or checkpoint")


@torch.no_grad()
def evidence_for_video(model: NormalityCircuit, item: dict, device: torch.device, topk: int) -> dict[str, np.ndarray]:
    length = int(item["length"])
    outputs = model(
        item["circuit"].unsqueeze(0).to(device),
        item["last_hidden"].unsqueeze(0).to(device),
        torch.tensor([length], dtype=torch.int64, device=device),
    )
    contribution = outputs["dimension_state"] + outputs["dimension_transition"]
    count = min(int(topk), contribution.shape[-1])
    values, indices = contribution[0, :length].topk(count, dim=-1)
    return {
        "circuit_score": outputs["score"][0, :length].cpu().numpy().astype(np.float32),
        "state_score": outputs["state_score"][0, :length].cpu().numpy().astype(np.float32),
        "transition_score": outputs["transition_score"][0, :length].cpu().numpy().astype(np.float32),
        "text_margin": outputs["text_margin"][0, :length].cpu().numpy().astype(np.float32),
        "text_similarity": outputs["text_similarity"][0, :length].cpu().numpy().astype(np.float32),
        "normal_context": outputs["context"].cpu().numpy().astype(np.int64),
        "gate": outputs["gates"].cpu().numpy().astype(np.float32),
        "top_circuit_index": indices.cpu().numpy().astype(np.int64),
        "top_circuit_contribution": values.cpu().numpy().astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CTNC layer/dimension/text/normal-context evidence from cached hidden states.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only audit/<split-name>/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute valid per-video audit artifacts.")
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_root = args.output_root or str(default_output_root(args.dataset))
    root = stage_dir(output_root, "audit")
    output = root / args.split_name
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError("asset dataset does not match --dataset")
    hidden = hidden_manifest_paths(
        args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to
    )
    selected_layers = assets["selected_layers"].cpu().numpy()
    selected_dimensions = assets["selected_dimensions"].cpu().numpy()
    # ``visual_length`` is only used by training samples; audit retains every original test segment.
    dataset = HiddenBagDataset(
        args.dataset, args.source_test_csv, args.source_path_base, hidden, selected_layers, selected_dimensions,
        visual_length=256, training=False, alignment=args.alignment, allow_missing_hidden=False,
    )
    device = torch.device(args.device)
    model = NormalityCircuit(assets).to(device)
    model.load_state_dict(model_state(args.model_path), strict=True)
    rows: list[list[object]] = []
    for item in tqdm(dataset, desc=f"CTNC audit {args.split_name}", unit="video"):
        target = output / f"{item['key']}.npz"
        if target.is_file() and not args.no_resume:
            rows.append([item["key"], item["label"], item["length"], target.name, "reused"])
            continue
        evidence = evidence_for_video(model, item, device, args.topk)
        evidence["selected_layers"] = selected_layers.astype(np.int64)
        evidence["selected_dimensions"] = selected_dimensions.astype(np.int64)
        atomic_save_npz(target, **evidence)
        rows.append([item["key"], item["label"], item["length"], target.name, "new"])
    write_csv(output / "index.csv", ["video_key", "label", "length", "audit_file", "action"], rows)
    write_csv(
        output / "circuit_dimensions.csv",
        ["circuit_index", "layer_1based", "dimension"],
        [[index, int(layer) + 1, int(dimension)] for index, (layer, dimension) in enumerate(zip(selected_layers, selected_dimensions))],
    )
    print(f"wrote {len(rows)} resumable explanation artifacts under {output}", flush=True)


if __name__ == "__main__":
    main()

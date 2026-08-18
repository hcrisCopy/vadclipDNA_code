"""Export per-video CTNC evidence without rerunning or modifying VadCLIP."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .assets import load_assets
from .circuit import ChannelRankVerifier, load_verifier_state
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
def evidence_for_video(model: ChannelRankVerifier, item: dict, device: torch.device, topk: int) -> dict[str, np.ndarray]:
    length = int(item["length"])
    # Audit concerns hidden-channel evidence. It therefore uses a uniform
    # prompt prior and does not load, alter, or need any VadCLIP score.
    class_count = model.class_count
    outputs = model(
        item["circuit"].unsqueeze(0).to(device),
        item["last_hidden"].unsqueeze(0).to(device),
        torch.full((1, length, class_count), 1.0 / class_count, dtype=torch.float32, device=device),
        torch.tensor([length], dtype=torch.int64, device=device),
        return_channel_contribution=True,
    )
    contribution = outputs["channel_contribution"].abs()
    count = min(int(topk), contribution.shape[-2])
    values, indices = contribution[0, :length].topk(count, dim=-2)
    semantic_contribution = outputs["semantic_hidden_contribution"].abs()
    semantic_count = min(int(topk), semantic_contribution.shape[-2])
    semantic_values, semantic_indices = semantic_contribution[0, :length].topk(semantic_count, dim=-2)
    return {
        "hidden_anomaly": outputs["hidden_anomaly"][0, :length].cpu().numpy().astype(np.float32),
        "sparse_hidden_anomaly": outputs["sparse_hidden_anomaly"][0, :length].cpu().numpy().astype(np.float32),
        "semantic_anomaly": outputs["semantic_anomaly"][0, :length].cpu().numpy().astype(np.float32),
        "class_evidence": outputs["class_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "state_evidence": outputs["state_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "transition_evidence": outputs["transition_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "normalized_state": outputs["normalized_state"][0, :length].cpu().numpy().astype(np.float32),
        "nearest_normal_prototype_index": outputs["nearest_prototype_index"][0, :length].cpu().numpy().astype(np.int64),
        "nearest_normal_prototype": outputs["nearest_prototype"][0, :length].cpu().numpy().astype(np.float32),
        "prototype_residual": outputs["prototype_residual"][0, :length].cpu().numpy().astype(np.float32),
        "normalized_transition": outputs["normalized_transition"][0, :length].cpu().numpy().astype(np.float32),
        "transition_novelty": outputs["transition_novelty"][0, :length].cpu().numpy().astype(np.float32),
        "normal_context": outputs["context"].cpu().numpy().astype(np.int64),
        "gate": outputs["gates"].cpu().numpy().astype(np.float32),
        "class_gate": outputs["class_gates"].cpu().numpy().astype(np.float32),
        "transition_gate": outputs["transition_gates"].cpu().numpy().astype(np.float32),
        "state_weight": outputs["state_weights"].cpu().numpy().astype(np.float32),
        "transition_weight": outputs["transition_weights"].cpu().numpy().astype(np.float32),
        "state_correction": outputs["state_correction"].cpu().numpy().astype(np.float32),
        "state_scale": outputs["state_scales"].cpu().numpy().astype(np.float32),
        "transition_scale": outputs["transition_scales"].cpu().numpy().astype(np.float32),
        "rank_scale": outputs["rank_scales"].cpu().numpy().astype(np.float32),
        "class_gain": outputs["class_gains"].cpu().numpy().astype(np.float32),
        "semantic_binary_scale": outputs["semantic_binary_scale"].cpu().numpy().astype(np.float32),
        "semantic_probability": outputs["semantic_probability"][0, :length].cpu().numpy().astype(np.float32),
        "semantic_logit": outputs["semantic_logit"][0, :length].cpu().numpy().astype(np.float32),
        "semantic_text_weight": outputs["semantic_weights"].cpu().numpy().astype(np.float32),
        "semantic_projection_norm": outputs["semantic_projection_norm"][0, :length].cpu().numpy().astype(np.float32),
        "verification_strength": outputs["verification_strength"].reshape(1).cpu().numpy().astype(np.float32),
        "top_circuit_index_by_class": indices.cpu().numpy().astype(np.int64),
        "top_circuit_contribution_by_class": values.cpu().numpy().astype(np.float32),
        "top_semantic_hidden_index_by_class": semantic_indices.cpu().numpy().astype(np.int64),
        "top_semantic_hidden_contribution_by_class": semantic_values.cpu().numpy().astype(np.float32),
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
    model = ChannelRankVerifier(assets).to(device)
    load_verifier_state(model, model_state(args.model_path))
    rows: list[list[object]] = []
    for item in tqdm(dataset, desc=f"CTNC audit {args.split_name}", unit="video"):
        target = output / f"{item['key']}.npz"
        if target.is_file() and not args.no_resume:
            rows.append([item["key"], item["label"], item["length"], target.name, "reused"])
            continue
        evidence = evidence_for_video(model, item, device, args.topk)
        evidence["selected_layers"] = selected_layers.astype(np.int64)
        evidence["selected_dimensions"] = selected_dimensions.astype(np.int64)
        evidence["selected_text_direction"] = assets["selected_text_direction"].cpu().numpy().astype(np.float32)
        evidence["selected_text_class"] = assets["selected_text_class"].cpu().numpy().astype(np.int64)
        atomic_save_npz(target, **evidence)
        rows.append([item["key"], item["label"], item["length"], target.name, "new"])
    write_csv(output / "index.csv", ["video_key", "label", "length", "audit_file", "action"], rows)
    write_csv(
        output / "circuit_dimensions.csv",
        ["circuit_index", "layer_1based", "dimension", "dominant_anomaly_text", "selection_text", "signed_direction"],
        [
            [
                index, int(layer) + 1, int(dimension),
                assets["prompts"][int(assets["selected_text_class"][index])],
                assets["prompts"][int(assets.get("selected_by_text_class", assets["selected_text_class"])[index])],
                float(assets["selected_text_direction"][index]),
            ]
            for index, (layer, dimension) in enumerate(zip(selected_layers, selected_dimensions))
        ],
    )
    print(f"wrote {len(rows)} resumable explanation artifacts under {output}", flush=True)


if __name__ == "__main__":
    main()

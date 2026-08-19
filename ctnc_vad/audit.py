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
    )
    contribution = outputs["channel_contribution"]
    count = min(int(topk), contribution.shape[-1])
    values, indices = contribution[0, :length].topk(count, dim=-1)
    return {
        "hidden_anomaly": outputs["hidden_anomaly"][0, :length].cpu().numpy().astype(np.float32),
        "channel_anomaly": outputs["channel_anomaly"][0, :length].cpu().numpy().astype(np.float32),
        "channel_probability": outputs["channel_probability"][0, :length].cpu().numpy().astype(np.float32),
        "class_evidence": outputs["class_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "visual_score": outputs["visual_score"][0, :length].cpu().numpy().astype(np.float32),
        "semantic_score": outputs["semantic_score"][0, :length].cpu().numpy().astype(np.float32),
        "lake_logit": outputs["lake_logit"][0, :length].cpu().numpy().astype(np.float32),
        "text_probability": outputs["text_probability"][0, :length].cpu().numpy().astype(np.float32),
        "query_normalized": outputs["query_normalized"][0, :length].cpu().numpy().astype(np.float32),
        "nearest_normal_gallery_index": outputs["nearest_normal_gallery_index"][0, :length].cpu().numpy().astype(np.int64),
        "nearest_normal_gallery": outputs["nearest_normal_gallery"][0, :length].cpu().numpy().astype(np.float32),
        "nearest_normal_similarity": outputs["nearest_normal_similarity"][0, :length].cpu().numpy().astype(np.float32),
        "channel_delta": outputs["channel_delta"][0, :length].cpu().numpy().astype(np.float32),
        "channel_deviation": outputs["channel_deviation"][0, :length].cpu().numpy().astype(np.float32),
        "channel_contribution": outputs["channel_contribution"][0, :length].cpu().numpy().astype(np.float32),
        "channel_gates": outputs["channel_gates"].cpu().numpy().astype(np.float32),
        "class_top_channel_index": outputs["class_top_channel_index"][0, :length].cpu().numpy().astype(np.int64),
        "channel_evidence": outputs["channel_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "centered_channel_evidence": outputs["centered_channel_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "normal_context": outputs["context"].cpu().numpy().astype(np.int64),
        "text_temperature": outputs["text_temperature"].cpu().numpy().astype(np.float32),
        "fusion_scale": outputs["fusion_scale"].cpu().numpy().astype(np.float32),
        "verification_strength": outputs["verification_strength"].reshape(1).cpu().numpy().astype(np.float32),
        "visual_scale": outputs["visual_scale"].cpu().numpy().astype(np.float32),
        "semantic_scale": outputs["semantic_scale"].cpu().numpy().astype(np.float32),
        "top_circuit_index": indices.cpu().numpy().astype(np.int64),
        "top_circuit_deviation": values.cpu().numpy().astype(np.float32),
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

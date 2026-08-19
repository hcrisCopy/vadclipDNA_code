"""Export per-class, per-frame raw-channel circuit witnesses."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .assets import load_assets
from .baseline import build_frozen_baseline, score_sequence
from .circuit import SparseClassCircuit
from .common import atomic_save_npz, default_output_root, hidden_manifest_paths, stage_dir, write_csv
from .data import ChannelBagDataset


def model_state(path: str | Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["model_state_dict"] if isinstance(value, dict) and "model_state_dict" in value else value


@torch.no_grad()
def audit_video(model, item: dict, baseline, visual_length: int, dataset: str, device: torch.device, topk: int) -> dict[str, np.ndarray]:
    length = int(item["length"])
    _prob1, baseline_anomaly, baseline_all = score_sequence(baseline, item["clip_feature"].numpy(), visual_length, dataset, device)
    output = model(item["circuit"].unsqueeze(0).to(device), item["final_hidden"].unsqueeze(0).to(device), torch.from_numpy(baseline_all).unsqueeze(0).to(device), torch.tensor([length], dtype=torch.int64, device=device), return_contributions=True)
    contribution = output["class_channel_contribution"][0, :length].permute(0, 2, 1)
    count = min(int(topk), contribution.shape[-1])
    top_value, top_index = contribution.topk(count, dim=-1)
    zscore = output["zscore"][0, :length]
    top_zscore = torch.gather(zscore.unsqueeze(1).expand(-1, contribution.shape[1], -1), 2, top_index)
    return {
        "baseline": baseline_anomaly.astype(np.float32),
        "baseline_all": baseline_all.astype(np.float32),
        "fused": output["score"][0, :length].cpu().numpy().astype(np.float32),
        "fused_all": output["fused_probability"][0, :length].cpu().numpy().astype(np.float32),
        "class_evidence": output["class_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "circuit_class_probability": output["class_probability"][0, :length].cpu().numpy().astype(np.float32),
        "context": output["context"].cpu().numpy().astype(np.int64),
        "normalized_channel_weight": output["normalized_channel_weight"].cpu().numpy().astype(np.float32),
        "raw_channel_weight": output["raw_channel_weight"].cpu().numpy().astype(np.float32),
        "fusion_gamma": output["fusion_gamma"].cpu().numpy().astype(np.float32),
        "top_channel_index": top_index.cpu().numpy().astype(np.int64),
        "top_channel_contribution": top_value.cpu().numpy().astype(np.float32),
        "top_channel_zscore": top_zscore.cpu().numpy().astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CTSC class-specific layer/dimension evidence without changing VadCLIP.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only audit/<split-name>/ under --output-root.")
    parser.add_argument("--no-resume", action="store_true")
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
    paths = hidden_manifest_paths(args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to)
    device = torch.device(args.device)
    baseline, options = build_frozen_baseline(args.dataset, args.init_baseline_model, device)
    dataset = ChannelBagDataset(args.dataset, args.source_test_csv, args.source_path_base, paths, assets["selected_layers"].numpy(), assets["selected_dimensions"].numpy(), options.visual_length, False, args.alignment)
    model = SparseClassCircuit(assets).to(device)
    model.load_state_dict(model_state(args.model_path), strict=True)
    rows: list[list[object]] = []
    for item in tqdm(dataset, desc=f"CTSC audit {args.split_name}", unit="video"):
        target = output / f"{item['key']}.npz"
        if target.is_file() and not args.no_resume:
            rows.append([item["key"], item["label"], item["length"], target.name, "reused"])
            continue
        evidence = audit_video(model, item, baseline, options.visual_length, args.dataset, device, args.topk)
        evidence["selected_layers"] = assets["selected_layers"].cpu().numpy().astype(np.int64)
        evidence["selected_dimensions"] = assets["selected_dimensions"].cpu().numpy().astype(np.int64)
        evidence["semantic_response"] = assets["semantic_response"].cpu().numpy().astype(np.float32)
        atomic_save_npz(target, **evidence)
        rows.append([item["key"], item["label"], item["length"], target.name, "new"])
    write_csv(output / "index.csv", ["video_key", "label", "length", "audit_file", "action"], rows)
    prompts = list(assets["prompts"])
    responses = assets["semantic_response"].cpu().numpy()
    write_csv(output / "circuit_channels.csv", ["circuit_index", "layer_1based", "dimension", "anomaly_text", "signed_text_response"], [[index, int(layer) + 1, int(dimension), prompts[class_index + 1], float(responses[index, class_index])] for index, (layer, dimension) in enumerate(zip(assets["selected_layers"].cpu().numpy(), assets["selected_dimensions"].cpu().numpy())) for class_index in range(len(prompts) - 1)])
    print(f"wrote {len(rows)} resumable raw-channel audit artifacts under {output}", flush=True)


if __name__ == "__main__":
    main()

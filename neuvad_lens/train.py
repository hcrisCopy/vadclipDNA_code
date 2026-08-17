"""Single-GPU XD NeuVAD-Lens training with unchanged VadCLIP losses and schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from xd_dna.dataset import XDDNAFeatureDataset
from xd_dna.metrics import evaluate_loader
from xd_dna.train import clas_2, clas_m
from xd_dna.vadclip import add_local_vadclip_source, load_options

from .common import (
    atomic_torch_save,
    default_output_root,
    labels_for_dataset,
    load_json,
    set_seed,
    stage_dir,
    write_csv,
)
from .lens import load_lens_asset
from .model import NeuVADLensVadCLIP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NeuVAD-Lens with VadCLIP's unchanged XD optimizer, losses, validation and AP2 selection."
    )
    parser.add_argument("--train-list", required=True, help="xd_neuvad_lens_train.csv made by neuvad_lens.build_features.")
    parser.add_argument("--test-list", required=True, help="xd_neuvad_lens_test.csv made by neuvad_lens.build_features.")
    parser.add_argument("--neuron-json", required=True, help="Reusable xd_dna selected_neurons.json.")
    parser.add_argument("--lens-assets", required=True, help="lens/lens_assets.pt made by neuvad_lens.build_lens_assets.")
    parser.add_argument("--init-baseline-model", required=True, help="Official 512D VadCLIP XD checkpoint.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--gt-path", default="VadCLIP/list/gt.npy")
    parser.add_argument("--gt-segment-path", default="VadCLIP/list/gt_segment.npy")
    parser.add_argument("--gt-label-path", default="VadCLIP/list/gt_label.npy")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=[3, 6, 10])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dna-hidden-dim", type=int, default=1024)
    parser.add_argument("--dna-depth", type=int, default=3)
    parser.add_argument("--text-hidden-dim", type=int, default=512)
    parser.add_argument("--text-depth", type=int, default=2)
    parser.add_argument("--text-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume exactly from training/checkpoint_last.pth.")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only training under --output-root.")
    args = parser.parse_args()
    if (
        args.max_epoch <= 0
        or args.batch_size <= 0
        or args.lr <= 0
        or args.num_workers < 0
        or args.text_temperature <= 0
    ):
        parser.error("epochs, batch size, learning rate and text temperature must be positive; workers may be zero")
    if args.clean and args.resume:
        raise ValueError("--clean and --resume cannot be used together")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output = stage_dir(args.output_root, "training", clean=args.clean)
    checkpoint_path, best_path, history_path = output / "checkpoint_last.pth", output / "model_best.pth", output / "history.csv"
    contract = load_json(Path(args.neuron_json).resolve())
    neuron_width = int(contract["neuron_width"])
    lens = load_lens_asset(args.lens_assets)
    if str(lens["dataset"]) != "xd":
        raise ValueError(f"XD training requires an XD lens asset, got {lens['dataset']!r}")
    input_width = neuron_width + 512 + 768

    set_seed(args.seed)
    device = torch.device(args.device)
    options = load_options("xd")
    train_set = XDDNAFeatureDataset(args.train_list, options.visual_length, input_width, test_mode=False)
    test_set = XDDNAFeatureDataset(args.test_list, options.visual_length, input_width, test_mode=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = NeuVADLensVadCLIP(
        options, args.init_baseline_model, args.lens_assets, device, neuron_width,
        args.dna_hidden_dim, args.dna_depth, args.text_hidden_dim, args.text_depth, args.text_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    add_local_vadclip_source()
    from utils.tools import get_batch_label, get_prompt_text

    label_map = labels_for_dataset("xd")
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    start_epoch, best_ap2 = 0, float("-inf")
    history: list[dict] = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch, best_ap2 = int(state["epoch"]) + 1, float(state["best_ap2"])
        print(f"resume from epoch={start_epoch + 1}, best AP2={best_ap2:.6f}", flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        loss1_total = loss2_total = loss3_total = 0.0
        progress = tqdm(train_loader, desc=f"train Lens epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for visual, text_labels, lengths in progress:
            visual, lengths = visual.to(device), lengths.to(device)
            labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
            loss1 = clas_2(logits1, labels, lengths, device)
            loss2 = clas_m(logits2, labels, lengths, device)
            loss3 = torch.zeros(1, device=device)
            normal_feature = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for class_index in range(1, text_features.shape[0]):
                abnormal_feature = text_features[class_index] / text_features[class_index].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(normal_feature @ abnormal_feature)
            loss3 = loss3 / 6
            loss = loss1 + loss2 + loss3 * 1e-4
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss1_total += float(loss1.detach().item())
            loss2_total += float(loss2.detach().item())
            loss3_total += float(loss3.detach().item())
            progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}", gate=f"{float(model.residual_gate.detach().item()):.4f}")
        scheduler.step()
        metrics = evaluate_loader(
            model, test_loader, options.visual_length, gt, gtsegments, gtlabels, device,
            f"validate Lens epoch {epoch + 1}", dataset="xd", prompt_text=prompt_text,
        )
        improved = metrics.ap2 > best_ap2
        if improved:
            best_ap2 = metrics.ap2
            atomic_torch_save(best_path, model.state_dict())
        atomic_torch_save(checkpoint_path, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ap2": best_ap2,
            "selection_metric": "AP2",
            "metrics": metrics.to_dict(),
            "neuron_width": neuron_width,
            "lens_assets": args.lens_assets,
        })
        batches = max(1, len(train_loader))
        row = {
            "epoch": epoch + 1,
            "loss1": loss1_total / batches,
            "loss2": loss2_total / batches,
            "loss3": loss3_total / batches,
            "lr": optimizer.param_groups[0]["lr"],
            "gate": float(model.residual_gate.detach().item()),
            "best_ap2": best_ap2,
            "selected_best": improved,
            **metrics.to_dict(),
        }
        history = [item for item in history if int(item["epoch"]) != epoch + 1] + [row]
        history_frame = pd.DataFrame(history).sort_values("epoch")
        write_csv(history_path, history_frame.columns.tolist(), history_frame.itertuples(index=False, name=None))
        print(
            f"epoch {epoch + 1}/{args.max_epoch} | loss1={row['loss1']:.5f} loss2={row['loss2']:.5f} "
            f"| AUC1={metrics.auc1:.6f} AP1={metrics.ap1:.6f} | AUC2={metrics.auc2:.6f} AP2={metrics.ap2:.6f} "
            f"| det-mAP={metrics.detection_map_average:.2f}% | best(AP2)={best_ap2:.6f}",
            flush=True,
        )
    if not best_path.is_file():
        atomic_torch_save(best_path, model.state_dict())
    print(f"Lens training complete: best AP2={best_ap2:.6f}; model={best_path}", flush=True)


if __name__ == "__main__":
    main()

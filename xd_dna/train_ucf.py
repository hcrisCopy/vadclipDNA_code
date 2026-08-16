"""UCF-Crime DNA residual training with the official VadCLIP UCF protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import (
    UCF_TRAIN_LABELS,
    atomic_torch_save,
    default_output_root,
    load_json,
    set_seed,
    stage_dir,
    write_csv,
)
from .dataset import XDDNAFeatureDataset
from .metrics import evaluate_loader
from .model import DNAResidualVadCLIP
from .train import clas_2, clas_m
from .vadclip import add_local_vadclip_source, load_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DNA residual injection with VadCLIP's original UCF two-loader protocol.")
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--neuron-json", default="")
    parser.add_argument("--init-baseline-model", required=True, help="Official 512D VadCLIP UCF checkpoint.")
    parser.add_argument("--output-root", default="../vadclipDNA_data/ucf_normal_negative_top64")
    parser.add_argument("--gt-path", default="VadCLIP/list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="VadCLIP/list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="VadCLIP/list/gt_label_ucf.npy")
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--eval-interval-samples", type=int, default=1280)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--residual-hidden-dim", type=int, default=1024)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.max_epoch <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.eval_interval_samples <= 0 or args.num_workers < 0:
        parser.error("epochs, batch size, learning rate and evaluation interval must be positive; workers may be zero")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.clean and args.resume:
        raise ValueError("--clean and --resume cannot be used together")

    output = stage_dir(args.output_root, "training", clean=args.clean)
    root = output.parent
    neuron_json = Path(args.neuron_json).resolve() if args.neuron_json else (root / "localization" / "selected_neurons.json").resolve()
    contract = load_json(neuron_json)
    neuron_width, input_width = int(contract["neuron_width"]), int(contract["input_width"])
    if input_width != neuron_width + 512:
        raise ValueError(f"{neuron_json}: invalid selected-neuron input contract")
    checkpoint_path, best_path, history_path = output / "checkpoint_last.pth", output / "model_best.pth", output / "history.csv"

    set_seed(args.seed)
    device = torch.device(args.device)
    options = load_options("ucf")
    normal_set = XDDNAFeatureDataset(args.train_list, options.visual_length, input_width, test_mode=False, normal=True)
    anomaly_set = XDDNAFeatureDataset(args.train_list, options.visual_length, input_width, test_mode=False, normal=False)
    test_set = XDDNAFeatureDataset(args.test_list, options.visual_length, input_width, test_mode=True)
    normal_loader = DataLoader(normal_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    anomaly_loader = DataLoader(anomaly_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if not len(normal_loader) or not len(anomaly_loader):
        raise RuntimeError("UCF normal/anomaly loaders need at least one full batch each")
    model = DNAResidualVadCLIP(options, args.init_baseline_model, device, neuron_width, args.residual_hidden_dim, args.residual_depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    add_local_vadclip_source()
    from utils.tools import get_batch_label, get_prompt_text

    prompt_text = get_prompt_text(UCF_TRAIN_LABELS)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    start_epoch, best_auc1 = 0, float("-inf")
    history: list[dict] = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is absent: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch, best_auc1 = int(state["epoch"]) + 1, float(state["best_auc1"])
        print(f"resume from epoch={start_epoch + 1}, best AUC1={best_auc1:.6f}", flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        normal_iter, anomaly_iter = iter(normal_loader), iter(anomaly_loader)
        loss1_total = loss2_total = loss3_total = 0.0
        evaluated_this_epoch = False
        progress = tqdm(range(min(len(normal_loader), len(anomaly_loader))), desc=f"train UCF epoch {epoch + 1}/{args.max_epoch}", unit="batch")
        for index in progress:
            normal_features, normal_labels, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_labels, anomaly_lengths = next(anomaly_iter)
            visual = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            labels = get_batch_label(list(normal_labels) + list(anomaly_labels), prompt_text, UCF_TRAIN_LABELS).to(device)
            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
            loss1 = clas_2(logits1, labels, lengths, device)
            loss2 = clas_m(logits2, labels, lengths, device)
            loss3 = torch.zeros(1, device=device)
            normal_feature = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for class_index in range(1, text_features.shape[0]):
                abnormal_feature = text_features[class_index] / text_features[class_index].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(normal_feature @ abnormal_feature)
            loss3 = loss3 / 13 * 1e-1
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss1_total += float(loss1.detach().item())
            loss2_total += float(loss2.detach().item())
            loss3_total += float(loss3.detach().item())
            step = index * args.batch_size * 2
            progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}", gate=f"{float(model.residual_gate.detach().item()):.4f}")
            if step and step % args.eval_interval_samples == 0:
                evaluated_this_epoch = True
                metrics = evaluate_loader(
                    model, test_loader, options.visual_length, gt, gtsegments, gtlabels, device,
                    f"validate UCF epoch {epoch + 1} step {step}", dataset="ucf", prompt_text=prompt_text,
                )
                improved = metrics.auc1 > best_auc1
                if improved:
                    best_auc1 = metrics.auc1
                    atomic_torch_save(best_path, model.state_dict())
                history.append({
                    "epoch": epoch + 1, "step": step, "loss1": loss1_total / (index + 1),
                    "loss2": loss2_total / (index + 1), "loss3": loss3_total / (index + 1),
                    "lr": optimizer.param_groups[0]["lr"], "gate": float(model.residual_gate.detach().item()),
                    "selection_metric": "AUC1", "best_auc1": best_auc1, "selected_best": improved,
                    **metrics.to_dict(),
                })
                print(
                    f"epoch {epoch + 1} step {step} | AUC1={metrics.auc1:.6f} AP1={metrics.ap1:.6f} "
                    f"| AUC2={metrics.auc2:.6f} AP2={metrics.ap2:.6f} "
                    f"| Ano-AUC1={metrics.ano_auc1:.6f} Ano-AUC2={metrics.ano_auc2:.6f} "
                    f"| best(AUC1)={best_auc1:.6f}", flush=True,
                )
                model.train()
        scheduler.step()
        # The official launcher validates during the loop.  This fallback only
        # protects short debugging runs that never reach its 1,280-sample gate.
        if not evaluated_this_epoch and not best_path.exists():
            metrics = evaluate_loader(model, test_loader, options.visual_length, gt, gtsegments, gtlabels, device, f"validate UCF epoch {epoch + 1}", dataset="ucf", prompt_text=prompt_text)
            best_auc1 = metrics.auc1
            atomic_torch_save(best_path, model.state_dict())
            model.train()
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False), strict=True)
        atomic_torch_save(checkpoint_path, {
            "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "best_auc1": best_auc1, "selection_metric": "AUC1",
            "neuron_width": neuron_width,
        })
        if history:
            history_frame = pd.DataFrame(history)
            write_csv(history_path, history_frame.columns.tolist(), history_frame.itertuples(index=False, name=None))
    if not best_path.exists():
        atomic_torch_save(best_path, model.state_dict())
    print(f"UCF training complete: best AUC1={best_auc1:.6f}; model={best_path}", flush=True)


if __name__ == "__main__":
    main()

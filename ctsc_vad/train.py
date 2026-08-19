"""Train only direct CTSC channel weights; VadCLIP remains frozen."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .assets import load_assets
from .baseline import build_frozen_baseline
from .cache import prepare_cache
from .circuit import SparseClassCircuit, circuit_loss
from .common import atomic_torch_save, default_output_root, hidden_manifest_paths, save_json, set_seed, stage_dir, write_csv
from .data import ChannelBagDataset
from .evaluate import collect_predictions, summarize


def defaults(dataset: str) -> tuple[int, float, list[int]]:
    return (96, 1e-3, [6, 9]) if dataset == "xd" else (64, 1e-3, [6, 9])


def load_checkpoint(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or "model_state_dict" not in value:
        raise ValueError(f"{path}: expected CTSC training checkpoint")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTSC class-specific sparse raw-channel circuits with frozen VadCLIP.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".")
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True)
    parser.add_argument("--init-baseline-model", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reader-lr", type=float, default=None)
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=None)
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--top-fraction", type=float, default=0.125, help="MIL temporal fraction. It selects time segments, never channels.")
    parser.add_argument("--normal-frame-weight", type=float, default=0.25)
    parser.add_argument("--preserve-weight", type=float, default=0.01)
    parser.add_argument("--channel-entropy-weight", type=float, default=0.01, help="Lower entropy selects a sparse, explainable raw channel circuit.")
    parser.add_argument("--temporal-smoothness-weight", type=float, default=0.01)
    parser.add_argument("--gate-initial-logit", type=float, default=-2.0)
    parser.add_argument("--fusion-initial-logit", type=float, default=-2.0)
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-train-hidden-manifest", action="store_true")
    parser.add_argument("--no-resume-baseline-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only training/ under --output-root.")
    args = parser.parse_args()
    if args.clean and args.resume:
        raise ValueError("--clean and --resume cannot be combined")
    if args.max_epoch <= 0 or args.num_workers < 0 or not 0 < args.top_fraction <= 1 or args.scheduler_rate <= 0:
        parser.error("invalid epoch/worker/top-fraction/scheduler value")
    if min(args.normal_frame_weight, args.preserve_weight, args.channel_entropy_weight, args.temporal_smoothness_weight) < 0:
        parser.error("loss weights must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    batch_size, learning_rate, milestones = defaults(args.dataset)
    args.batch_size = batch_size if args.batch_size is None else args.batch_size
    args.reader_lr = learning_rate if args.reader_lr is None else args.reader_lr
    args.scheduler_milestones = milestones if args.scheduler_milestones is None else args.scheduler_milestones
    if args.batch_size <= 0 or args.reader_lr <= 0:
        parser.error("batch-size and reader-lr must be positive")

    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "training", clean=args.clean)
    last_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError("assets dataset does not match --dataset")
    layers, dimensions = assets["selected_layers"].numpy(), assets["selected_dimensions"].numpy()
    train_hidden = hidden_manifest_paths(args.train_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to)
    test_hidden = hidden_manifest_paths(args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to)
    device = torch.device(args.device)
    set_seed(args.seed)
    baseline, options = build_frozen_baseline(args.dataset, args.init_baseline_model, device)
    cache_reference = ChannelBagDataset(args.dataset, args.source_train_csv, args.source_path_base, train_hidden, layers, dimensions, options.visual_length, False, args.alignment, not args.strict_train_hidden_manifest)
    cache = prepare_cache(cache_reference, baseline, options.visual_length, args.dataset, device, output.parent / "baseline_cache" / "train", reuse=not args.no_resume_baseline_cache)
    train_set = ChannelBagDataset(args.dataset, args.source_train_csv, args.source_path_base, train_hidden, layers, dimensions, options.visual_length, True, args.alignment, not args.strict_train_hidden_manifest, cache)
    test_set = ChannelBagDataset(args.dataset, args.source_test_csv, args.source_path_base, test_hidden, layers, dimensions, options.visual_length, False, args.alignment)
    write_csv(output / "missing_train_hidden.csv", ["video_key"], [[key] for key in train_set.skipped])
    if train_set.skipped:
        print(f"warning: skipped {len(train_set.skipped)} training videos without hidden states; see {output / 'missing_train_hidden.csv'}", flush=True)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = SparseClassCircuit(assets, args.gate_initial_logit, args.fusion_initial_logit).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.reader_lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    gt, gtsegments, gtlabels = np.load(args.gt_path), np.load(args.gt_segment_path, allow_pickle=True), np.load(args.gt_label_path, allow_pickle=True)
    start, best, history = 0, float("-inf"), []
    if args.resume:
        checkpoint = load_checkpoint(last_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best, history = int(checkpoint["epoch"]) + 1, float(checkpoint["best_metric"]), list(checkpoint.get("history", []))
        print(f"resume from epoch {start + 1}; best={best:.6f}", flush=True)
    for epoch in range(start, args.max_epoch):
        model.train()
        totals, batches = {name: 0.0 for name in ("loss", "fused_bag", "circuit_bag", "normal", "preserve", "entropy", "temporal")}, 0
        progress = tqdm(loader, desc=f"CTSC train {epoch + 1}/{args.max_epoch}", unit="batch")
        for circuit, final_hidden, baseline_probability, lengths, targets in progress:
            output_values = model(circuit.to(device, non_blocking=True), final_hidden.to(device, non_blocking=True), baseline_probability.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
            loss, pieces = circuit_loss(output_values, targets.to(device, non_blocking=True), lengths.to(device, non_blocking=True), args.top_fraction, args.normal_frame_weight, args.preserve_weight, args.channel_entropy_weight, args.temporal_smoothness_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches += 1
            totals["loss"] += float(loss.detach())
            for name, value in pieces.items():
                totals[name] += value
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", gamma=f"{float(output_values['fusion_gamma'].mean().detach()):.3f}")
        scheduler.step()
        predictions, _labels, _rows = collect_predictions(model, test_set, baseline, options.visual_length, args.dataset, device, None, False, f"CTSC validate {epoch + 1}/{args.max_epoch}")
        validation = summarize(predictions, gt, gtsegments, gtlabels, args.dataset)
        final = validation["classwise_poe"]
        selection = "ap2" if args.dataset == "xd" else "ap1"
        metric = float(final[selection])
        improved = metric > best
        if improved:
            best = metric
            atomic_torch_save(best_path, model.state_dict())
        row = {"epoch": epoch + 1, **{name: value / max(1, batches) for name, value in totals.items()}, "baseline_ap2": float(validation["baseline"]["ap2"]), "baseline_dmap": float(validation["baseline"]["detection_map_average"]), "circuit_auc": float(validation["channel_circuit_only"]["auc2"]), "circuit_ap": float(validation["channel_circuit_only"]["ap2"]), "circuit_dmap": float(validation["channel_circuit_only"]["detection_map_average"]), "final_auc": float(final["auc2"]), "final_ap": float(final[selection]), "final_dmap": float(final["detection_map_average"]), "selected_best": improved}
        history = [old for old in history if int(old["epoch"]) != epoch + 1] + [row]
        history.sort(key=lambda item: int(item["epoch"]))
        atomic_torch_save(last_path, {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_metric": best, "history": history, "args": vars(args)})
        write_csv(output / "history.csv", history[0].keys(), [item.values() for item in history])
        save_json(output / "validation_last.json", validation)
        print(f"epoch {epoch + 1}/{args.max_epoch} | loss={row['loss']:.5f} | baseline AP2={row['baseline_ap2']:.6f} dMAP={row['baseline_dmap']:.2f}% | circuit AUC/AP={row['circuit_auc']:.6f}/{row['circuit_ap']:.6f} dMAP={row['circuit_dmap']:.2f}% | final AUC={row['final_auc']:.6f} {selection}={row['final_ap']:.6f} dMAP={row['final_dmap']:.2f}% | best={best:.6f}", flush=True)
    save_json(output / "summary.json", {"dataset": args.dataset, "assets": args.assets, "baseline_checkpoint": args.init_baseline_model, "best_metric": best, "selection_metric": "ap2" if args.dataset == "xd" else "ap1", "train_videos": len(train_set), "test_videos": len(test_set), "skipped_train_hidden": train_set.skipped})
    print(f"training complete; best classwise circuit expert is {best_path}", flush=True)


if __name__ == "__main__":
    main()

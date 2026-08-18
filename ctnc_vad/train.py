"""Train the small CTNC circuit reader while keeping VadCLIP entirely frozen."""
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
from .baseline_cache import prepare_score_cache
from .circuit import ChannelRankVerifier, verifier_loss
from .common import (
    atomic_torch_save,
    default_output_root,
    hidden_manifest_paths,
    load_json,
    save_json,
    set_seed,
    stage_dir,
    write_csv,
)
from .dataset import HiddenBagDataset
from .evaluate import collect_predictions, summarize_predictions


def dataset_defaults(dataset: str) -> tuple[int, float, list[int]]:
    # VadCLIP keeps its original learning rate. This is the separate, tiny
    # CTNC verifier (roughly hundreds of trainable scalars), so it needs a
    # reader-scale rate rather than the baseline full-model fine-tuning rate.
    return (96, 1e-3, [6, 9]) if dataset == "xd" else (64, 1e-3, [6, 9])


def checkpoint_state(path: Path) -> dict:
    content = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(content, dict) or "model_state_dict" not in content:
        raise ValueError(f"{path}: expected a CTNC training checkpoint")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CTNC text-conditioned channel verifier with frozen VadCLIP evaluation.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--source-test-csv", required=True)
    parser.add_argument("--source-path-base", default=".", help="Base for relative VadCLIP 512D feature paths in both source CSVs.")
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--hidden-path-base", default=".")
    parser.add_argument("--hidden-prefix-from", default="")
    parser.add_argument("--hidden-prefix-to", default="")
    parser.add_argument("--assets", required=True, help="discovery/circuit_assets.pt")
    parser.add_argument("--init-baseline-model", required=True, help="Unmodified VadCLIP checkpoint; it remains frozen.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--gt-segment-path", required=True)
    parser.add_argument("--gt-label-path", required=True)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--semantic-lr", type=float, default=None,
        help="Learning rate only for the new all-channel text reader; defaults to 10x --lr because it starts from a frozen text direction.",
    )
    parser.add_argument("--scheduler-milestones", type=int, nargs="+", default=None)
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normal-frame-weight", type=float, default=0.25)
    parser.add_argument("--preserve-weight", type=float, default=0.01, help="Keep uncertain outputs close to the frozen baseline.")
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument(
        "--semantic-anchor-weight", type=float, default=0.05,
        help="Keep state and all-channel text-direction corrections near their frozen CLIP priors.",
    )
    parser.add_argument(
        "--hidden-mil-weight", type=float, default=1.0,
        help="Video-label MIL supervision for both sparse and all-channel hidden circuits.",
    )
    parser.add_argument("--gate-initial-logit", type=float, default=0.0)
    parser.add_argument(
        "--verification-initial-logit", type=float, default=-3.0,
        help="Initial sparse-circuit likelihood scale after sigmoid; -3 starts at about 0.05.",
    )
    parser.add_argument("--alignment", choices=["strict", "crop_hidden", "pad_hidden"], default="crop_hidden")
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--strict-train-hidden-manifest",
        action="store_true",
        help="Fail when a training video is absent from the train hidden manifest. The default is to skip and record only those training videos.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume exactly from training/checkpoint_last.pth.")
    parser.add_argument("--no-resume-baseline-cache", action="store_true", help="Recompute frozen-baseline training scores.")
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only training/ under --output-root.")
    args = parser.parse_args()
    if args.clean and args.resume:
        raise ValueError("--clean and --resume cannot be used together")
    if args.max_epoch <= 0 or args.num_workers < 0 or args.scheduler_rate <= 0:
        parser.error("epochs and scheduler rate must be positive; workers may be zero")
    if min(
        args.normal_frame_weight, args.preserve_weight, args.sparsity_weight,
        args.hidden_mil_weight, args.semantic_anchor_weight,
    ) < 0:
        parser.error("loss weights must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    default_batch, default_lr, default_milestones = dataset_defaults(args.dataset)
    args.batch_size = default_batch if args.batch_size is None else args.batch_size
    args.lr = default_lr if args.lr is None else args.lr
    args.scheduler_milestones = default_milestones if args.scheduler_milestones is None else args.scheduler_milestones
    args.semantic_lr = (10.0 * args.lr) if args.semantic_lr is None else args.semantic_lr
    if args.batch_size <= 0 or args.lr <= 0 or args.semantic_lr <= 0:
        parser.error("batch size and learning rates must be positive")

    output_root = args.output_root or str(default_output_root(args.dataset))
    output = stage_dir(output_root, "training", clean=args.clean)
    last_path, best_path = output / "checkpoint_last.pth", output / "model_best.pth"
    assets = load_assets(args.assets, "cpu")
    if assets["dataset"] != args.dataset:
        raise ValueError(f"asset dataset={assets['dataset']!r} does not match --dataset={args.dataset!r}")
    selected_layers = assets["selected_layers"].cpu().numpy()
    selected_dimensions = assets["selected_dimensions"].cpu().numpy()
    set_seed(args.seed)
    train_hidden = hidden_manifest_paths(
        args.train_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to
    )
    test_hidden = hidden_manifest_paths(
        args.test_hidden_manifest, args.hidden_path_base, args.hidden_prefix_from, args.hidden_prefix_to
    )
    device = torch.device(args.device)
    baseline, options = build_frozen_baseline(args.dataset, args.init_baseline_model, device)
    # This inference-only dataset preserves the original source order and
    # lets us cache baseline scores once. Scores are reader inputs, never MIL
    # pseudo-labels or channel-discovery targets.
    cache_reference = HiddenBagDataset(
        args.dataset, args.source_train_csv, args.source_path_base, train_hidden, selected_layers, selected_dimensions,
        options.visual_length, False, args.alignment, not args.strict_train_hidden_manifest,
    )
    cache_dir = output.parent / "baseline_cache" / "train"
    cached_scores = prepare_score_cache(
        cache_reference, baseline, options.visual_length, args.dataset, device, cache_dir,
        reuse=not args.no_resume_baseline_cache, progress="Cache frozen baseline train scores",
    )
    train_set = HiddenBagDataset(
        args.dataset, args.source_train_csv, args.source_path_base, train_hidden, selected_layers, selected_dimensions,
        options.visual_length, True, args.alignment, not args.strict_train_hidden_manifest, cached_scores,
    )
    test_set = HiddenBagDataset(
        args.dataset, args.source_test_csv, args.source_path_base, test_hidden, selected_layers, selected_dimensions,
        options.visual_length, False, args.alignment, False,
    )
    write_csv(output / "missing_train_hidden.csv", ["video_key"], [[key] for key in train_set.skipped])
    if train_set.skipped:
        print(
            f"warning: skipped {len(train_set.skipped)} training videos without cached hidden states; "
            f"see {output / 'missing_train_hidden.csv'}",
            flush=True,
        )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = ChannelRankVerifier(assets, args.gate_initial_logit, args.verification_initial_logit).to(device)
    # The full semantic reader begins exactly at frozen CLIP text directions.
    # Its few correction/calibration parameters need to move faster than the
    # sparse gates; this does not change VadCLIP or its optimizer at all.
    semantic_parameters = [
        model.semantic_correction,
        model.semantic_bias,
        model.semantic_rank_scale_logits,
    ]
    semantic_ids = {id(parameter) for parameter in semantic_parameters}
    circuit_parameters = [parameter for parameter in model.parameters() if id(parameter) not in semantic_ids]
    optimizer = torch.optim.AdamW([
        {"params": circuit_parameters, "lr": args.lr},
        {"params": semantic_parameters, "lr": args.semantic_lr},
    ])
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    start_epoch, best_metric, history = 0, float("-inf"), []
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"--resume requested but {last_path} is absent")
        checkpoint = checkpoint_state(last_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        history = list(checkpoint.get("history", []))
        print(f"resume from epoch {start_epoch + 1}; best={best_metric:.6f}", flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        totals = {
            "loss": 0.0, "bag": 0.0, "hidden_bag": 0.0, "normal": 0.0,
            "semantic_bag": 0.0, "semantic_normal": 0.0,
            "preserve": 0.0, "sparse": 0.0, "semantic_anchor": 0.0,
        }
        batches = 0
        progress = tqdm(train_loader, desc=f"CTNC train {epoch + 1}/{args.max_epoch}", unit="batch")
        for circuit, last_hidden, baseline_probability, lengths, class_targets in progress:
            circuit = circuit.to(device, non_blocking=True)
            last_hidden = last_hidden.to(device, non_blocking=True)
            baseline_probability = baseline_probability.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            class_targets = class_targets.to(device, non_blocking=True)
            outputs = model(circuit, last_hidden, baseline_probability, lengths)
            loss, pieces = verifier_loss(
                outputs,
                class_targets,
                lengths,
                args.normal_frame_weight,
                args.preserve_weight,
                args.sparsity_weight,
                args.hidden_mil_weight,
                args.semantic_anchor_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches += 1
            totals["loss"] += float(loss.detach())
            for name in (
                "bag", "hidden_bag", "semantic_bag", "normal", "semantic_normal",
                "preserve", "sparse", "semantic_anchor",
            ):
                totals[name] += pieces[name]
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                strength=f"{float(outputs['verification_strength'].detach()):.3f}",
            )
        scheduler.step()

        # Same model mode and official frame metrics as VadCLIP; the frozen baseline is deliberately re-evaluated.
        predictions, labels, _rows = collect_predictions(
            model, test_set, baseline, options.visual_length, args.dataset, device,
            prediction_dir=None, reuse_predictions=False, progress=f"CTNC validate {epoch + 1}/{args.max_epoch}",
        )
        validation = summarize_predictions(predictions, labels, gt, gtsegments, gtlabels, args.dataset)
        final_metrics = validation["rank_verified"]
        selection_name = "ap2" if args.dataset == "xd" else "ap1"
        metric = float(final_metrics[selection_name])
        improved = metric > best_metric
        if improved:
            best_metric = metric
            atomic_torch_save(best_path, model.state_dict())
        row = {
            "epoch": epoch + 1,
            "loss": totals["loss"] / max(1, batches),
            "bag_loss": totals["bag"] / max(1, batches),
            "hidden_bag_loss": totals["hidden_bag"] / max(1, batches),
            "semantic_bag_loss": totals["semantic_bag"] / max(1, batches),
            "normal_loss": totals["normal"] / max(1, batches),
            "semantic_normal_loss": totals["semantic_normal"] / max(1, batches),
            "preserve_loss": totals["preserve"] / max(1, batches),
            "gate_mean": totals["sparse"] / max(1, batches),
            "semantic_anchor_loss": totals["semantic_anchor"] / max(1, batches),
            "baseline_ap2": float(validation["baseline"]["ap2"]),
            "evidence_auc": float(validation["channel_evidence_only"]["auc"]),
            "evidence_ap": float(validation["channel_evidence_only"]["ap"]),
            "sparse_evidence_ap": float(validation["sparse_evidence_only"]["ap"]),
            "semantic_evidence_ap": float(validation["semantic_evidence_only"]["ap"]),
            "final_auc": float(final_metrics["auc2"]),
            "final_ap": float(final_metrics[selection_name]),
            "final_dmap": float(final_metrics["detection_map_average"]),
            "selection_metric": selection_name,
            "selected_best": improved,
        }
        history = [entry for entry in history if int(entry["epoch"]) != epoch + 1] + [row]
        history.sort(key=lambda entry: int(entry["epoch"]))
        atomic_torch_save(last_path, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "selection_metric": selection_name,
            "history": history,
            "args": vars(args),
        })
        write_csv(output / "history.csv", history[0].keys(), [row.values() for row in history])
        save_json(output / "validation_last.json", validation)
        print(
            f"epoch {epoch + 1}/{args.max_epoch} | loss={row['loss']:.5f} | "
            f"baseline AP2={row['baseline_ap2']:.6f} | hidden-only AUC/AP={row['evidence_auc']:.6f}/{row['evidence_ap']:.6f} | "
            f"sparse/semantic AP={row['sparse_evidence_ap']:.6f}/{row['semantic_evidence_ap']:.6f} | "
            f"final AUC={row['final_auc']:.6f} {selection_name}={row['final_ap']:.6f} dMAP={row['final_dmap']:.2f}% | "
            f"best={best_metric:.6f}",
            flush=True,
        )
    save_json(output / "summary.json", {
        "dataset": args.dataset,
        "assets": args.assets,
        "baseline_checkpoint": args.init_baseline_model,
        "selection_metric": "ap2" if args.dataset == "xd" else "ap1",
        "best_metric": best_metric,
        "train_videos": len(train_set),
        "test_videos": len(test_set),
        "skipped_train_hidden": train_set.skipped,
        "baseline_cache": str(cache_dir),
    })
    print(f"training complete; best frozen-baseline sidecar is {best_path}", flush=True)


if __name__ == "__main__":
    main()

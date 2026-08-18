"""Reusable frozen-baseline + CTNC evaluation loop with resumable prediction artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import score_sequence
from .circuit import ChannelRankVerifier
from .common import atomic_save_npz, write_csv
from .metrics import metrics_from_predictions, score_only_metrics


def valid_prediction(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        artifact = np.load(path, allow_pickle=False)
        try:
            return all(
                name in artifact.files and len(artifact[name]) == length
                for name in ("prob1", "prob2", "evidence", "verified", "prob2_all", "verified_all", "class_evidence")
            )
        finally:
            artifact.close()
    except Exception:
        return False


@torch.no_grad()
def verifier_sequence(
    model: ChannelRankVerifier,
    circuit: torch.Tensor,
    last_hidden: torch.Tensor,
    baseline_probability: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    length = len(circuit)
    outputs = model(
        circuit.unsqueeze(0).to(device),
        last_hidden.unsqueeze(0).to(device),
        torch.from_numpy(baseline_probability).unsqueeze(0).to(device),
        torch.tensor([length], dtype=torch.int64, device=device),
    )
    detail = {
        "context": outputs["context"].detach().cpu().numpy().astype(np.int64),
        "hidden_anomaly": outputs["hidden_anomaly"][0, :length].detach().cpu().numpy().astype(np.float32),
        "class_evidence": outputs["class_evidence"][0, :length].detach().cpu().numpy().astype(np.float32),
        "class_gains": outputs["class_gains"].detach().cpu().numpy().astype(np.float32),
        "verification_strength": outputs["verification_strength"].reshape(1).detach().cpu().numpy().astype(np.float32),
    }
    return (
        outputs["score"][0, :length].detach().cpu().numpy().astype(np.float32),
        outputs["verified_all"][0, :length].detach().cpu().numpy().astype(np.float32),
        detail,
    )


def collect_predictions(
    circuit_model,
    dataset,
    baseline_model,
    visual_length: int,
    dataset_name: str,
    device: torch.device,
    prediction_dir: Path | None,
    reuse_predictions: bool,
    progress: str,
) -> tuple[dict[str, list[np.ndarray]], list[str], list[list[object]]]:
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    output = {"prob1": [], "prob2": [], "prob2_all": [], "evidence": [], "verified": [], "verified_all": []}
    labels: list[str] = []
    index_rows: list[list[object]] = []
    circuit_model.eval()
    baseline_model.eval()
    for item in tqdm(dataset, desc=progress, unit="video"):
        length = int(item["length"])
        target = prediction_dir / f"{item['key']}.npz" if prediction_dir is not None else None
        if target is not None and reuse_predictions and valid_prediction(target, length):
            artifact = np.load(target, allow_pickle=False)
            try:
                result = {name: np.asarray(artifact[name], dtype=np.float32) for name in output}
                context = int(np.asarray(artifact["context"]).reshape(-1)[0])
            finally:
                artifact.close()
            action = "reused"
        else:
            feature = item["clip_feature"].numpy()
            if len(feature) != length:
                raise RuntimeError(f"{item['key']}: feature/hidden length mismatch after alignment")
            probability1, probability2, probability2_all = score_sequence(
                baseline_model, feature, visual_length, dataset_name, device
            )
            verified, verified_all, detail = verifier_sequence(
                circuit_model, item["circuit"], item["last_hidden"], probability2_all, device
            )
            result = {
                "prob1": probability1,
                "prob2": probability2,
                "prob2_all": probability2_all,
                "evidence": detail["hidden_anomaly"],
                "verified": verified,
                "verified_all": verified_all,
            }
            context = int(detail["context"][0])
            if target is not None:
                atomic_save_npz(target, **result, **detail)
            action = "new"
        for name, values in result.items():
            output[name].append(values)
        labels.append(str(item["label"]))
        index_rows.append([item["key"], item["label"], length, context, action, target.name if target is not None else ""])
    return output, labels, index_rows


def summarize_predictions(predictions: dict[str, list[np.ndarray]], labels: list[str], gt: np.ndarray, gtsegments: np.ndarray, gtlabels: np.ndarray, dataset: str) -> dict:
    baseline = metrics_from_predictions(
        predictions["prob1"], predictions["prob2"], predictions["prob2_all"], gt, gtsegments, gtlabels, dataset, labels
    )
    rectified = metrics_from_predictions(
        predictions["verified"], predictions["verified"], predictions["verified_all"], gt, gtsegments, gtlabels, dataset, labels
    )
    return {
        "baseline": baseline.to_dict(),
        "channel_evidence_only": score_only_metrics(predictions["evidence"], gt),
        "rank_verified": rectified.to_dict(),
    }


def write_prediction_index(path: Path, rows: list[list[object]]) -> None:
    write_csv(path, ["video_key", "label", "length", "normal_context", "action", "prediction_file"], rows)

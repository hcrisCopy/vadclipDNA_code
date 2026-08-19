"""Resumable official evaluation for the external CTSC circuit expert."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .baseline import score_sequence
from .common import atomic_save_npz, write_csv
from .metrics import evaluate_vadclip


def valid_prediction(path: Path, length: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, allow_pickle=False)
        try:
            names = {"prob1", "prob2", "prob2_all", "circuit", "circuit_all", "fused", "fused_all", "class_evidence", "context"}
            return names <= set(value.files) and all(len(value[name]) == length for name in ("prob1", "prob2", "prob2_all", "circuit", "circuit_all", "fused", "fused_all", "class_evidence"))
        finally:
            value.close()
    except Exception:
        return False


@torch.no_grad()
def circuit_sequence(model, circuit: torch.Tensor, final_hidden: torch.Tensor, baseline_probability: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    length = len(circuit)
    output = model(circuit.unsqueeze(0).to(device), final_hidden.unsqueeze(0).to(device), torch.from_numpy(baseline_probability).unsqueeze(0).to(device), torch.tensor([length], dtype=torch.int64, device=device))
    detail = {
        "context": output["context"].cpu().numpy().astype(np.int64),
        "class_evidence": output["class_evidence"][0, :length].cpu().numpy().astype(np.float32),
        "circuit_class_probability": output["class_probability"][0, :length].cpu().numpy().astype(np.float32),
        "circuit_all": output["circuit_distribution"][0, :length].cpu().numpy().astype(np.float32),
        "fusion_gamma": output["fusion_gamma"].cpu().numpy().astype(np.float32),
    }
    return output["score"][0, :length].cpu().numpy().astype(np.float32), output["fused_probability"][0, :length].cpu().numpy().astype(np.float32), detail


def collect_predictions(model, dataset, baseline, visual_length: int, dataset_name: str, device: torch.device, prediction_dir: Path | None, reuse: bool, progress: str) -> tuple[dict[str, list[np.ndarray]], list[str], list[list[object]]]:
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    values = {name: [] for name in ("prob1", "prob2", "prob2_all", "circuit", "circuit_all", "fused", "fused_all")}
    labels: list[str] = []
    rows: list[list[object]] = []
    model.eval()
    baseline.eval()
    for item in tqdm(dataset, desc=progress, unit="video"):
        key, length = str(item["key"]), int(item["length"])
        target = None if prediction_dir is None else prediction_dir / f"{key}.npz"
        if target is not None and reuse and valid_prediction(target, length):
            artifact = np.load(target, allow_pickle=False)
            try:
                result = {name: np.asarray(artifact[name], dtype=np.float32) for name in values}
                context = int(np.asarray(artifact["context"]).reshape(-1)[0])
            finally:
                artifact.close()
            action = "reused"
        else:
            prob1, prob2, prob2_all = score_sequence(baseline, item["clip_feature"].numpy(), visual_length, dataset_name, device)
            fused, fused_all, detail = circuit_sequence(model, item["circuit"], item["final_hidden"], prob2_all, device)
            result = {"prob1": prob1, "prob2": prob2, "prob2_all": prob2_all, "circuit": 1.0 - detail["circuit_all"][:, 0], "circuit_all": detail["circuit_all"], "fused": fused, "fused_all": fused_all}
            context = int(detail["context"][0])
            if target is not None:
                atomic_save_npz(target, **result, **detail)
            action = "new"
        for name in values:
            values[name].append(result[name])
        labels.append(str(item["label"]))
        rows.append([key, item["label"], length, context, action, target.name if target is not None else ""])
    return values, labels, rows


def summarize(predictions: dict[str, list[np.ndarray]], gt: np.ndarray, gtsegments: np.ndarray, gtlabels: np.ndarray, dataset: str) -> dict:
    baseline = evaluate_vadclip(predictions["prob1"], predictions["prob2"], predictions["prob2_all"], gt, gtsegments, gtlabels, dataset)
    fused = evaluate_vadclip(predictions["fused"], predictions["fused"], predictions["fused_all"], gt, gtsegments, gtlabels, dataset)
    circuit = evaluate_vadclip(predictions["circuit"], predictions["circuit"], predictions["circuit_all"], gt, gtsegments, gtlabels, dataset)
    return {"baseline": baseline.to_dict(), "channel_circuit_only": circuit.to_dict(), "classwise_poe": fused.to_dict()}


def write_prediction_index(path: Path, rows: list[list[object]]) -> None:
    write_csv(path, ["video_key", "label", "length", "normal_context", "action", "prediction_file"], rows)

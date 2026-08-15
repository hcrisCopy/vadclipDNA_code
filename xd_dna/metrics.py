"""XD evaluation that reproduces the unmodified VadCLIP test metric protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .common import XD_LABELS
from .vadclip import add_local_vadclip_source


@dataclass
class XDMetrics:
    auc1: float
    ap1: float
    auc2: float
    ap2: float
    detection_map_by_iou: dict[str, float]
    detection_map_average: float

    def to_dict(self) -> dict:
        return asdict(self)


def _chunk_lengths(length: int, maxlen: int) -> torch.Tensor:
    """Replicate the length construction in VadCLIP/src/xd_test.py."""
    remaining = int(length)
    values = torch.zeros(int(length / maxlen) + 1, dtype=torch.int64)
    for index in range(len(values)):
        if index == 0 and length < maxlen:
            values[index] = length
        elif index == 0 and length > maxlen:
            values[index] = maxlen
            remaining -= maxlen
        elif remaining > maxlen:
            values[index] = maxlen
            remaining -= maxlen
        else:
            values[index] = remaining
    return values


def infer_item(model, item, maxlen: int, prompt_text: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one test item using the original VadCLIP reshape/masking behavior."""
    add_local_vadclip_source()
    from utils.tools import get_batch_mask

    visual = item[0].squeeze(0)
    length = int(item[2])
    if length < maxlen:
        visual = visual.unsqueeze(0)
    visual = visual.to(device)
    lengths = _chunk_lengths(length, maxlen).to(device)
    padding_mask = get_batch_mask(lengths, maxlen).to(device)
    with torch.no_grad():
        _text, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
    logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])[:length]
    logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])[:length]
    prob1 = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy().astype(np.float32)
    logits2_probability = logits2.softmax(dim=-1).cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2_probability[:, 0]).astype(np.float32)
    return prob1, prob2, logits2_probability


def metrics_from_predictions(
    probabilities1: list[np.ndarray],
    probabilities2: list[np.ndarray],
    logits2_probability: list[np.ndarray],
    gt: np.ndarray,
    gtsegments: np.ndarray,
    gtlabels: np.ndarray,
) -> XDMetrics:
    add_local_vadclip_source()
    from utils.xd_detectionMAP import getDetectionMAP

    prob1 = np.concatenate(probabilities1, axis=0)
    prob2 = np.concatenate(probabilities2, axis=0)
    repeated1, repeated2 = np.repeat(prob1, 16), np.repeat(prob2, 16)
    if len(repeated1) != len(gt):
        raise ValueError(f"XD frame-ground-truth length mismatch: predictions={len(repeated1)} gt={len(gt)}")
    dmap, ious = getDetectionMAP([np.repeat(item, 16, axis=0) for item in logits2_probability], gtsegments, gtlabels, excludeNormal=False)
    dmap_by_iou = {f"{float(iou):.1f}": float(value) for iou, value in zip(ious, dmap)}
    return XDMetrics(
        auc1=float(roc_auc_score(gt, repeated1)),
        ap1=float(average_precision_score(gt, repeated1)),
        auc2=float(roc_auc_score(gt, repeated2)),
        ap2=float(average_precision_score(gt, repeated2)),
        detection_map_by_iou=dmap_by_iou,
        detection_map_average=float(np.mean(dmap)),
    )


def evaluate_loader(model, loader, maxlen: int, gt: np.ndarray, gtsegments: np.ndarray, gtlabels: np.ndarray, device: torch.device, progress: str) -> XDMetrics:
    model.to(device).eval()
    prompt_text = list(XD_LABELS.values())
    probabilities1: list[np.ndarray] = []
    probabilities2: list[np.ndarray] = []
    logits2_probability: list[np.ndarray] = []
    for item in tqdm(loader, desc=progress, unit="video", leave=False):
        prob1, prob2, logits2 = infer_item(model, item, maxlen, prompt_text, device)
        probabilities1.append(prob1)
        probabilities2.append(prob2)
        logits2_probability.append(logits2)
    return metrics_from_predictions(probabilities1, probabilities2, logits2_probability, gt, gtsegments, gtlabels)

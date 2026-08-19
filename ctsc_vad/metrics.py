"""Exact VadCLIP frame AP/AUC and detection-mAP wrappers."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .baseline import add_vadclip_source


@dataclass
class VADMetrics:
    auc1: float
    ap1: float
    auc2: float
    ap2: float
    detection_map_by_iou: dict[str, float]
    detection_map_average: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_vadclip(probability1: list[np.ndarray], probability2: list[np.ndarray], class_probability: list[np.ndarray], gt: np.ndarray, gtsegments: np.ndarray, gtlabels: np.ndarray, dataset: str) -> VADMetrics:
    add_vadclip_source()
    if dataset == "xd":
        from utils.xd_detectionMAP import getDetectionMAP
    elif dataset == "ucf":
        from utils.ucf_detectionMAP import getDetectionMAP
    else:
        raise ValueError(f"unsupported dataset={dataset!r}")
    p1, p2 = np.repeat(np.concatenate(probability1), 16), np.repeat(np.concatenate(probability2), 16)
    if len(p1) != len(gt) or len(p2) != len(gt):
        raise ValueError(f"frame GT mismatch: predictions={len(p2)}, gt={len(gt)}")
    dmap, ious = getDetectionMAP([np.repeat(value, 16, axis=0) for value in class_probability], gtsegments, gtlabels, excludeNormal=False)
    return VADMetrics(
        auc1=float(roc_auc_score(gt, p1)), ap1=float(average_precision_score(gt, p1)),
        auc2=float(roc_auc_score(gt, p2)), ap2=float(average_precision_score(gt, p2)),
        detection_map_by_iou={f"{float(iou):.1f}": float(value) for iou, value in zip(ious, dmap)},
        detection_map_average=float(np.mean(dmap)),
    )


def print_vadclip_metrics(prefix: str, values: dict[str, object]) -> None:
    """Use the original VadCLIP metric labels verbatim, with a clear prefix."""
    print(f"{prefix} AUC1: {float(values['auc1']):.6f}  AP1: {float(values['ap1']):.6f}", flush=True)
    print(f"{prefix} AUC2: {float(values['auc2']):.6f}  AP2: {float(values['ap2']):.6f}", flush=True)
    for iou, score in dict(values["detection_map_by_iou"]).items():
        print(f"{prefix} mAP@{float(iou):.1f} ={float(score):.2f}%", flush=True)
    print(f"{prefix} average MAP: {float(values['detection_map_average']):.2f}", flush=True)

"""VadCLIP-compatible frame metrics for frozen-baseline and rectified scores."""
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
    ano_auc1: float | None
    ano_auc2: float | None
    detection_map_by_iou: dict[str, float]
    detection_map_average: float

    def to_dict(self) -> dict:
        return asdict(self)


def metrics_from_predictions(
    probabilities1: list[np.ndarray],
    probabilities2: list[np.ndarray],
    class_probabilities: list[np.ndarray],
    gt: np.ndarray,
    gtsegments: np.ndarray,
    gtlabels: np.ndarray,
    dataset: str,
    video_labels: list[str],
) -> VADMetrics:
    """Use the exact frame repeat and detection-mAP calls from local VadCLIP."""
    add_vadclip_source()
    if dataset == "xd":
        from utils.xd_detectionMAP import getDetectionMAP
    elif dataset == "ucf":
        from utils.ucf_detectionMAP import getDetectionMAP
    else:
        raise ValueError(f"unsupported dataset={dataset!r}")
    probability1 = np.concatenate(probabilities1)
    probability2 = np.concatenate(probabilities2)
    repeated1, repeated2 = np.repeat(probability1, 16), np.repeat(probability2, 16)
    if len(repeated1) != len(gt) or len(repeated2) != len(gt):
        raise ValueError(f"frame ground-truth mismatch: predictions={len(repeated1)}, gt={len(gt)}")
    ano_auc1 = ano_auc2 = None
    if dataset == "ucf":
        offset, only_anomaly_gt, only_anomaly_1, only_anomaly_2 = 0, [], [], []
        for label, score1, score2 in zip(video_labels, probabilities1, probabilities2):
            frame_count = len(score1) * 16
            segment_gt = gt[offset:offset + frame_count]
            if len(segment_gt) != frame_count:
                raise ValueError("UCF Ano-AUC video/frame alignment failed")
            if label != "Normal":
                only_anomaly_gt.append(segment_gt)
                only_anomaly_1.append(np.repeat(score1, 16))
                only_anomaly_2.append(np.repeat(score2, 16))
            offset += frame_count
        if offset != len(gt) or not only_anomaly_gt:
            raise ValueError("could not build UCF anomalous-video-only evaluation set")
        ano_auc1 = float(roc_auc_score(np.concatenate(only_anomaly_gt), np.concatenate(only_anomaly_1)))
        ano_auc2 = float(roc_auc_score(np.concatenate(only_anomaly_gt), np.concatenate(only_anomaly_2)))
    dmap, ious = getDetectionMAP(
        [np.repeat(item, 16, axis=0) for item in class_probabilities], gtsegments, gtlabels, excludeNormal=False
    )
    return VADMetrics(
        auc1=float(roc_auc_score(gt, repeated1)),
        ap1=float(average_precision_score(gt, repeated1)),
        auc2=float(roc_auc_score(gt, repeated2)),
        ap2=float(average_precision_score(gt, repeated2)),
        ano_auc1=ano_auc1,
        ano_auc2=ano_auc2,
        detection_map_by_iou={f"{float(iou):.1f}": float(value) for iou, value in zip(ious, dmap)},
        detection_map_average=float(np.mean(dmap)),
    )


def score_only_metrics(scores: list[np.ndarray], gt: np.ndarray) -> dict[str, float]:
    values = np.repeat(np.concatenate(scores), 16)
    if len(values) != len(gt):
        raise ValueError(f"frame ground-truth mismatch: predictions={len(values)}, gt={len(gt)}")
    return {"auc": float(roc_auc_score(gt, values)), "ap": float(average_precision_score(gt, values))}

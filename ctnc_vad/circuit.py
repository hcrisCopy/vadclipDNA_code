"""Sparse text-grounded normality circuit and explanation-guided rank projection."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .assets import asset_selected_width


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class NormalityCircuit(nn.Module):
    """A small trainable reader over a frozen, sparse CLIP hidden-state circuit."""

    def __init__(self, assets: dict, gate_initial_logit: float = 0.0) -> None:
        super().__init__()
        width = asset_selected_width(assets)
        self.width = width
        self.normal_index = 0
        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("state_mean", assets["state_mean"].float())
        self.register_buffer("state_std", assets["state_std"].float().clamp_min(1e-6))
        self.register_buffer("transition_mean", assets["transition_mean"].float())
        self.register_buffer("transition_std", assets["transition_std"].float().clamp_min(1e-6))
        self.register_buffer("ln_post_weight", assets["ln_post_weight"].float())
        self.register_buffer("ln_post_bias", assets["ln_post_bias"].float())
        self.register_buffer("visual_projection", assets["visual_projection"].float())
        self.register_buffer("text_features", F.normalize(assets["text_features"].float(), dim=-1))
        self.ln_post_eps = float(assets["ln_post_eps"])
        self.gate_logits = nn.Parameter(torch.full((width,), float(gate_initial_logit)))
        self.mix_logits = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(()))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _text_margin(self, last_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        post = F.layer_norm(
            last_hidden.float(), (last_hidden.shape[-1],), self.ln_post_weight, self.ln_post_bias, self.ln_post_eps
        )
        visual = F.normalize(post @ self.visual_projection, dim=-1, eps=1e-6)
        similarity = visual @ self.text_features.t()
        margin = similarity[..., 1:].max(dim=-1).values - similarity[..., self.normal_index]
        return margin.to(last_hidden.dtype), similarity.to(last_hidden.dtype)

    def forward(self, circuit: torch.Tensor, last_hidden: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        if circuit.ndim != 3 or circuit.shape[-1] != self.width:
            raise ValueError(f"expected [B,T,{self.width}] circuit, got {tuple(circuit.shape)}")
        if last_hidden.ndim != 3 or last_hidden.shape[:2] != circuit.shape[:2] or last_hidden.shape[-1] != 768:
            raise ValueError(f"expected [B,T,768] final hidden aligned to circuit, got {tuple(last_hidden.shape)}")
        times = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = times < lengths.unsqueeze(1)
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]
        transition_mean, transition_std = self.transition_mean[context], self.transition_std[context]
        state_z = (circuit - state_mean.unsqueeze(1)).abs() / state_std.unsqueeze(1)
        difference = torch.zeros_like(circuit)
        difference[:, 1:] = circuit[:, 1:] - circuit[:, :-1]
        transition_z = (difference - transition_mean.unsqueeze(1)).abs() / transition_std.unsqueeze(1)
        transition_z[:, 0] = 0.0
        gates = torch.sigmoid(self.gate_logits)
        denominator = gates.sum().clamp_min(1e-6)
        state_score = (state_z * gates).sum(dim=-1) / denominator
        transition_score = (transition_z * gates).sum(dim=-1) / denominator
        text_margin, text_similarity = self._text_margin(last_hidden)
        scales = F.softplus(self.mix_logits)
        logits = scales[0] * state_score + scales[1] * transition_score + scales[2] * text_margin + self.bias
        logits = logits.masked_fill(~mask, -20.0)
        return {
            "logits": logits,
            "score": torch.sigmoid(logits),
            "mask": mask,
            "context": context,
            "state_score": state_score,
            "transition_score": transition_score,
            "text_margin": text_margin,
            "text_similarity": text_similarity,
            "gates": gates,
            "dimension_state": state_z * gates,
            "dimension_transition": transition_z * gates,
        }


def mil_topk_mean(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for index in range(len(scores)):
        length = int(lengths[index])
        count = max(1, int(length / 16 + 1))
        values.append(scores[index, :length].topk(count).values.mean())
    return torch.stack(values)


def circuit_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, lengths: torch.Tensor, normal_weight: float, sparsity_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    pooled = mil_topk_mean(outputs["score"], lengths)
    bag = F.binary_cross_entropy(pooled, labels)
    normal = labels < 0.5
    if bool(normal.any()):
        frame_scores = outputs["score"][normal]
        frame_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(frame_scores[frame_mask], torch.zeros_like(frame_scores[frame_mask]))
    else:
        normal_loss = torch.zeros((), device=labels.device)
    sparse = outputs["gates"].mean()
    total = bag + float(normal_weight) * normal_loss + float(sparsity_weight) * sparse
    return total, {
        "bag": float(bag.detach()), "normal": float(normal_loss.detach()), "sparse": float(sparse.detach()),
        "pooled_normal": float(pooled[normal].mean().detach()) if bool(normal.any()) else float("nan"),
        "pooled_abnormal": float(pooled[~normal].mean().detach()) if bool((~normal).any()) else float("nan"),
    }


def rank_rectify(
    baseline_score: np.ndarray,
    circuit_score: np.ndarray,
    anchor_fraction: float,
    margin: float,
    strength: float,
    steps: int,
) -> np.ndarray:
    """One-dimensional projection enforcing circuit-supported anomaly/normal orderings.

    The frozen baseline supplies the initial ranking.  Only high circuit-score
    candidates and low circuit-score normal anchors are moved; no baseline
    feature or parameter is changed.
    """
    base = np.asarray(baseline_score, dtype=np.float32).reshape(-1)
    evidence = np.asarray(circuit_score, dtype=np.float32).reshape(-1)
    if len(base) != len(evidence) or len(base) == 0:
        raise ValueError("baseline and circuit scores must be non-empty and aligned")
    if len(base) == 1:
        return base.copy()
    count = min(max(1, int(math.ceil(len(base) * float(anchor_fraction)))), len(base) // 2)
    high = np.argpartition(evidence, -count)[-count:]
    low = np.argpartition(evidence, count - 1)[:count]
    result = base.copy()
    for _ in range(max(0, int(steps))):
        violation = np.maximum(0.0, float(margin) - (result[high, None] - result[None, low]))
        if not np.any(violation > 0):
            break
        result[high] += float(strength) * violation.mean(axis=1)
        result[low] -= float(strength) * violation.mean(axis=0)
        result = np.clip(result, 0.0, 1.0)
    return result.astype(np.float32)


def rectified_class_probabilities(original: np.ndarray, rectified_anomaly_score: np.ndarray) -> np.ndarray:
    """Preserve VadCLIP's abnormal-class conditional distribution for official dMAP."""
    probability = np.asarray(original, dtype=np.float32)
    score = np.asarray(rectified_anomaly_score, dtype=np.float32).reshape(-1)
    if probability.ndim != 2 or len(probability) != len(score) or probability.shape[1] < 2:
        raise ValueError("invalid logits2 probability contract")
    abnormal = probability[:, 1:]
    conditional = abnormal / np.maximum(abnormal.sum(axis=1, keepdims=True), 1e-6)
    result = np.empty_like(probability)
    result[:, 0] = 1.0 - score
    result[:, 1:] = conditional * score[:, None]
    return result

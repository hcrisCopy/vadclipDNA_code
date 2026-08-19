"""Class-specific sparse circuits over original CLIP hidden coordinates.

The reader has no hidden adapter.  For each anomaly text it averages signed,
context-normalized shifts of concrete ``(layer, dimension)`` coordinates using
learned direct channel weights.  The frozen baseline is combined afterwards as
an external probability expert, never modified internally.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .assets import selected_width


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_temporal_topk(values: torch.Tensor, lengths: torch.Tensor, fraction: float) -> torch.Tensor:
    """Standard weak temporal localization pooling: choose time, not channels."""
    pooled: list[torch.Tensor] = []
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        count = max(1, int(math.ceil(length * fraction)))
        pooled.append(values[index, :length].topk(count, dim=0).values.mean(dim=0))
    return torch.stack(pooled)


def probability_distribution(class_probability: torch.Tensor) -> torch.Tensor:
    """Turn independent anomaly-class probabilities into normal+class probabilities."""
    normal = torch.prod(1.0 - class_probability, dim=-1, keepdim=True).clamp_min(1e-6)
    values = torch.cat([normal, class_probability.clamp_min(1e-6)], dim=-1)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class SparseClassCircuit(nn.Module):
    """Direct text-conditioned channel circuits and class-wise PoE re-ranking.

    ``channel_logits[k,c]`` is directly auditable: after a sigmoid and a fixed
    CLIP text prior, it is the weight assigned to original hidden coordinate k
    for class c.  There is no learned projection, attention block, or MLP.
    """

    def __init__(self, assets: dict, gate_initial_logit: float = -2.0, fusion_initial_logit: float = -3.0) -> None:
        super().__init__()
        self.width = selected_width(assets)
        prompts = list(assets["prompts"])
        self.class_count, self.anomaly_count = len(prompts), len(prompts) - 1
        response = assets["semantic_response"].float()
        if response.shape != (self.width, self.anomaly_count):
            raise ValueError("discovery semantic responses do not match circuit shape")
        direction = torch.sign(response)
        self.register_buffer("text_direction", torch.where(direction == 0, torch.ones_like(direction), direction))
        strength = response.abs()
        # A text prior gates possible channels, but it does not determine the
        # result: weak labels learn the direct per-class gate afterwards.
        # Keep a small floor: a weak normal-only semantic probe should not
        # make an original channel mathematically impossible to recover from
        # standard video-level supervision.
        self.register_buffer("text_prior", 0.05 + 0.95 * strength / strength.amax(dim=0, keepdim=True).clamp_min(1e-6))
        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("context_mean", assets["context_mean"].float())
        self.register_buffer("context_std", assets["context_std"].float())
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_dimensions", assets["selected_dimensions"].long())
        self.register_buffer("semantic_response", response)

        self.channel_logits = nn.Parameter(torch.full((self.width, self.anomaly_count), float(gate_initial_logit)))
        self.class_scale_logits = nn.Parameter(torch.full((self.anomaly_count,), math.log(math.expm1(4.0))))
        self.class_bias = nn.Parameter(torch.full((self.anomaly_count,), -2.0))
        self.fusion_logits = nn.Parameter(torch.full((self.anomaly_count,), float(fusion_initial_logit)))

    def _context(self, final_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(final_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _class_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.sigmoid(self.channel_logits) * self.text_prior
        # Each class is a weighted average, so score scale is independent of
        # the number of retained candidate channels.
        normalized = raw / raw.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return raw, normalized

    def forward(self, circuit: torch.Tensor, final_hidden: torch.Tensor, baseline_probability: torch.Tensor, lengths: torch.Tensor, return_contributions: bool = False) -> dict[str, torch.Tensor]:
        if circuit.ndim != 3 or circuit.shape[-1] != self.width:
            raise ValueError(f"expected [B,T,{self.width}] raw channel circuit")
        if final_hidden.shape[:2] != circuit.shape[:2] or final_hidden.shape[-1] != 768:
            raise ValueError("final hidden states do not align with raw channel circuit")
        if baseline_probability.shape != (*circuit.shape[:2], self.class_count):
            raise ValueError("baseline class probabilities do not match prompt classes")
        time = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = time < lengths.unsqueeze(1)
        context = self._context(final_hidden, mask)
        mean, std = self.context_mean[context].unsqueeze(1), self.context_std[context].unsqueeze(1)
        zscore = ((circuit - mean) / std.clamp_min(1e-6)).masked_fill(~mask.unsqueeze(-1), 0.0)
        raw_weight, normalized_weight = self._class_weights()

        scores: list[torch.Tensor] = []
        contributions: list[torch.Tensor] = []
        for class_index in range(self.anomaly_count):
            directional_shift = F.relu(zscore * self.text_direction[:, class_index].view(1, 1, -1))
            contribution = directional_shift * normalized_weight[:, class_index].view(1, 1, -1)
            scores.append(contribution.sum(dim=-1))
            if return_contributions:
                contributions.append(contribution)
        evidence = torch.stack(scores, dim=-1)
        class_logit = evidence * F.softplus(self.class_scale_logits).view(1, 1, -1) + self.class_bias.view(1, 1, -1)
        class_probability = torch.sigmoid(class_logit).masked_fill(~mask.unsqueeze(-1), 0.0)
        circuit_distribution = probability_distribution(class_probability)

        # Product-of-experts fusion happens outside the frozen baseline.  It
        # gives each anomaly class a different external agreement strength.
        gamma = torch.sigmoid(self.fusion_logits)
        gamma_all = torch.cat([gamma.mean().reshape(1), gamma], dim=0).view(1, 1, -1)
        fused_log_probability = (1.0 - gamma_all) * torch.log(baseline_probability.clamp_min(1e-6)) + gamma_all * torch.log(circuit_distribution.clamp_min(1e-6))
        fused_probability = F.softmax(fused_log_probability, dim=-1).masked_fill(~mask.unsqueeze(-1), 0.0)
        result = {
            "mask": mask, "context": context, "zscore": zscore,
            "raw_channel_weight": raw_weight, "normalized_channel_weight": normalized_weight,
            "class_evidence": evidence.masked_fill(~mask.unsqueeze(-1), 0.0),
            "class_logit": class_logit.masked_fill(~mask.unsqueeze(-1), 0.0),
            "class_probability": class_probability,
            "circuit_distribution": circuit_distribution.masked_fill(~mask.unsqueeze(-1), 0.0),
            "fused_probability": fused_probability,
            "baseline_probability": baseline_probability,
            "fusion_gamma": gamma,
            "score": (1.0 - fused_probability[..., 0]).masked_fill(~mask, 0.0),
            "circuit_score": class_probability.max(dim=-1).values.masked_fill(~mask, 0.0),
        }
        if return_contributions:
            result["class_channel_contribution"] = torch.stack(contributions, dim=-1).masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        return result


def circuit_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, lengths: torch.Tensor, top_fraction: float, normal_weight: float, preserve_weight: float, entropy_weight: float, temporal_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    """Weak-label learning with sparse temporal pooling and direct weight entropy."""
    circuit_bag = F.binary_cross_entropy(masked_temporal_topk(outputs["class_probability"], lengths, top_fraction), targets)
    fused_bag = F.binary_cross_entropy(masked_temporal_topk(outputs["fused_probability"][..., 1:], lengths, top_fraction), targets)
    normal_bags = targets.sum(dim=-1) < 0.5
    if bool(normal_bags.any()):
        normal_values = outputs["class_probability"][normal_bags]
        normal_mask = outputs["mask"][normal_bags]
        normal = F.binary_cross_entropy(normal_values[normal_mask], torch.zeros_like(normal_values[normal_mask]))
    else:
        normal = torch.zeros((), device=targets.device)
    weight = outputs["normalized_channel_weight"].clamp_min(1e-8)
    entropy = -(weight * weight.log()).sum(dim=0).mean() / math.log(weight.shape[0])
    difference = (outputs["fused_probability"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    if outputs["class_logit"].shape[1] > 1:
        pair_mask = outputs["mask"][:, 1:] & outputs["mask"][:, :-1]
        delta = (outputs["class_logit"][:, 1:] - outputs["class_logit"][:, :-1]).abs().mean(dim=-1)
        temporal = (delta * pair_mask.to(delta.dtype)).sum() / pair_mask.sum().clamp_min(1)
    else:
        temporal = torch.zeros((), device=targets.device)
    total = fused_bag + circuit_bag + normal_weight * normal + preserve_weight * preserve + entropy_weight * entropy + temporal_weight * temporal
    return total, {"fused_bag": float(fused_bag.detach()), "circuit_bag": float(circuit_bag.detach()), "normal": float(normal.detach()), "preserve": float(preserve.detach()), "entropy": float(entropy.detach()), "temporal": float(temporal.detach())}

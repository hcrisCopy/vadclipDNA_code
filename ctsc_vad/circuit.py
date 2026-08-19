"""Sparse temporal circuits over concrete CLIP hidden-state coordinates.

CTSC never changes VadCLIP. It reads selected raw ``(layer, dimension)``
trajectories and gives each anomaly class a sparse, directly auditable mixture
of fixed temporal operators. The frozen baseline is only *promoted* at
segments for which the circuit provides a strong, class-specific and
temporally supported certificate; it is otherwise reproduced exactly.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .assets import selected_width
from .temporal import OPERATOR_NAMES, masked_centered_mean, masked_temporal_dynamics


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def masked_temporal_topk(values: torch.Tensor, lengths: torch.Tensor, fraction: float) -> torch.Tensor:
    """Standard weak temporal pooling: choose segments, never hidden channels."""
    pooled: list[torch.Tensor] = []
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        count = max(1, int(math.ceil(length * fraction)))
        pooled.append(values[index, :length].topk(count, dim=0).values.mean(dim=0))
    return torch.stack(pooled)


def masked_temporal_bottomk(values: torch.Tensor, lengths: torch.Tensor, fraction: float) -> torch.Tensor:
    """Low-evidence companion of top-k pooling, used only for positive bags."""
    pooled: list[torch.Tensor] = []
    for index, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        count = max(1, int(math.ceil(length * fraction)))
        pooled.append(values[index, :length].topk(count, dim=0, largest=False).values.mean(dim=0))
    return torch.stack(pooled)


def probability_distribution(class_probability: torch.Tensor) -> torch.Tensor:
    """Turn independent anomaly-class probabilities into normal+class probabilities."""
    normal = torch.prod(1.0 - class_probability, dim=-1, keepdim=True).clamp_min(1e-6)
    values = torch.cat([normal, class_probability.clamp_min(1e-6)], dim=-1)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class SparseClassCircuit(nn.Module):
    """Direct class-specific raw-channel temporal circuits.

    ``channel_operator_logits[k, operator, c]`` is a direct explanation
    weight. ``k`` maps to one original CLIP layer/dimension and ``operator``
    is a fixed temporal event. There is no learned projection, temporal
    encoder, attention, or adapter.
    """

    def __init__(self, assets: dict, gate_initial_logit: float = -2.0, fusion_initial_logit: float = -5.0) -> None:
        super().__init__()
        self.width = selected_width(assets)
        prompts = list(assets["prompts"])
        self.class_count, self.anomaly_count = len(prompts), len(prompts) - 1
        self.operator_count = len(OPERATOR_NAMES)
        response = assets["semantic_response"].float()
        if response.shape != (self.width, self.anomaly_count):
            raise ValueError("discovery semantic responses do not match circuit shape")
        direction = torch.sign(response)
        self.register_buffer("text_direction", torch.where(direction == 0, torch.ones_like(direction), direction))
        strength = response.abs()
        # Frozen CLIP text alignment is a prior, not a pseudo-label. A small
        # floor still lets weak video labels recover a valid raw coordinate.
        self.register_buffer("text_prior", 0.05 + 0.95 * strength / strength.amax(dim=0, keepdim=True).clamp_min(1e-6))
        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("context_mean", assets["context_mean"].float())
        self.register_buffer("context_std", assets["context_std"].float())
        self.register_buffer("context_temporal_mean", assets["context_temporal_mean"].float())
        self.register_buffer("context_temporal_std", assets["context_temporal_std"].float())
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_dimensions", assets["selected_dimensions"].long())
        self.register_buffer("semantic_response", response)
        self.short_window = int(assets["temporal_short_window"])
        self.long_window = int(assets["temporal_long_window"])
        self.persistence_window = int(assets["temporal_persistence_window"])

        self.channel_operator_logits = nn.Parameter(
            torch.full((self.width, self.operator_count, self.anomaly_count), float(gate_initial_logit))
        )
        self.class_scale_logits = nn.Parameter(torch.full((self.anomaly_count,), math.log(math.expm1(4.0))))
        self.class_bias = nn.Parameter(torch.full((self.anomaly_count,), -2.0))
        # Very small initially: a weak, new circuit cannot corrupt VadCLIP.
        self.fusion_logits = nn.Parameter(torch.full((self.anomaly_count,), float(fusion_initial_logit)))

    def _context(self, final_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(final_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _class_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.sigmoid(self.channel_operator_logits) * self.text_prior.unsqueeze(1)
        normalized = raw / raw.sum(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return raw, normalized, normalized.sum(dim=1)

    def _operator_values(
        self,
        zscore: torch.Tensor,
        velocity: torch.Tensor,
        short_long: torch.Tensor,
        temporal_mean: torch.Tensor,
        temporal_std: torch.Tensor,
        mask: torch.Tensor,
        class_index: int,
    ) -> tuple[torch.Tensor, ...]:
        """Return five non-negative, named values for one anomaly class."""
        direction = self.text_direction[:, class_index].view(1, 1, -1)
        normalized_velocity = (velocity - temporal_mean[..., 0].unsqueeze(1)) / temporal_std[..., 0].unsqueeze(1).clamp_min(1e-6)
        normalized_short_long = (short_long - temporal_mean[..., 1].unsqueeze(1)) / temporal_std[..., 1].unsqueeze(1).clamp_min(1e-6)
        level = F.relu(zscore * direction)
        values = (
            level,
            F.relu(normalized_velocity * direction),
            F.relu(-normalized_velocity * direction),
            F.relu(normalized_short_long * direction),
            masked_centered_mean(level, mask, self.persistence_window),
        )
        return tuple(value.masked_fill(~mask.unsqueeze(-1), 0.0) for value in values)

    def forward(
        self,
        circuit: torch.Tensor,
        final_hidden: torch.Tensor,
        baseline_probability: torch.Tensor,
        lengths: torch.Tensor,
        return_contributions: bool = False,
    ) -> dict[str, torch.Tensor]:
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
        velocity, short_long = masked_temporal_dynamics(zscore, mask, self.short_window, self.long_window)
        temporal_mean, temporal_std = self.context_temporal_mean[context], self.context_temporal_std[context]
        raw_weight, normalized_weight, normalized_channel_weight = self._class_weights()

        scores: list[torch.Tensor] = []
        operator_evidence: list[torch.Tensor] = []
        contributions: list[torch.Tensor] = []
        event_contributions: list[torch.Tensor] = []
        for class_index in range(self.anomaly_count):
            values = self._operator_values(zscore, velocity, short_long, temporal_mean, temporal_std, mask, class_index)
            weights = normalized_weight[:, :, class_index]
            per_operator: list[torch.Tensor] = []
            per_channel = torch.zeros_like(zscore) if return_contributions else None
            per_channel_event: list[torch.Tensor] = []
            for operator_index, value in enumerate(values):
                weighted = value * weights[:, operator_index].view(1, 1, -1)
                per_operator.append(weighted.sum(dim=-1))
                if return_contributions:
                    assert per_channel is not None
                    per_channel = per_channel + weighted
                    per_channel_event.append(weighted)
            scores.append(torch.stack(per_operator, dim=-1).sum(dim=-1))
            operator_evidence.append(torch.stack(per_operator, dim=-1))
            if return_contributions:
                assert per_channel is not None
                contributions.append(per_channel)
                event_contributions.append(torch.stack(per_channel_event, dim=-1))
        evidence = torch.stack(scores, dim=-1)
        class_operator_evidence = torch.stack(operator_evidence, dim=-1)
        class_logit = evidence * F.softplus(self.class_scale_logits).view(1, 1, -1) + self.class_bias.view(1, 1, -1)
        class_probability = torch.sigmoid(class_logit).masked_fill(~mask.unsqueeze(-1), 0.0)
        circuit_distribution = probability_distribution(class_probability)

        # The transparent certificate needs high evidence, a clear category,
        # and neighbouring support. It is not a learned verifier.
        local_support = masked_centered_mean(class_probability, mask, self.persistence_window)
        class_specificity = F.softmax(class_logit, dim=-1)
        certificate = (torch.sqrt((class_probability * local_support).clamp_min(0.0)) * class_specificity).masked_fill(~mask.unsqueeze(-1), 0.0)
        gamma = torch.sigmoid(self.fusion_logits)
        promotion = F.relu(class_logit) * certificate * gamma.view(1, 1, -1)

        # Exact baseline reconstruction plus one-way certified promotion.
        baseline_log_odds = torch.log(baseline_probability[..., 1:].clamp_min(1e-6)) - torch.log(baseline_probability[..., :1].clamp_min(1e-6))
        fused_log_odds = baseline_log_odds + promotion
        fused_probability = F.softmax(torch.cat([torch.zeros_like(fused_log_odds[..., :1]), fused_log_odds], dim=-1), dim=-1)
        fused_probability = fused_probability.masked_fill(~mask.unsqueeze(-1), 0.0)
        result = {
            "mask": mask,
            "context": context,
            "zscore": zscore,
            "raw_channel_operator_weight": raw_weight,
            "normalized_channel_operator_weight": normalized_weight,
            "normalized_channel_weight": normalized_channel_weight,
            "class_evidence": evidence.masked_fill(~mask.unsqueeze(-1), 0.0),
            "class_operator_evidence": class_operator_evidence.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0.0),
            "class_logit": class_logit.masked_fill(~mask.unsqueeze(-1), 0.0),
            "class_probability": class_probability,
            "circuit_distribution": circuit_distribution.masked_fill(~mask.unsqueeze(-1), 0.0),
            "certificate": certificate,
            "promotion": promotion.masked_fill(~mask.unsqueeze(-1), 0.0),
            "fused_probability": fused_probability,
            "baseline_probability": baseline_probability,
            "fusion_gamma": gamma,
            "score": (1.0 - fused_probability[..., 0]).masked_fill(~mask, 0.0),
            "circuit_score": class_probability.max(dim=-1).values.masked_fill(~mask, 0.0),
        }
        if return_contributions:
            result["class_channel_contribution"] = torch.stack(contributions, dim=-1).masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            result["class_channel_operator_contribution"] = torch.stack(event_contributions, dim=-1).masked_fill(~mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
        return result


def circuit_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    lengths: torch.Tensor,
    top_fraction: float,
    normal_weight: float,
    preserve_weight: float,
    entropy_weight: float,
    temporal_separation_weight: float,
    temporal_separation_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weak video-label learning with sparse channel and temporal selection."""
    circuit_top = masked_temporal_topk(outputs["class_probability"], lengths, top_fraction)
    fused_top = masked_temporal_topk(outputs["fused_probability"][..., 1:], lengths, top_fraction)
    circuit_bag = F.binary_cross_entropy(circuit_top, targets)
    fused_bag = F.binary_cross_entropy(fused_top, targets)
    normal_bags = targets.sum(dim=-1) < 0.5
    if bool(normal_bags.any()):
        normal_values = outputs["class_probability"][normal_bags]
        normal_mask = outputs["mask"][normal_bags]
        normal = F.binary_cross_entropy(normal_values[normal_mask], torch.zeros_like(normal_values[normal_mask]))
    else:
        normal = torch.zeros((), device=targets.device)
    weights = outputs["normalized_channel_operator_weight"].clamp_min(1e-8)
    entropy = -(weights * weights.log()).sum(dim=(0, 1)).mean() / math.log(weights.shape[0] * weights.shape[1])
    difference = (outputs["fused_probability"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    positive = targets > 0.5
    if bool(positive.any()):
        bottom = masked_temporal_bottomk(outputs["class_probability"], lengths, top_fraction)
        separation = F.relu(float(temporal_separation_margin) - (circuit_top - bottom))[positive].mean()
    else:
        separation = torch.zeros((), device=targets.device)
    total = fused_bag + circuit_bag + normal_weight * normal + preserve_weight * preserve + entropy_weight * entropy + temporal_separation_weight * separation
    return total, {
        "fused_bag": float(fused_bag.detach()), "circuit_bag": float(circuit_bag.detach()),
        "normal": float(normal.detach()), "preserve": float(preserve.detach()),
        "entropy": float(entropy.detach()), "separation": float(separation.detach()),
    }

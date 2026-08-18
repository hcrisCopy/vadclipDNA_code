"""Counterfactual temporal-normality circuits for a frozen VAD predictor.

The module never writes to VadCLIP. It reads two observable deviations of each
selected CLIP hidden coordinate from a normal-video memory: its state and its
frame-to-frame transition. Both are combined by an explicit
layer--dimension--text linear circuit and used only to re-rank frozen outputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .assets import asset_selected_width


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def inverse_softplus(value: float) -> float:
    """Stable scalar initialization for a positive ``softplus`` parameter."""
    if value <= 0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


class ChannelRankVerifier(nn.Module):
    """Auditable state-and-transition likelihood circuit over frozen VadCLIP.

    For anomaly text ``c`` and frame ``t``, the circuit computes

    ``E_c(t) = a_c sum_k w_state[k,c] z_state[t,k]
              + b_c sum_k w_transition[k,c] novelty[t,k]``.

    ``z_state`` and ``novelty`` are normalized against a context selected only
    from normal videos. Every weight is retained for audit; there is no hidden
    MLP or feature residual.
    """

    def __init__(
        self,
        assets: dict,
        gate_initial_logit: float = 0.0,
        verification_initial_logit: float = -1.5,
    ) -> None:
        super().__init__()
        width = asset_selected_width(assets)
        prompts = list(assets["prompts"])
        if len(prompts) < 2:
            raise ValueError("CTNC assets must contain a normal prompt and at least one anomaly prompt")
        affinity = assets["selected_text_affinity"].float()
        if affinity.shape != (width, len(prompts) - 1):
            raise ValueError(
                "selected_text_affinity does not match the prompt contract: "
                f"expected {(width, len(prompts) - 1)}, got {tuple(affinity.shape)}"
            )
        self.width = width
        self.layers = int(assets["hidden_layers"])
        self.class_count = len(prompts)
        self.anomaly_class_count = self.class_count - 1
        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("state_mean", assets["state_mean"].float())
        self.register_buffer("state_std", assets["state_std"].float().clamp_min(1e-6))
        self.register_buffer("transition_mean", assets["transition_mean"].float())
        self.register_buffer("transition_std", assets["transition_std"].float().clamp_min(1e-6))
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        self.register_buffer("selected_text_class", assets["selected_text_class"].long())

        # RMS normalization preserves the frozen signed semantic direction but
        # avoids an L1 denominator shrinking hundreds of atoms to zero.
        scale = affinity.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        self.register_buffer("text_affinity", affinity / scale)
        # State circuit: a gate keeps sparsity and an explicit, bounded local
        # correction can handle a domain-inverted text direction. The latter
        # is anchored to zero by the loss and exported for audit.
        self.state_gate_logits = nn.Parameter(
            torch.full((width, self.anomaly_class_count), float(gate_initial_logit))
        )
        self.state_correction_logits = nn.Parameter(torch.zeros(width, self.anomaly_class_count))
        # Transition circuit is a positive normality-surprise detector whose
        # text relevance comes from the same frozen visual-to-text lens.
        self.transition_gate_logits = nn.Parameter(
            torch.full((width, self.anomaly_class_count), float(gate_initial_logit))
        )
        self.state_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        self.transition_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        initial_rank_scale = max(0.05, float(torch.sigmoid(torch.tensor(verification_initial_logit))))
        self.rank_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(initial_rank_scale))
        )
        self.hidden_temperature_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        self.hidden_bias = nn.Parameter(torch.full((self.anomaly_class_count,), -2.0))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _channel_state(self, circuit: torch.Tensor, last_hidden: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]
        transition_mean, transition_std = self.transition_mean[context], self.transition_std[context]
        normalized_state = torch.tanh((circuit - state_mean.unsqueeze(1)) / (3.0 * state_std.unsqueeze(1)))
        delta = torch.cat((torch.zeros_like(circuit[:, :1]), circuit[:, 1:] - circuit[:, :-1]), dim=1)
        normalized_transition = torch.tanh(
            (delta - transition_mean.unsqueeze(1)) / (3.0 * transition_std.unsqueeze(1))
        )
        # Normal transitions have an expected bounded absolute response of
        # roughly .25. Removing this floor produces local novelty, not a
        # constant anomaly bias in every ordinary frame.
        transition_novelty = F.relu(normalized_transition.abs() - 0.25)
        transition_novelty[:, 0] = 0.0

        state_gates = torch.sigmoid(self.state_gate_logits)
        transition_gates = torch.sigmoid(self.transition_gate_logits)
        state_correction = 0.5 * torch.tanh(self.state_correction_logits)
        state_weights = self.text_affinity * state_gates + state_correction
        transition_weights = self.text_affinity.abs() * transition_gates
        state_denominator = state_weights.square().sum(dim=0).sqrt().clamp_min(1e-6)
        transition_denominator = transition_weights.square().sum(dim=0).sqrt().clamp_min(1e-6)
        state_evidence = torch.einsum("btk,kc->btc", normalized_state, state_weights)
        state_evidence = state_evidence / state_denominator.view(1, 1, -1)
        transition_evidence = torch.einsum("btk,kc->btc", transition_novelty, transition_weights)
        transition_evidence = transition_evidence / transition_denominator.view(1, 1, -1)
        state_scales = F.softplus(self.state_scale_logits)
        transition_scales = F.softplus(self.transition_scale_logits)
        class_evidence = (
            state_evidence * state_scales.view(1, 1, -1)
            + transition_evidence * transition_scales.view(1, 1, -1)
        )
        return {
            "context": context,
            "normalized_state": normalized_state,
            "normalized_transition": normalized_transition,
            "transition_novelty": transition_novelty,
            "state_gates": state_gates,
            "transition_gates": transition_gates,
            "state_correction": state_correction,
            "state_weights": state_weights,
            "transition_weights": transition_weights,
            "state_evidence": state_evidence,
            "transition_evidence": transition_evidence,
            "class_evidence": class_evidence,
            "state_scales": state_scales,
            "transition_scales": transition_scales,
        }

    def forward(
        self,
        circuit: torch.Tensor,
        last_hidden: torch.Tensor,
        baseline_probability: torch.Tensor,
        lengths: torch.Tensor,
        return_channel_contribution: bool = False,
    ) -> dict[str, torch.Tensor]:
        if circuit.ndim != 3 or circuit.shape[-1] != self.width:
            raise ValueError(f"expected [B,T,{self.width}] circuit, got {tuple(circuit.shape)}")
        if last_hidden.ndim != 3 or last_hidden.shape[:2] != circuit.shape[:2] or last_hidden.shape[-1] != 768:
            raise ValueError(f"expected [B,T,768] final hidden aligned to circuit, got {tuple(last_hidden.shape)}")
        expected = (*circuit.shape[:2], self.class_count)
        if tuple(baseline_probability.shape) != expected:
            raise ValueError(f"baseline probabilities must have shape {expected}, got {tuple(baseline_probability.shape)}")
        times = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = times < lengths.unsqueeze(1)
        atoms = self._channel_state(circuit, last_hidden, mask)
        rank_scales = F.softplus(self.rank_scale_logits)

        # Explicit likelihood-ratio fusion. It changes only prediction order;
        # no baseline feature, encoder, temporal module, or classifier changes.
        log_probability = baseline_probability.clamp_min(1e-6).log()
        hidden_factor = atoms["class_evidence"] * rank_scales.view(1, 1, -1)
        fused_logits = torch.cat((log_probability[..., :1], log_probability[..., 1:] + hidden_factor), dim=-1)
        verified_all = F.softmax(fused_logits, dim=-1).masked_fill(~mask.unsqueeze(-1), 0.0)
        verified = 1.0 - verified_all[..., 0]

        # The direct hidden probe is the same declared evidence under a
        # class-wise logistic calibration, not an independent black-box head.
        hidden_temperature = F.softplus(self.hidden_temperature_logits)
        hidden_probability = torch.sigmoid(
            atoms["class_evidence"] * hidden_temperature.view(1, 1, -1)
            + self.hidden_bias.view(1, 1, -1)
        )
        hidden_anomaly = (1.0 - (1.0 - hidden_probability).prod(dim=-1)).masked_fill(~mask, 0.0)
        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "rank_scales": rank_scales,
            "hidden_anomaly": hidden_anomaly,
            "hidden_probability": hidden_probability,
            # Aliases keep the evaluator contract stable.
            "gates": atoms["state_gates"].mean(dim=1),
            "class_gates": atoms["state_gates"],
            "class_gains": rank_scales,
            "verification_strength": rank_scales.mean(),
            **atoms,
        }
        if return_channel_contribution:
            result["state_channel_contribution"] = (
                atoms["normalized_state"].unsqueeze(-1)
                * atoms["state_weights"].view(1, 1, self.width, self.anomaly_class_count)
            )
            result["transition_channel_contribution"] = (
                atoms["transition_novelty"].unsqueeze(-1)
                * atoms["transition_weights"].view(1, 1, self.width, self.anomaly_class_count)
            )
            result["channel_contribution"] = (
                result["state_channel_contribution"] + result["transition_channel_contribution"]
            )
        return result


def mil_topk_mean(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Class-wise weak-video MIL pooling without baseline-generated labels."""
    values: list[torch.Tensor] = []
    for index in range(len(scores)):
        length = int(lengths[index])
        count = max(1, int(length / 16 + 1))
        values.append(scores[index, :length].topk(count, dim=0).values.mean(dim=0))
    return torch.stack(values)


def verifier_loss(
    outputs: dict[str, torch.Tensor],
    class_targets: torch.Tensor,
    lengths: torch.Tensor,
    normal_weight: float,
    preserve_weight: float,
    sparsity_weight: float,
    hidden_mil_weight: float,
    semantic_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weak-label circuit learning plus an explicit semantic-prior anchor."""
    pooled = mil_topk_mean(outputs["verified_all"][..., 1:], lengths)
    bag = F.binary_cross_entropy(pooled, class_targets)
    hidden_pooled = mil_topk_mean(outputs["hidden_probability"], lengths)
    hidden_bag = F.binary_cross_entropy(hidden_pooled, class_targets)
    normal = class_targets.sum(dim=-1) < 0.5
    if bool(normal.any()):
        normal_score = outputs["score"][normal]
        normal_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(normal_score[normal_mask], torch.zeros_like(normal_score[normal_mask]))
    else:
        normal_loss = torch.zeros((), device=class_targets.device)
    difference = (outputs["verified_all"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    sparse = 0.5 * (outputs["state_gates"].mean() + outputs["transition_gates"].mean())
    semantic_anchor = outputs["state_correction"].square().mean()
    total = (
        bag
        + float(hidden_mil_weight) * hidden_bag
        + float(normal_weight) * normal_loss
        + float(preserve_weight) * preserve
        + float(sparsity_weight) * sparse
        + float(semantic_anchor_weight) * semantic_anchor
    )
    return total, {
        "bag": float(bag.detach()),
        "hidden_bag": float(hidden_bag.detach()),
        "normal": float(normal_loss.detach()),
        "preserve": float(preserve.detach()),
        "sparse": float(sparse.detach()),
        "semantic_anchor": float(semantic_anchor.detach()),
        "pooled_normal": float(pooled[normal].mean().detach()) if bool(normal.any()) else float("nan"),
        "pooled_abnormal": float(pooled[~normal].mean().detach()) if bool((~normal).any()) else float("nan"),
    }

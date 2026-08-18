"""Text-conditioned, hidden-channel probability verifier for frozen VAD.

The verifier is deliberately a *prediction-space* sidecar: it never injects a
residual into VadCLIP features and never updates a VadCLIP weight. A sparse set
of CLIP coordinates is selected before reader training. Each coordinate has an
explicit signed affinity to every anomaly text prompt. At inference the reader
compares it with a scene-conditioned normal bank and uses the result as a
class-specific likelihood factor on the frozen VadCLIP probabilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .assets import asset_selected_width


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ChannelRankVerifier(nn.Module):
    """Interpretable class-wise product-of-experts over frozen VadCLIP output.

    ``baseline_probability`` contains the original VadCLIP prompt
    probabilities ``[normal, anomaly_1, ...]``. The only learned variables are
    one gate per discovered hidden coordinate and one non-negative gain per
    anomaly prompt. Thus every final-score change can be traced to a layer, a
    coordinate, its normal-state deviation, and a text prompt.
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
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        self.register_buffer("selected_text_class", assets["selected_text_class"].long())

        # Column normalization prevents one prompt from winning solely because
        # its offline semantic-lens coefficient has a larger numeric scale.
        # The signed affinity itself is retained as the explanation.
        scale = affinity.abs().mean(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("text_affinity", affinity / scale)
        self.gate_logits = nn.Parameter(torch.full((width,), float(gate_initial_logit)))
        self.class_gain_logits = nn.Parameter(torch.full((self.anomaly_class_count,), -0.5))
        self.verification_logit = nn.Parameter(torch.tensor(float(verification_initial_logit)))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _channel_state(
        self, circuit: torch.Tensor, last_hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]
        # The bounded standardized deviation is the primitive explanation. It
        # is comparable across layers/coordinates and robust to one outlier.
        normalized_state = torch.tanh((circuit - state_mean.unsqueeze(1)) / (3.0 * state_std.unsqueeze(1)))
        gates = torch.sigmoid(self.gate_logits)
        weighted_state = normalized_state * gates.view(1, 1, -1)
        denominator = (gates.unsqueeze(1) * self.text_affinity.abs()).sum(dim=0).clamp_min(1e-6)
        class_evidence = torch.einsum("btk,kc->btc", weighted_state, self.text_affinity) / denominator.view(1, 1, -1)
        return context, normalized_state, gates, class_evidence

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
        context, normalized_state, gates, class_evidence = self._channel_state(circuit, last_hidden, mask)

        # Product-of-experts fusion: log frozen probabilities + transparent
        # hidden likelihood factors. This is output re-ranking, not an injected
        # hidden residual or a changed baseline classifier.
        gains = F.softplus(self.class_gain_logits)
        strength = torch.sigmoid(self.verification_logit)
        hidden_factor = torch.tanh(class_evidence) * gains.view(1, 1, -1) * strength
        log_probability = baseline_probability.clamp_min(1e-6).log()
        fused_logits = torch.cat((log_probability[..., :1], log_probability[..., 1:] + hidden_factor), dim=-1)
        verified_all = F.softmax(fused_logits, dim=-1)
        verified_all = verified_all.masked_fill(~mask.unsqueeze(-1), 0.0)
        verified = 1.0 - verified_all[..., 0]

        # This reports whether the discovered circuit itself predicts anomaly,
        # independently of the frozen VAD output used for final re-ranking.
        hidden_only = F.softmax(torch.cat((torch.zeros_like(class_evidence[..., :1]), class_evidence), dim=-1), dim=-1)
        hidden_anomaly = (1.0 - hidden_only[..., 0]).masked_fill(~mask, 0.0)
        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "context": context,
            "normalized_state": normalized_state,
            "class_evidence": class_evidence,
            "hidden_anomaly": hidden_anomaly,
            "gates": gates,
            "class_gains": gains,
            "verification_strength": strength,
        }
        if return_channel_contribution:
            # Requested only by audit (one video at a time), avoiding an
            # unnecessary [B,T,K,C] training allocation.
            result["channel_contribution"] = (
                normalized_state.unsqueeze(-1)
                * gates.view(1, 1, -1, 1)
                * self.text_affinity.view(1, 1, self.width, self.anomaly_class_count)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Official video-category MIL supervision plus a frozen-output tether.

    ``class_targets`` comes from the dataset annotation (and is multi-label
    for XD combinations), never from a VadCLIP score threshold.
    """
    pooled = mil_topk_mean(outputs["verified_all"][..., 1:], lengths)
    bag = F.binary_cross_entropy(pooled, class_targets)
    normal = class_targets.sum(dim=-1) < 0.5
    if bool(normal.any()):
        normal_score = outputs["score"][normal]
        normal_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(normal_score[normal_mask], torch.zeros_like(normal_score[normal_mask]))
    else:
        normal_loss = torch.zeros((), device=class_targets.device)
    difference = (outputs["verified_all"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    sparse = outputs["gates"].mean()
    total = bag + float(normal_weight) * normal_loss + float(preserve_weight) * preserve + float(sparsity_weight) * sparse
    return total, {
        "bag": float(bag.detach()),
        "normal": float(normal_loss.detach()),
        "preserve": float(preserve.detach()),
        "sparse": float(sparse.detach()),
        "pooled_normal": float(pooled[normal].mean().detach()) if bool(normal.any()) else float("nan"),
        "pooled_abnormal": float(pooled[~normal].mean().detach()) if bool((~normal).any()) else float("nan"),
    }

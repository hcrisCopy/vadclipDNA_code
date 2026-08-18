"""Text-directed hidden-channel verifier for frozen VAD ranking.

The module never changes the baseline feature extractor or its weights. It
turns sparse signed CLIP hidden coordinates into interpretable evidence, then
chooses among keep, suppress, or promote actions. Keep is deliberately
favoured at initialization, so uncertain evidence preserves baseline ranking.
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
    """Small, auditable policy over frozen baseline scores and CLIP channels."""

    def __init__(self, assets: dict, gate_initial_logit: float = 0.0, keep_initial_logit: float = 5.0) -> None:
        super().__init__()
        width = asset_selected_width(assets)
        self.width = width
        self.layers = int(assets["hidden_layers"])
        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("state_mean", assets["state_mean"].float())
        self.register_buffer("state_std", assets["state_std"].float().clamp_min(1e-6))
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("text_direction", assets["selected_text_direction"].float())
        self.register_buffer("text_class", assets["selected_text_class"].long())
        self.register_buffer("text_affinity", assets["selected_text_affinity"].float())
        self.gate_logits = nn.Parameter(torch.full((width,), float(gate_initial_logit)))

        # The policy sees only four transparent quantities: signed channel
        # evidence, cross-layer agreement, frozen CLIP text margin and frozen
        # baseline logit. Every final action can therefore be audited.
        self.action_weight = nn.Parameter(torch.empty(3, 4))
        nn.init.normal_(self.action_weight, mean=0.0, std=0.02)
        self.action_bias = nn.Parameter(
            torch.tensor([float(keep_initial_logit), -float(keep_initial_logit), -float(keep_initial_logit)])
        )

        self.register_buffer("ln_post_weight", assets["ln_post_weight"].float())
        self.register_buffer("ln_post_bias", assets["ln_post_bias"].float())
        self.register_buffer("visual_projection", assets["visual_projection"].float())
        self.register_buffer("text_features", F.normalize(assets["text_features"].float(), dim=-1))
        self.ln_post_eps = float(assets["ln_post_eps"])

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _text_margin(self, last_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        post = F.layer_norm(
            last_hidden.float(), (last_hidden.shape[-1],), self.ln_post_weight, self.ln_post_bias, self.ln_post_eps
        )
        visual = F.normalize(post @ self.visual_projection, dim=-1, eps=1e-6)
        similarity = visual @ self.text_features.t()
        margin = similarity[..., 1:].max(dim=-1).values - similarity[..., 0]
        return margin.to(last_hidden.dtype), similarity.to(last_hidden.dtype)

    def _layer_evidence(self, channel_evidence: torch.Tensor) -> torch.Tensor:
        """Average signed evidence per original CLIP layer without black-box mixing."""
        batch, time, _width = channel_evidence.shape
        values = channel_evidence.new_zeros(batch, time, self.layers)
        counts = channel_evidence.new_zeros(self.layers)
        values.index_add_(2, self.selected_layers, channel_evidence)
        counts.index_add_(0, self.selected_layers, torch.ones_like(self.selected_layers, dtype=channel_evidence.dtype))
        return values / counts.clamp_min(1.0).view(1, 1, -1)

    def forward(
        self,
        circuit: torch.Tensor,
        last_hidden: torch.Tensor,
        baseline_score: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if circuit.ndim != 3 or circuit.shape[-1] != self.width:
            raise ValueError(f"expected [B,T,{self.width}] circuit, got {tuple(circuit.shape)}")
        if last_hidden.ndim != 3 or last_hidden.shape[:2] != circuit.shape[:2] or last_hidden.shape[-1] != 768:
            raise ValueError(f"expected [B,T,768] final hidden aligned to circuit, got {tuple(last_hidden.shape)}")
        if baseline_score.shape != circuit.shape[:2]:
            raise ValueError(f"baseline scores must have shape {tuple(circuit.shape[:2])}, got {tuple(baseline_score.shape)}")
        times = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = times < lengths.unsqueeze(1)
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]

        # Signs come from the frozen hidden-to-text route discovered for each
        # coordinate. Unlike the old |z| score, normal-looking shifts and
        # anomaly-text shifts cannot be treated as the same evidence.
        signed_z = ((circuit - state_mean.unsqueeze(1)) / state_std.unsqueeze(1)) * self.text_direction.view(1, 1, -1)
        signed_z = torch.tanh(signed_z / 3.0)
        gates = torch.sigmoid(self.gate_logits)
        channel_evidence = signed_z * gates.view(1, 1, -1)
        evidence = channel_evidence.sum(dim=-1) / gates.sum().clamp_min(1e-6)
        layer_evidence = self._layer_evidence(channel_evidence)
        agreement = F.relu(layer_evidence).mean(dim=-1) - F.relu(-layer_evidence).mean(dim=-1)

        text_margin, text_similarity = self._text_margin(last_hidden)
        base = baseline_score.clamp(1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(base)
        policy_features = torch.stack((evidence, agreement, text_margin * 10.0, base_logit), dim=-1)
        action_logits = F.linear(policy_features, self.action_weight, self.action_bias)
        action = F.softmax(action_logits, dim=-1)
        keep, suppress, promote = action.unbind(dim=-1)
        verified = keep * base + promote
        verified = verified.masked_fill(~mask, 0.0)

        return {
            "score": verified,
            "baseline_score": base,
            "mask": mask,
            "context": context,
            "signed_evidence": evidence,
            "layer_evidence": layer_evidence,
            "agreement": agreement,
            "text_margin": text_margin,
            "text_similarity": text_similarity,
            "gates": gates,
            "channel_evidence": channel_evidence,
            "action": action,
            "keep": keep,
            "suppress": suppress,
            "promote": promote,
        }


def mil_topk_mean(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for index in range(len(scores)):
        length = int(lengths[index])
        count = max(1, int(length / 16 + 1))
        values.append(scores[index, :length].topk(count).values.mean())
    return torch.stack(values)


def verifier_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    lengths: torch.Tensor,
    normal_weight: float,
    preserve_weight: float,
    sparsity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weak VAD MIL loss; frozen-baseline scores never define labels."""
    pooled = mil_topk_mean(outputs["score"], lengths)
    bag = F.binary_cross_entropy(pooled, labels)
    normal = labels < 0.5
    if bool(normal.any()):
        normal_scores = outputs["score"][normal]
        normal_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(normal_scores[normal_mask], torch.zeros_like(normal_scores[normal_mask]))
    else:
        normal_loss = torch.zeros((), device=labels.device)
    preserve = ((outputs["score"] - outputs["baseline_score"]).abs() * outputs["mask"].to(outputs["score"].dtype)).sum()
    preserve = preserve / outputs["mask"].sum().clamp_min(1)
    sparse = outputs["gates"].mean()
    total = bag + float(normal_weight) * normal_loss + float(preserve_weight) * preserve + float(sparsity_weight) * sparse
    return total, {
        "bag": float(bag.detach()), "normal": float(normal_loss.detach()), "preserve": float(preserve.detach()),
        "sparse": float(sparse.detach()), "pooled_normal": float(pooled[normal].mean().detach()) if bool(normal.any()) else float("nan"),
        "pooled_abnormal": float(pooled[~normal].mean().detach()) if bool((~normal).any()) else float("nan"),
    }

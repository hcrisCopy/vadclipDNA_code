"""Channel-gallery VAD reader for a frozen baseline.

The reader follows a simple, inspectable principle: discovery fixes a small
set of high-variance CLIP hidden channels that describe the normal manifold;
at test time a frame is anomalous when that *same original-channel vector* is
far from its nearest normal gallery vector and/or activates abnormal CLIP text.
It never writes into VadCLIP or builds a learned feature projection.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .assets import asset_selected_width


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid time steps for context retrieval."""
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


def load_verifier_state(model: "ChannelRankVerifier", state: dict[str, torch.Tensor]) -> None:
    """Reject checkpoints produced by an earlier reader definition."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "incompatible CTNC channel-gallery checkpoint; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


class ChannelRankVerifier(nn.Module):
    """Frozen-CLIP channel-gallery evidence with scalar, auditable fusion.

    Every visual score is a nearest-neighbour cosine distance in the fixed
    selected ``layer--dimension`` coordinates. Every semantic score is the
    exact frozen final CLIP image-to-text route. The only learned parameters
    are scalars that calibrate declared scores and their influence on frozen
    baseline odds; there is no MLP and no trainable embedding.
    """

    def __init__(
        self,
        assets: dict,
        gate_initial_logit: float = 0.0,
        verification_initial_logit: float = -3.0,
    ) -> None:
        super().__init__()
        del gate_initial_logit  # Kept in the CLI for backward-compatible commands.
        self.width = asset_selected_width(assets)
        self.layers = int(assets["hidden_layers"])
        prompts = list(assets["prompts"])
        if len(prompts) < 2:
            raise ValueError("CTNC assets need a normal prompt and at least one anomaly prompt")
        self.class_count = len(prompts)
        self.anomaly_class_count = self.class_count - 1

        self.register_buffer("context_centers", assets["context_centers"].float())
        # This is a sampled gallery of real, raw normal channel states. It is
        # not a learned memory and stays indexed by original channel IDs.
        self.register_buffer("normal_gallery", assets["normal_prototypes"].float())
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_dimensions", assets["selected_dimensions"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        self.register_buffer("selected_text_class", assets["selected_text_class"].long())

        self.register_buffer("ln_post_weight", assets["ln_post_weight"].float())
        self.register_buffer("ln_post_bias", assets["ln_post_bias"].float())
        self.register_buffer("visual_projection", assets["visual_projection"].float())
        self.register_buffer("text_features", assets["text_features"].float())
        self.ln_post_eps = float(assets["ln_post_eps"])
        if self.text_features.shape != (self.class_count, 512):
            raise ValueError("frozen text features do not match the prompt contract")

        # Positive scalars calibrate already-defined gallery/text evidence;
        # they do not create a learned VAD head or hidden feature.
        self.visual_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.semantic_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.lake_bias = nn.Parameter(torch.tensor(-1.0))
        self.text_temperature_logits = nn.Parameter(torch.tensor(inverse_softplus(4.0)))
        initial_fusion = max(0.01, float(torch.sigmoid(torch.tensor(verification_initial_logit))))
        self.fusion_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(initial_fusion)))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _gallery_probe(
        self, circuit: torch.Tensor, last_hidden: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Nearest-normal probing in the selected original channels."""
        context = self._context_indices(last_hidden, mask)
        gallery = self.normal_gallery[context]
        query_normalized = F.normalize(circuit, dim=-1, eps=1e-6)
        gallery_normalized = F.normalize(gallery, dim=-1, eps=1e-6)
        cosine = torch.einsum("btk,bpk->btp", query_normalized, gallery_normalized)
        nearest_similarity, nearest_index = cosine.max(dim=-1)
        nearest_gallery = torch.gather(
            gallery, 1, nearest_index.unsqueeze(-1).expand(-1, -1, self.width)
        )
        nearest_normalized = F.normalize(nearest_gallery, dim=-1, eps=1e-6)
        visual_score = ((1.0 - nearest_similarity) * 0.5).clamp(1e-6, 1.0 - 1e-6)
        # The additive coordinate decomposition is exported for explanation.
        # Its sum is the squared L2 distance in the same normalized channels.
        channel_deviation = (query_normalized - nearest_normalized).square()
        return {
            "context": context,
            "query_normalized": query_normalized,
            "nearest_normal_gallery_index": nearest_index,
            "nearest_normal_gallery": nearest_gallery,
            "nearest_normal_gallery_normalized": nearest_normalized,
            "nearest_normal_similarity": nearest_similarity,
            "visual_score": visual_score.masked_fill(~mask, 0.0),
            "channel_deviation": channel_deviation.masked_fill(~mask.unsqueeze(-1), 0.0),
        }

    def _text_probe(self, last_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact frozen CLIP ``ln_post + projection + text`` route."""
        mean = last_hidden.mean(dim=-1, keepdim=True)
        variance = (last_hidden - mean).square().mean(dim=-1, keepdim=True)
        post = (last_hidden - mean) / torch.sqrt(variance + self.ln_post_eps)
        post = post * self.ln_post_weight + self.ln_post_bias
        visual = F.normalize(post @ self.visual_projection, dim=-1, eps=1e-6)
        temperature = F.softplus(self.text_temperature_logits)
        text_logits = temperature * torch.einsum("btd,cd->btc", visual, self.text_features)
        text_probability = F.softmax(text_logits, dim=-1)
        semantic_score = (1.0 - text_probability[..., 0]).clamp(1e-6, 1.0 - 1e-6)
        return {
            "semantic_visual": visual,
            "text_logits": text_logits,
            "text_probability": text_probability,
            "semantic_score": semantic_score,
            "text_temperature": temperature.reshape(1),
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
            raise ValueError("final hidden states do not align with the selected circuit")
        if tuple(baseline_probability.shape) != (*circuit.shape[:2], self.class_count):
            raise ValueError("baseline probabilities do not match the CTNC prompt contract")
        times = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = times < lengths.unsqueeze(1)
        gallery = self._gallery_probe(circuit, last_hidden, mask)
        semantic = self._text_probe(last_hidden)

        visual_logit = torch.logit(gallery["visual_score"].clamp(1e-6, 1.0 - 1e-6))
        semantic_logit = torch.logit(semantic["semantic_score"])
        visual_scale = F.softplus(self.visual_scale_logits)
        semantic_scale = F.softplus(self.semantic_scale_logits)
        lake_logit = visual_scale * visual_logit + semantic_scale * semantic_logit + self.lake_bias
        lake_anomaly = torch.sigmoid(lake_logit).masked_fill(~mask, 0.0)

        baseline_anomaly = (1.0 - baseline_probability[..., 0]).clamp(1e-6, 1.0 - 1e-6)
        fusion_scale = F.softplus(self.fusion_scale_logits)
        verified = torch.sigmoid(
            torch.logit(baseline_anomaly) + fusion_scale * lake_logit
        ).masked_fill(~mask, 0.0)
        baseline_conditional = baseline_probability[..., 1:] / baseline_anomaly.unsqueeze(-1)
        verified_all = torch.cat(
            ((1.0 - verified).unsqueeze(-1), verified.unsqueeze(-1) * baseline_conditional), dim=-1
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        # Text alignment distributes scalar gallery evidence over anomaly
        # texts only for weak MIL supervision/audit. Baseline classes remain
        # intact in ``verified_all``.
        text_anomaly = semantic["text_probability"][..., 1:]
        text_anomaly = text_anomaly / text_anomaly.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        channel_probability = lake_anomaly.unsqueeze(-1) * text_anomaly
        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "channel_probability": channel_probability,
            "channel_anomaly": lake_anomaly,
            "hidden_anomaly": lake_anomaly,
            "visual_score": gallery["visual_score"],
            "semantic_score": semantic["semantic_score"].masked_fill(~mask, 0.0),
            "lake_logit": lake_logit.masked_fill(~mask, 0.0),
            "class_evidence": semantic["text_logits"][..., 1:] - semantic["text_logits"][..., :1],
            "fusion_scale": fusion_scale.reshape(1),
            "verification_strength": fusion_scale.reshape(1),
            "visual_scale": visual_scale.reshape(1),
            "semantic_scale": semantic_scale.reshape(1),
            **gallery,
            **semantic,
        }
        if return_channel_contribution:
            result["channel_contribution"] = gallery["channel_deviation"]
        return result


def mil_topk_mean(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Weak-video MIL pooling; labels are never generated from baseline scores."""
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibrate declared gallery/text evidence with original weak labels."""
    pooled = mil_topk_mean(outputs["verified_all"][..., 1:], lengths)
    bag = F.binary_cross_entropy(pooled, class_targets)
    witness_pooled = mil_topk_mean(outputs["channel_probability"], lengths)
    witness_bag = F.binary_cross_entropy(witness_pooled, class_targets)
    normal = class_targets.sum(dim=-1) < 0.5
    if bool(normal.any()):
        normal_score = outputs["channel_anomaly"][normal]
        normal_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(
            normal_score[normal_mask], torch.zeros_like(normal_score[normal_mask])
        )
    else:
        normal_loss = torch.zeros((), device=class_targets.device)
    difference = (outputs["verified_all"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    # No learned channel gates/dense head exist in this reader.
    sparsity = torch.zeros((), device=class_targets.device)
    total = (
        bag
        + float(hidden_mil_weight) * witness_bag
        + float(normal_weight) * normal_loss
        + float(preserve_weight) * preserve
        + float(sparsity_weight) * sparsity
    )
    return total, {
        "bag": float(bag.detach()),
        "witness_bag": float(witness_bag.detach()),
        "normal": float(normal_loss.detach()),
        "preserve": float(preserve.detach()),
        "sparsity": float(sparsity.detach()),
    }

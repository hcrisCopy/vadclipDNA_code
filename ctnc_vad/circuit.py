"""Text-conditioned original-channel reader for a frozen VAD baseline.

Discovery fixes a small set of original CLIP ``layer--dimension`` channels and
their frozen CLIP text associations.  The reader measures each channel against
its nearest normal-gallery state, keeps those channel-wise deviations separate,
and only then aggregates the strongest evidence for each anomaly text.  It
never writes into VadCLIP or builds a learned feature projection.
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
            "incompatible CTNC channel-text reader checkpoint; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


class ChannelRankVerifier(nn.Module):
    """Frozen-CLIP text-conditioned channel evidence plus local re-ranking.

    Every channel value remains an original CLIP coordinate.  Its contribution
    is its nearest-normal deviation multiplied by a fixed discovered
    channel-to-text affinity and a single learned, directly auditable gate.
    The only trainable parameters are one gate per selected coordinate and a
    few scalar calibrations; there is no adapter, MLP, or trainable embedding.
    """

    def __init__(
        self,
        assets: dict,
        gate_initial_logit: float = 0.0,
        verification_initial_logit: float = -3.0,
    ) -> None:
        super().__init__()
        self.width = asset_selected_width(assets)
        self.layers = int(assets["hidden_layers"])
        prompts = list(assets["prompts"])
        if len(prompts) < 2:
            raise ValueError("CTNC assets need a normal prompt and at least one anomaly prompt")
        self.class_count = len(prompts)
        self.anomaly_class_count = self.class_count - 1

        class_indices = assets["selected_text_class"].long()
        if bool((class_indices < 1).any()) or bool((class_indices >= self.class_count).any()):
            raise ValueError("selected_text_class must use prompt indices 1..anomaly_class_count")

        self.register_buffer("context_centers", assets["context_centers"].float())
        # This is a sampled gallery of real, raw normal channel states. It is
        # not a learned memory and stays indexed by original channel IDs.
        self.register_buffer("normal_gallery", assets["normal_prototypes"].float())
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_dimensions", assets["selected_dimensions"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        self.register_buffer("selected_text_class", class_indices)
        affinity = assets["selected_text_affinity"].float().abs()
        # Fixed channel-to-text links were discovered from frozen CLIP. They
        # are normalized but never replaced by a learned hidden projection.
        self.register_buffer(
            "channel_text_affinity",
            affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6),
        )
        direction = torch.sign(assets["selected_text_affinity"].float())
        self.register_buffer("channel_text_direction", torch.where(direction == 0, torch.ones_like(direction), direction))
        self.register_buffer(
            "dominant_channel_text_affinity",
            self.channel_text_affinity.gather(1, (self.selected_text_class - 1).unsqueeze(1)).squeeze(1),
        )

        self.register_buffer("ln_post_weight", assets["ln_post_weight"].float())
        self.register_buffer("ln_post_bias", assets["ln_post_bias"].float())
        self.register_buffer("visual_projection", assets["visual_projection"].float())
        self.register_buffer("text_features", assets["text_features"].float())
        self.ln_post_eps = float(assets["ln_post_eps"])
        if self.text_features.shape != (self.class_count, 512):
            raise ValueError("frozen text features do not match the prompt contract")

        # One gate per selected original CLIP channel. These gates control
        # only that channel's declared evidence and can be exported directly.
        self.channel_gate_logits = nn.Parameter(torch.full((self.width,), float(gate_initial_logit)))
        # Coordinate deviations are small because they are normalized CLIP
        # coordinates.  A nonzero fixed-scale start lets their gate receive a
        # useful weak-MIL gradient from the first epoch; the scalar remains
        # learned and is exported in every audit artifact.
        self.channel_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(32.0)))
        self.channel_bias = nn.Parameter(torch.tensor(-1.0))
        # Positive scalars calibrate declared evidence; no learned hidden
        # projection, adapter, or baseline weight is introduced.
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
        channel_delta = query_normalized - nearest_normalized
        channel_deviation = channel_delta.square()
        return {
            "context": context,
            "query_normalized": query_normalized,
            "nearest_normal_gallery_index": nearest_index,
            "nearest_normal_gallery": nearest_gallery,
            "nearest_normal_gallery_normalized": nearest_normalized,
            "nearest_normal_similarity": nearest_similarity,
            "visual_score": visual_score.masked_fill(~mask, 0.0),
            "channel_delta": channel_delta.masked_fill(~mask.unsqueeze(-1), 0.0),
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

        # Preserve every selected coordinate until final aggregation. A
        # contribution has an explicit meaning: original channel k, its signed
        # movement away from the nearest normal state, and its frozen CLIP
        # text direction/affinity.  Squared distance alone would lose this
        # text-relevant positive/negative direction.
        gates = torch.sigmoid(self.channel_gate_logits)
        dominant_directional_change = F.relu(
            gallery["channel_delta"] * self.selected_text_direction.view(1, 1, self.width)
        )
        gated_directional_change = dominant_directional_change * gates.view(1, 1, self.width)
        # This is the exact quantity shown in the audit/figure: one original
        # coordinate's text-directional normal departure, its learned gate,
        # and its discovered most related anomaly text.  It is not a latent
        # component or projection.
        channel_contribution = (
            gated_directional_change * self.dominant_channel_text_affinity.view(1, 1, self.width)
        )
        top_count = min(8, self.width)
        # Do not materialize [B,T,K,C]: on the standard VadCLIP batch size it
        # would take several GB.  Each class is independent, so this loop keeps
        # exactly the same top-k operation with O(B*T*K) peak memory.
        class_evidence_values: list[torch.Tensor] = []
        class_top_indices_values: list[torch.Tensor] = []
        for class_index in range(self.anomaly_class_count):
            class_directional_change = F.relu(
                gallery["channel_delta"] * self.channel_text_direction[:, class_index].view(1, 1, self.width)
            )
            class_contribution = (
                class_directional_change
                * gates.view(1, 1, self.width)
                * self.channel_text_affinity[:, class_index].view(1, 1, self.width)
            )
            top_values, top_indices = class_contribution.topk(top_count, dim=-1)
            class_evidence_values.append(top_values.mean(dim=-1))
            class_top_indices_values.append(top_indices)
        class_evidence = torch.stack(class_evidence_values, dim=-1)
        class_top_indices = torch.stack(class_top_indices_values, dim=-1)
        channel_evidence = class_evidence.max(dim=-1).values
        # Centering within each video targets local temporal ordering while
        # still retaining evidence from a sustained anomalous interval.
        centered_evidence = channel_evidence - masked_mean(channel_evidence, mask).unsqueeze(1)
        visual_scale = F.softplus(self.channel_scale_logits)
        channel_logit = visual_scale * centered_evidence + self.channel_bias
        channel_anomaly = torch.sigmoid(channel_logit).masked_fill(~mask, 0.0)
        semantic_logit = torch.logit(semantic["semantic_score"])
        semantic_scale = F.softplus(self.semantic_scale_logits)
        lake_logit = channel_logit + semantic_scale * semantic_logit + self.lake_bias
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

        # Weak MIL sees fixed text-conditioned channel evidence. The final
        # anomaly-class distribution stays frozen in ``verified_all``.
        channel_probability = torch.sigmoid(
            class_evidence + semantic["text_logits"][..., 1:] - semantic["text_logits"][..., :1]
        ).masked_fill(~mask.unsqueeze(-1), 0.0)
        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "channel_probability": channel_probability,
            "channel_anomaly": channel_anomaly,
            "hidden_anomaly": lake_anomaly,
            "visual_score": gallery["visual_score"],
            "semantic_score": semantic["semantic_score"].masked_fill(~mask, 0.0),
            "lake_logit": lake_logit.masked_fill(~mask, 0.0),
            "class_evidence": class_evidence,
            "channel_contribution": channel_contribution,
            "class_top_channel_index": class_top_indices,
            "channel_gates": gates,
            "channel_evidence": channel_evidence.masked_fill(~mask, 0.0),
            "centered_channel_evidence": centered_evidence.masked_fill(~mask, 0.0),
            "fusion_scale": fusion_scale.reshape(1),
            "verification_strength": fusion_scale.reshape(1),
            "visual_scale": visual_scale.reshape(1),
            "semantic_scale": semantic_scale.reshape(1),
            **gallery,
            **semantic,
        }
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
    """Calibrate declared channel/text evidence with original weak labels."""
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
    # Gates belong to concrete original channels, so this is both a sparse
    # regularizer and an interpretable channel-selection diagnostic.
    sparsity = outputs["channel_gates"].mean()
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

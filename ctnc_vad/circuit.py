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


def load_verifier_state(
    model: "ChannelRankVerifier", state: dict[str, torch.Tensor], *, reader_only: bool = False
) -> None:
    """Load a verifier checkpoint, including the one safe fusion migration.

    The move from class-wise semantic addition to a binary semantic odds
    scale changes only the final fusion parameter.  All frozen-path reader
    weights remain compatible, so retaining them makes comparisons fair and
    avoids retraining an already learned hidden/text circuit.
    """
    # A normal-memory discovery artifact contains data buffers (contexts,
    # prototypes, CLIP route), so warm-starting must never replace those
    # buffers with an earlier experiment's artifact.  Evaluation and exact
    # resume still use the full state by default.
    parameter_names = set(dict(model.named_parameters()))
    all_state_names = set(model.state_dict())
    source_unexpected = set(state) - all_state_names
    if reader_only:
        state = {name: value for name, value in state.items() if name in parameter_names}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "semantic_binary_scale_logits",
        "memory_temperature_logits",
        "memory_bias",
        "memory_binary_scale_logits",
        # Version-5 normal-video memory is an immutable discovery asset, not
        # a learned checkpoint value.  A version-4 reader can therefore seed
        # the compatible semantic/circuit parameters of a version-5 reader;
        # the newly discovered normal-video bank remains in the model.
        "normal_video_signatures",
        "normal_video_visual_prototypes",
    }
    allowed_unexpected = {"semantic_rank_scale_logits"}
    if reader_only:
        allowed_missing |= all_state_names - parameter_names
        # ``load_state_dict`` cannot report source entries filtered above.
        unexpected = list(set(unexpected) | source_unexpected)
    if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
        raise RuntimeError(
            "incompatible CTNC verifier checkpoint; "
            f"missing={sorted(set(missing) - allowed_missing)}, "
            f"unexpected={sorted(set(unexpected) - allowed_unexpected)}"
        )


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
        verification_initial_logit: float = -3.0,
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
        self.register_buffer("normal_prototypes", assets["normal_prototypes"].float())
        self.register_buffer("normal_video_signatures", assets["normal_video_signatures"].float())
        self.register_buffer("normal_video_visual_prototypes", assets["normal_video_visual_prototypes"].float())
        self.normal_video_neighbor_count = int(assets["normal_video_neighbor_count"])
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        self.register_buffer("selected_text_class", assets["selected_text_class"].long())
        self.register_buffer("ln_post_weight", assets["ln_post_weight"].float())
        self.register_buffer("ln_post_bias", assets["ln_post_bias"].float())
        self.register_buffer("visual_projection", assets["visual_projection"].float())
        self.ln_post_eps = float(assets["ln_post_eps"])
        text_features = assets["text_features"].float()
        if text_features.shape != (self.class_count, 512):
            raise ValueError(f"text_features must have shape [{self.class_count},512], got {tuple(text_features.shape)}")
        # Exact frozen CLIP visual-space directions: every learned adjustment
        # remains an explicit displacement from these text directions.
        self.register_buffer("semantic_text_prior", (text_features[1:] - text_features[:1]).t())

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
        # This all-channel semantic circuit does not learn a new visual
        # encoder. It reads the already-frozen final CLIP visual embedding and
        # learns one auditable correction vector per anomaly text.
        self.semantic_correction = nn.Parameter(torch.zeros(512, self.anomaly_class_count))
        self.semantic_bias = nn.Parameter(torch.full((self.anomaly_class_count,), -2.0))
        # A single, positive semantic odds scale is deliberately shared by
        # all anomaly classes. The frozen baseline retains responsibility for
        # class allocation; the hidden circuit answers only the question that
        # matters for localization: is this frame anomalous rather than normal?
        self.semantic_binary_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(0.10)))
        # Full-hidden normal-video memory calibration. It has no trainable
        # encoder or prototype: only a readable distance-to-probability map
        # and a conservative binary-odds scale.
        self.memory_temperature_logits = nn.Parameter(torch.tensor(inverse_softplus(10.0)))
        self.memory_bias = nn.Parameter(torch.tensor(-2.0))
        self.memory_binary_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(0.05)))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _channel_state(self, circuit: torch.Tensor, last_hidden: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]
        transition_mean, transition_std = self.transition_mean[context], self.transition_std[context]
        normalized_state = torch.tanh((circuit - state_mean.unsqueeze(1)) / (3.0 * state_std.unsqueeze(1)))
        # The nearest real normal state is the counterfactual reference.  It
        # fixes the central limitation of a context-wide mean: a frame should
        # not be called anomalous merely because it differs from another,
        # nevertheless normal, camera view or motion mode in the same scene.
        prototypes = self.normal_prototypes[context]  # [B, P, K]
        state_square = normalized_state.square().sum(dim=-1, keepdim=True)
        prototype_square = prototypes.square().sum(dim=-1).unsqueeze(1)
        cross = torch.einsum("btk,bpk->btp", normalized_state, prototypes)
        nearest_index = (state_square + prototype_square - 2.0 * cross).argmin(dim=-1)
        nearest_prototype = torch.gather(
            prototypes, 1, nearest_index.unsqueeze(-1).expand(-1, -1, self.width)
        )
        prototype_residual = normalized_state - nearest_prototype
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
        state_evidence = torch.einsum("btk,kc->btc", prototype_residual, state_weights)
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
            "nearest_prototype_index": nearest_index,
            "nearest_prototype": nearest_prototype,
            "prototype_residual": prototype_residual,
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

    def _semantic_circuit(self, last_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read the exact frozen CLIP final route with explicit text directions.

        This branch covers every final-layer hidden coordinate, complementing
        the sparse multi-layer normality circuit.  Its only learnable matrix
        is a per-text correction to the frozen visual--text direction, so a
        final semantic score can be expanded back to hidden coordinates.
        """
        mean = last_hidden.mean(dim=-1, keepdim=True)
        variance = (last_hidden - mean).square().mean(dim=-1, keepdim=True)
        ln_hidden = (last_hidden - mean) / torch.sqrt(variance + self.ln_post_eps)
        post_hidden = ln_hidden * self.ln_post_weight + self.ln_post_bias
        projected = post_hidden @ self.visual_projection
        projection_norm = projected.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        visual = projected / projection_norm
        semantic_weights = 4.0 * self.semantic_text_prior + self.semantic_correction
        semantic_logit = torch.einsum("btd,dc->btc", visual, semantic_weights) + self.semantic_bias.view(1, 1, -1)
        semantic_probability = torch.sigmoid(semantic_logit)
        # Before the final L2 normalization, this is the exact linear map from
        # a LayerNorm hidden coordinate to each text direction.  It is stored
        # for channel-level explanation rather than treated as a black box.
        hidden_text_weight = (self.ln_post_weight.unsqueeze(1) * self.visual_projection) @ semantic_weights
        return {
            "semantic_visual": visual,
            "semantic_logit": semantic_logit,
            "semantic_probability": semantic_probability,
            "semantic_weights": semantic_weights,
            "semantic_text_prior": self.semantic_text_prior,
            "semantic_hidden_weight": hidden_text_weight,
            "semantic_ln_hidden": ln_hidden,
            "semantic_projection_norm": projection_norm,
            "semantic_bias_contribution": (
                (self.ln_post_bias @ self.visual_projection @ semantic_weights).view(1, 1, -1)
                / projection_norm
            ),
        }

    def _normal_video_memory(
        self, last_hidden: torch.Tensor, visual: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compare each frame with matched real normal videos in frozen CLIP space."""
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        video_similarity = signature @ self.normal_video_signatures.t()
        neighbor_similarity, neighbor_video_index = video_similarity.topk(
            self.normal_video_neighbor_count, dim=-1
        )
        # [B, neighbors, prototype_frames, 512]; all entries originated in
        # training-set normal videos and are never optimized by this module.
        candidate_visual = self.normal_video_visual_prototypes[neighbor_video_index]
        frame_similarity = torch.einsum("btd,bkpd->btkp", visual, candidate_visual)
        flattened_similarity = frame_similarity.flatten(start_dim=2)
        nearest_similarity, nearest_flat_index = flattened_similarity.max(dim=-1)
        prototype_count = candidate_visual.shape[2]
        neighbor_slot = torch.div(nearest_flat_index, prototype_count, rounding_mode="floor")
        prototype_index = nearest_flat_index.remainder(prototype_count)
        nearest_video_index = torch.gather(neighbor_video_index, 1, neighbor_slot)
        distance = (1.0 - nearest_similarity).clamp_min(0.0)
        temperature = F.softplus(self.memory_temperature_logits)
        probability = torch.sigmoid(temperature * distance + self.memory_bias)
        return {
            "normal_video_neighbor_similarity": neighbor_similarity,
            "normal_video_neighbor_index": neighbor_video_index,
            "normal_memory_nearest_similarity": nearest_similarity,
            "normal_memory_nearest_video_index": nearest_video_index,
            "normal_memory_nearest_prototype_index": prototype_index,
            "normal_memory_distance": distance,
            "normal_memory_probability": probability,
            "normal_memory_temperature": temperature.reshape(1),
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
        semantic = self._semantic_circuit(last_hidden)
        memory = self._normal_video_memory(last_hidden, semantic["semantic_visual"], mask)
        rank_scales = F.softplus(self.rank_scale_logits)
        semantic_binary_scale = F.softplus(self.semantic_binary_scale_logits)
        memory_binary_scale = F.softplus(self.memory_binary_scale_logits)

        # Do not let a weak sparse normality atom globally promote anomalous
        # frames.  Instead, fuse the reliable, all-channel semantic evidence
        # at the *binary normal/anomaly odds* level, then preserve the frozen
        # baseline's conditional anomaly-class distribution.  This changes
        # localization/ranking while leaving every baseline weight untouched.
        baseline_anomaly = (1.0 - baseline_probability[..., 0]).clamp(1e-6, 1.0 - 1e-6)
        semantic_anomaly_unmasked = 1.0 - (1.0 - semantic["semantic_probability"]).prod(dim=-1)
        semantic_anomaly_clamped = semantic_anomaly_unmasked.clamp(1e-6, 1.0 - 1e-6)
        baseline_log_odds = torch.logit(baseline_anomaly)
        semantic_log_odds = torch.logit(semantic_anomaly_clamped)
        memory_log_odds = torch.logit(memory["normal_memory_probability"].clamp(1e-6, 1.0 - 1e-6))
        verified = torch.sigmoid(
            baseline_log_odds
            + semantic_binary_scale * semantic_log_odds
            + memory_binary_scale * memory_log_odds
        ).masked_fill(~mask, 0.0)
        conditional_class = baseline_probability[..., 1:] / baseline_anomaly.unsqueeze(-1)
        verified_all = torch.cat(
            ((1.0 - verified).unsqueeze(-1), verified.unsqueeze(-1) * conditional_class), dim=-1
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        # The direct hidden probe is the same declared evidence under a
        # class-wise logistic calibration, not an independent black-box head.
        hidden_temperature = F.softplus(self.hidden_temperature_logits)
        sparse_hidden_probability = torch.sigmoid(
            atoms["class_evidence"] * hidden_temperature.view(1, 1, -1)
            + self.hidden_bias.view(1, 1, -1)
        )
        semantic_anomaly = semantic_anomaly_unmasked
        sparse_hidden_anomaly = (
            1.0 - (1.0 - sparse_hidden_probability).prod(dim=-1)
        ).masked_fill(~mask, 0.0)
        hidden_probability = torch.maximum(sparse_hidden_probability, semantic["semantic_probability"])
        hidden_probability = torch.maximum(hidden_probability, memory["normal_memory_probability"].unsqueeze(-1))
        hidden_anomaly = (1.0 - (1.0 - hidden_probability).prod(dim=-1)).masked_fill(~mask, 0.0)
        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "rank_scales": rank_scales,
            "semantic_binary_scale": semantic_binary_scale.reshape(1),
            "memory_binary_scale": memory_binary_scale.reshape(1),
            "hidden_anomaly": hidden_anomaly,
            "hidden_probability": hidden_probability,
            "sparse_hidden_probability": sparse_hidden_probability,
            "sparse_hidden_anomaly": sparse_hidden_anomaly,
            "semantic_anomaly": semantic_anomaly.masked_fill(~mask, 0.0),
            "normal_memory_anomaly": memory["normal_memory_probability"].masked_fill(~mask, 0.0),
            # Aliases keep the evaluator contract stable.
            "gates": atoms["state_gates"].mean(dim=1),
            "class_gates": atoms["state_gates"],
            "class_gains": rank_scales,
            "verification_strength": semantic_binary_scale + memory_binary_scale,
            **atoms,
            **semantic,
            **memory,
        }
        if return_channel_contribution:
            result["state_channel_contribution"] = (
                atoms["prototype_residual"].unsqueeze(-1)
                * atoms["state_weights"].view(1, 1, self.width, self.anomaly_class_count)
            )
            result["transition_channel_contribution"] = (
                atoms["transition_novelty"].unsqueeze(-1)
                * atoms["transition_weights"].view(1, 1, self.width, self.anomaly_class_count)
            )
            result["channel_contribution"] = (
                result["state_channel_contribution"] + result["transition_channel_contribution"]
            )
            result["semantic_hidden_contribution"] = (
                semantic["semantic_ln_hidden"].unsqueeze(-1)
                * semantic["semantic_hidden_weight"].view(1, 1, 768, self.anomaly_class_count)
                / semantic["semantic_projection_norm"].unsqueeze(-1)
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
    # Keep the all-channel text circuit independently predictive.  Without
    # this term it can be ignored by a strong frozen baseline and never earn
    # a meaningful, auditable contribution to frame ranking.
    semantic_pooled = mil_topk_mean(outputs["semantic_probability"], lengths)
    semantic_bag = F.binary_cross_entropy(semantic_pooled, class_targets)
    video_anomaly = (class_targets.sum(dim=-1) > 0).to(outputs["score"].dtype)
    memory_pooled = mil_topk_mean(outputs["normal_memory_anomaly"].unsqueeze(-1), lengths).squeeze(-1)
    memory_bag = F.binary_cross_entropy(memory_pooled, video_anomaly)
    normal = class_targets.sum(dim=-1) < 0.5
    if bool(normal.any()):
        normal_score = outputs["score"][normal]
        normal_mask = outputs["mask"][normal]
        normal_loss = F.binary_cross_entropy(normal_score[normal_mask], torch.zeros_like(normal_score[normal_mask]))
        semantic_normal = outputs["semantic_probability"][normal]
        semantic_mask = outputs["mask"][normal].unsqueeze(-1).expand_as(semantic_normal)
        semantic_normal_loss = F.binary_cross_entropy(
            semantic_normal[semantic_mask], torch.zeros_like(semantic_normal[semantic_mask])
        )
        memory_normal = outputs["normal_memory_anomaly"][normal]
        memory_normal_mask = outputs["mask"][normal]
        memory_normal_loss = F.binary_cross_entropy(
            memory_normal[memory_normal_mask], torch.zeros_like(memory_normal[memory_normal_mask])
        )
    else:
        normal_loss = torch.zeros((), device=class_targets.device)
        semantic_normal_loss = torch.zeros((), device=class_targets.device)
        memory_normal_loss = torch.zeros((), device=class_targets.device)
    difference = (outputs["verified_all"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    sparse = 0.5 * (outputs["state_gates"].mean() + outputs["transition_gates"].mean())
    semantic_anchor = (
        outputs["state_correction"].square().mean()
        + outputs["semantic_weights"].sub(4.0 * outputs["semantic_text_prior"]).square().mean()
    )
    total = (
        bag
        + float(hidden_mil_weight) * hidden_bag
        + float(hidden_mil_weight) * semantic_bag
        + float(hidden_mil_weight) * memory_bag
        + float(normal_weight) * normal_loss
        + float(normal_weight) * semantic_normal_loss
        + float(normal_weight) * memory_normal_loss
        + float(preserve_weight) * preserve
        + float(sparsity_weight) * sparse
        + float(semantic_anchor_weight) * semantic_anchor
    )
    return total, {
        "bag": float(bag.detach()),
        "hidden_bag": float(hidden_bag.detach()),
        "semantic_bag": float(semantic_bag.detach()),
        "memory_bag": float(memory_bag.detach()),
        "normal": float(normal_loss.detach()),
        "semantic_normal": float(semantic_normal_loss.detach()),
        "memory_normal": float(memory_normal_loss.detach()),
        "preserve": float(preserve.detach()),
        "sparse": float(sparse.detach()),
        "semantic_anchor": float(semantic_anchor.detach()),
        "pooled_normal": float(pooled[normal].mean().detach()) if bool(normal.any()) else float("nan"),
        "pooled_abnormal": float(pooled[~normal].mean().detach()) if bool((~normal).any()) else float("nan"),
    }

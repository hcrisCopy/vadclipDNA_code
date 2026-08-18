"""Selected-hidden-channel witness reader for a frozen VAD predictor.

CTNC never writes a residual into VadCLIP. It first fixes a small dictionary
of ``layer--dimension--text`` witnesses in :mod:`ctnc_vad.discover`, then each
frame is scored only by the selected channels' deviation from normal states
and normal transitions. This keeps the ranking signal inspectable.
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
    if value <= 0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


def load_verifier_state(model: "ChannelRankVerifier", state: dict[str, torch.Tensor]) -> None:
    """Load only a checkpoint produced by this exact channel-witness reader."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "incompatible CTNC channel-witness checkpoint; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


class ChannelRankVerifier(nn.Module):
    """A direct, class-assigned witness circuit over selected CLIP channels.

    For selected channel ``k`` and its assigned anomaly text ``c``, the
    reader measures three non-negative, human-readable quantities:

    ``state_excess(t,k) = relu(|h(t,k)-nearest_normal(k)|-threshold_state(k))``
    ``motion_excess(t,k) = relu(|delta_h(t,k)-normal_delta(k)|-threshold_motion(k))``.
    ``subspace_excess(t,k) = relu(|h(t,k)-projection_normal_subspace(k)|-threshold_subspace(k))``.

    Learned gates only choose which *preselected* witnesses remain useful for
    that same text. There is no MLP, no learned visual embedding and no
    all-hidden fallback path.
    """

    def __init__(
        self,
        assets: dict,
        gate_initial_logit: float = 0.0,
        verification_initial_logit: float = -3.0,
    ) -> None:
        super().__init__()
        self.width = asset_selected_width(assets)
        prompts = list(assets["prompts"])
        if len(prompts) < 2:
            raise ValueError("CTNC assets need a normal prompt and at least one anomaly prompt")
        self.layers = int(assets["hidden_layers"])
        self.class_count = len(prompts)
        self.anomaly_class_count = self.class_count - 1

        self.register_buffer("context_centers", assets["context_centers"].float())
        self.register_buffer("state_mean", assets["state_mean"].float())
        self.register_buffer("state_std", assets["state_std"].float().clamp_min(1e-6))
        self.register_buffer("transition_mean", assets["transition_mean"].float())
        self.register_buffer("transition_std", assets["transition_std"].float().clamp_min(1e-6))
        self.register_buffer("normal_prototypes", assets["normal_prototypes"].float())
        self.register_buffer("normal_subspace_basis", assets["normal_subspace_basis"].float())
        self.subspace_rank = int(assets["subspace_rank"])
        if self.width % self.layers != 0:
            raise ValueError("selected witnesses must have the same count in every hidden layer")
        self.channels_per_layer = self.width // self.layers
        self.register_buffer("selected_layers", assets["selected_layers"].long())
        self.register_buffer("selected_dimensions", assets["selected_dimensions"].long())
        self.register_buffer("selected_text_direction", assets["selected_text_direction"].float())
        selected_class = assets.get("selected_by_text_class", assets["selected_text_class"]).long()
        if (
            selected_class.shape != (self.width,)
            or bool((selected_class < 1).any())
            or bool((selected_class > self.anomaly_class_count).any())
        ):
            raise ValueError("selected channel text assignments do not match the prompt classes")
        self.register_buffer("selected_text_class", selected_class)
        self.register_buffer(
            "channel_class_mask",
            F.one_hot(selected_class - 1, num_classes=self.anomaly_class_count).float(),
        )

        # A channel is assigned to exactly one text at discovery. Its other
        # class gates are permanently masked, so learned evidence still points
        # to a concrete layer/dimension/text witness.
        active = self.channel_class_mask.bool()
        initial_gate = torch.full((self.width, self.anomaly_class_count), -12.0)
        initial_gate[active] = float(gate_initial_logit)
        self.state_gate_logits = nn.Parameter(initial_gate)
        self.motion_gate_logits = nn.Parameter(initial_gate.clone())
        self.subspace_gate_logits = nn.Parameter(initial_gate.clone())
        self.state_threshold_logits = nn.Parameter(torch.full((self.width,), inverse_softplus(0.25)))
        self.motion_threshold_logits = nn.Parameter(torch.full((self.width,), inverse_softplus(0.20)))
        self.subspace_threshold_logits = nn.Parameter(torch.full((self.width,), inverse_softplus(0.20)))
        self.state_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        self.motion_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        self.subspace_scale_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(1.0))
        )
        self.channel_temperature_logits = nn.Parameter(
            torch.full((self.anomaly_class_count,), inverse_softplus(2.0))
        )
        # Six independent anomaly texts are unioned below. A conservative
        # individual prior prevents that union from promoting ordinary frames
        # before weak MIL has identified reliable channel witnesses.
        self.channel_bias = nn.Parameter(torch.full((self.anomaly_class_count,), -4.0))
        initial_fusion_scale = max(0.01, float(torch.sigmoid(torch.tensor(verification_initial_logit))))
        self.fusion_scale_logits = nn.Parameter(torch.tensor(inverse_softplus(initial_fusion_scale)))

    def _context_indices(self, last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        signature = F.normalize(masked_mean(last_hidden, mask), dim=-1, eps=1e-6)
        return (signature @ self.context_centers.t()).argmax(dim=-1)

    def _normal_deviations(
        self, circuit: torch.Tensor, last_hidden: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        context = self._context_indices(last_hidden, mask)
        state_mean, state_std = self.state_mean[context], self.state_std[context]
        transition_mean, transition_std = self.transition_mean[context], self.transition_std[context]
        standardized_state = (circuit - state_mean.unsqueeze(1)) / state_std.unsqueeze(1)
        normalized_state = torch.tanh(standardized_state / 3.0)

        # ``normal_subspace_basis`` is a fixed, context-specific SVD basis
        # of only the selected channels.  Reconstruction removes normal
        # channel co-variation; each residual component is still the original
        # layer/dimension coordinate that can be shown in an audit.
        layer_state = standardized_state.view(
            circuit.shape[0], circuit.shape[1], self.layers, self.channels_per_layer
        )
        basis = self.normal_subspace_basis[context]
        coefficient = torch.einsum("btlp,blpr->btlr", layer_state, basis)
        reconstruction = torch.einsum("btlr,blpr->btlp", coefficient, basis)
        subspace_residual = (layer_state - reconstruction).reshape_as(circuit)

        # The normal reference is still expressed in the same selected
        # channels. It handles multiple normal camera/pose modes without
        # introducing an unselected visual feature.
        prototypes = self.normal_prototypes[context]
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
        motion_deviation = normalized_transition.abs()
        motion_deviation[:, 0] = 0.0
        return {
            "context": context,
            "normalized_state": normalized_state,
            "nearest_prototype_index": nearest_index,
            "nearest_prototype": nearest_prototype,
            "prototype_residual": prototype_residual,
            "state_deviation": prototype_residual.abs(),
            "subspace_residual": subspace_residual,
            "subspace_deviation": torch.tanh(subspace_residual.abs() / 3.0),
            "normalized_transition": normalized_transition,
            "motion_deviation": motion_deviation,
        }

    def _channel_evidence(self, deviations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state_threshold = F.softplus(self.state_threshold_logits)
        motion_threshold = F.softplus(self.motion_threshold_logits)
        subspace_threshold = F.softplus(self.subspace_threshold_logits)
        state_excess = F.relu(deviations["state_deviation"] - state_threshold.view(1, 1, -1))
        motion_excess = F.relu(deviations["motion_deviation"] - motion_threshold.view(1, 1, -1))
        subspace_excess = F.relu(deviations["subspace_deviation"] - subspace_threshold.view(1, 1, -1))

        state_gates = torch.sigmoid(self.state_gate_logits) * self.channel_class_mask
        motion_gates = torch.sigmoid(self.motion_gate_logits) * self.channel_class_mask
        subspace_gates = torch.sigmoid(self.subspace_gate_logits) * self.channel_class_mask
        state_weights = state_gates / state_gates.sum(dim=0, keepdim=True).clamp_min(1e-6)
        motion_weights = motion_gates / motion_gates.sum(dim=0, keepdim=True).clamp_min(1e-6)
        subspace_weights = subspace_gates / subspace_gates.sum(dim=0, keepdim=True).clamp_min(1e-6)
        state_evidence = torch.einsum("btk,kc->btc", state_excess, state_weights)
        motion_evidence = torch.einsum("btk,kc->btc", motion_excess, motion_weights)
        subspace_evidence = torch.einsum("btk,kc->btc", subspace_excess, subspace_weights)
        class_evidence = (
            state_evidence * F.softplus(self.state_scale_logits).view(1, 1, -1)
            + motion_evidence * F.softplus(self.motion_scale_logits).view(1, 1, -1)
            + subspace_evidence * F.softplus(self.subspace_scale_logits).view(1, 1, -1)
        )
        return {
            "state_threshold": state_threshold,
            "motion_threshold": motion_threshold,
            "subspace_threshold": subspace_threshold,
            "state_excess": state_excess,
            "motion_excess": motion_excess,
            "subspace_excess": subspace_excess,
            "state_gates": state_gates,
            "motion_gates": motion_gates,
            "subspace_gates": subspace_gates,
            "state_weights": state_weights,
            "motion_weights": motion_weights,
            "subspace_weights": subspace_weights,
            "state_evidence": state_evidence,
            "motion_evidence": motion_evidence,
            "subspace_evidence": subspace_evidence,
            "class_evidence": class_evidence,
            "state_scales": F.softplus(self.state_scale_logits),
            "motion_scales": F.softplus(self.motion_scale_logits),
            "subspace_scales": F.softplus(self.subspace_scale_logits),
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
        if tuple(baseline_probability.shape) != (*circuit.shape[:2], self.class_count):
            raise ValueError("baseline probabilities do not match the CTNC prompt contract")
        times = torch.arange(circuit.shape[1], device=circuit.device).unsqueeze(0)
        mask = times < lengths.unsqueeze(1)
        deviations = self._normal_deviations(circuit, last_hidden, mask)
        atoms = self._channel_evidence(deviations)

        channel_probability = torch.sigmoid(
            atoms["class_evidence"] * F.softplus(self.channel_temperature_logits).view(1, 1, -1)
            + self.channel_bias.view(1, 1, -1)
        )
        channel_anomaly = 1.0 - (1.0 - channel_probability).prod(dim=-1)
        baseline_anomaly = (1.0 - baseline_probability[..., 0]).clamp(1e-6, 1.0 - 1e-6)
        fusion_scale = F.softplus(self.fusion_scale_logits)
        verified = torch.sigmoid(
            torch.logit(baseline_anomaly)
            + fusion_scale * torch.logit(channel_anomaly.clamp(1e-6, 1.0 - 1e-6))
        ).masked_fill(~mask, 0.0)
        conditional_class = baseline_probability[..., 1:] / baseline_anomaly.unsqueeze(-1)
        verified_all = torch.cat(
            ((1.0 - verified).unsqueeze(-1), verified.unsqueeze(-1) * conditional_class), dim=-1
        ).masked_fill(~mask.unsqueeze(-1), 0.0)

        result = {
            "score": verified,
            "verified_all": verified_all,
            "baseline_probability": baseline_probability,
            "mask": mask,
            "channel_probability": channel_probability,
            "channel_anomaly": channel_anomaly.masked_fill(~mask, 0.0),
            "hidden_anomaly": channel_anomaly.masked_fill(~mask, 0.0),
            "class_evidence": atoms["class_evidence"],
            "class_gates": atoms["state_gates"],
            "fusion_scale": fusion_scale.reshape(1),
            "verification_strength": fusion_scale.reshape(1),
            **deviations,
            **atoms,
        }
        if return_channel_contribution:
            result["state_channel_contribution"] = (
                atoms["state_excess"].unsqueeze(-1) * atoms["state_weights"].view(1, 1, self.width, -1)
            )
            result["motion_channel_contribution"] = (
                atoms["motion_excess"].unsqueeze(-1) * atoms["motion_weights"].view(1, 1, self.width, -1)
            )
            result["subspace_channel_contribution"] = (
                atoms["subspace_excess"].unsqueeze(-1) * atoms["subspace_weights"].view(1, 1, self.width, -1)
            )
            result["channel_contribution"] = (
                result["state_channel_contribution"]
                + result["motion_channel_contribution"]
                + result["subspace_channel_contribution"]
            )
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
    """Train direct channel witnesses using only original video-level labels."""
    pooled = mil_topk_mean(outputs["verified_all"][..., 1:], lengths)
    bag = F.binary_cross_entropy(pooled, class_targets)
    witness_pooled = mil_topk_mean(outputs["channel_probability"], lengths)
    witness_bag = F.binary_cross_entropy(witness_pooled, class_targets)
    normal = class_targets.sum(dim=-1) < 0.5
    if bool(normal.any()):
        normal_probability = outputs["channel_probability"][normal]
        normal_mask = outputs["mask"][normal].unsqueeze(-1).expand_as(normal_probability)
        normal_loss = F.binary_cross_entropy(
            normal_probability[normal_mask], torch.zeros_like(normal_probability[normal_mask])
        )
    else:
        normal_loss = torch.zeros((), device=class_targets.device)
    difference = (outputs["verified_all"] - outputs["baseline_probability"]).abs().mean(dim=-1)
    preserve = (difference * outputs["mask"].to(difference.dtype)).sum() / outputs["mask"].sum().clamp_min(1)
    sparsity = (
        outputs["state_gates"].mean()
        + outputs["motion_gates"].mean()
        + outputs["subspace_gates"].mean()
    ) / 3.0
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

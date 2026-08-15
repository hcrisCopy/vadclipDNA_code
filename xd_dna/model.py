"""Residual selected-neuron injection before the unchanged local VadCLIP model."""
from __future__ import annotations

import torch
from torch import nn

from .vadclip import build_baseline


class DNAResidualVadCLIP(nn.Module):
    """Map selected neurons to a zero-initialized residual over 512D CLIP input.

    The local VadCLIP baseline is never edited. Its 512D interface, losses and
    evaluation protocol remain intact; only this explicit DNA branch learns.
    """

    def __init__(
        self,
        options,
        baseline_checkpoint: str,
        device: torch.device,
        neuron_width: int,
        hidden_dim: int = 1024,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if neuron_width <= 0 or hidden_dim <= 0 or depth < 1:
            raise ValueError("neuron width, residual hidden dimension and depth must be positive")
        if int(options.visual_width) != 512:
            raise ValueError("the local VadCLIP XD interface must remain 512D")
        self.neuron_width, self.clip_dim = int(neuron_width), 512
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        dimensions = [self.neuron_width] + [int(hidden_dim)] * (int(depth) - 1) + [self.clip_dim]
        modules: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            modules.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                modules.append(nn.GELU())
        self.neuron_to_clip = nn.Sequential(*modules)
        final_linear = next(module for module in reversed(self.neuron_to_clip) if isinstance(module, nn.Linear))
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.base = build_baseline(options, baseline_checkpoint, device)
        self.base.requires_grad_(False)

    @property
    def residual_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        expected = self.neuron_width + self.clip_dim
        if visual.ndim != 3 or visual.shape[-1] != expected:
            raise ValueError(f"expected [B,T,{expected}] [DNA-neurons|CLIP], got {tuple(visual.shape)}")
        neurons, clip = visual[..., :self.neuron_width], visual[..., self.neuron_width:]
        correction = self.neuron_to_clip(self.neuron_norm(neurons))
        enhanced = clip + self.residual_gate.to(dtype=clip.dtype) * correction.to(dtype=clip.dtype)
        return self.base(enhanced, padding_mask, text, lengths)

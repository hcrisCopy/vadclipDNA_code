"""NeuVAD-Lens residual wrapper over the unchanged local VadCLIP baseline."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from xd_dna.vadclip import build_baseline

from .lens import TextProjectionLens


def residual_mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int = 512) -> nn.Sequential:
    """Build the GELU MLP style used by the established DNA residual branch."""
    if input_dim <= 0 or hidden_dim <= 0 or depth < 1:
        raise ValueError("residual MLP dimensions and depth must be positive")
    dimensions = [int(input_dim)] + [int(hidden_dim)] * (int(depth) - 1) + [int(output_dim)]
    modules: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        modules.append(nn.Linear(in_dim, out_dim))
        if index < len(dimensions) - 2:
            modules.append(nn.GELU())
    result = nn.Sequential(*modules)
    final_linear = next(module for module in reversed(result) if isinstance(module, nn.Linear))
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)
    return result


class NeuVADLensVadCLIP(nn.Module):
    """Inject parallel DNA and complete text-lens residuals before frozen VadCLIP."""

    def __init__(
        self,
        options,
        baseline_checkpoint: str,
        lens_assets: str | Path,
        device: torch.device,
        neuron_width: int,
        dna_hidden_dim: int = 1024,
        dna_depth: int = 3,
        text_hidden_dim: int = 512,
        text_depth: int = 2,
        text_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if int(options.visual_width) != 512:
            raise ValueError("the local VadCLIP visual interface must remain 512D")
        self.neuron_width, self.clip_dim, self.last_hidden_dim = int(neuron_width), 512, 768
        self.text_temperature = float(text_temperature)
        self.lens = TextProjectionLens(lens_assets)
        if self.lens.abnormal_class_count != int(options.classes_num) - 1:
            raise ValueError(
                f"lens has {self.lens.abnormal_class_count} abnormal classes but VadCLIP expects "
                f"{int(options.classes_num) - 1}"
            )
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        self.dna_to_clip = residual_mlp(self.neuron_width, dna_hidden_dim, dna_depth)
        self.text_to_clip = residual_mlp(self.last_hidden_dim + 1, text_hidden_dim, text_depth)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.base = build_baseline(options, baseline_checkpoint, device)
        self.base.requires_grad_(False)

    @property
    def residual_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    @property
    def input_width(self) -> int:
        return self.neuron_width + self.clip_dim + self.last_hidden_dim

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        if visual.ndim != 3 or visual.shape[-1] != self.input_width:
            raise ValueError(
                f"expected [B,T,{self.input_width}] [DNA-neurons|CLIP|last-hidden], got {tuple(visual.shape)}"
            )
        neuron_stop = self.neuron_width
        clip_stop = neuron_stop + self.clip_dim
        neurons = visual[..., :neuron_stop]
        clip_feature = visual[..., neuron_stop:clip_stop]
        last_hidden = visual[..., clip_stop:]
        dna_correction = self.dna_to_clip(self.neuron_norm(neurons))
        text_evidence, _route, normal_distance, _contributions = self.lens(last_hidden, self.text_temperature)
        text_input = torch.cat([text_evidence, normal_distance.unsqueeze(-1)], dim=-1)
        text_correction = self.text_to_clip(text_input)
        correction = dna_correction + text_correction
        enhanced = clip_feature + self.residual_gate.to(dtype=clip_feature.dtype) * correction.to(dtype=clip_feature.dtype)
        return self.base(enhanced, padding_mask, text, lengths)

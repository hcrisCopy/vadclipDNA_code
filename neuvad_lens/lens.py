"""Frozen text-projection lens shared by training, evaluation and auditing."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REQUIRED_LENS_KEYS = {
    "schema_version",
    "dataset",
    "clip_model",
    "last_layer_index",
    "ln_weight",
    "ln_bias",
    "ln_eps",
    "visual_projection",
    "normal_text",
    "abnormal_text",
    "normal_class_name",
    "abnormal_class_names",
    "text_directions",
    "normal_mean",
    "normal_std",
    "normal_indices",
    "normal_prototypes",
}


def load_lens_asset(path: str | Path) -> dict:
    """Load a CPU lens asset and reject incomplete artifacts early."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"lens asset is absent: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: lens asset must be a dictionary")
    missing = REQUIRED_LENS_KEYS - set(payload)
    if missing:
        raise ValueError(f"{source}: lens asset is missing {sorted(missing)}")
    if int(payload["schema_version"]) != 1:
        raise ValueError(f"{source}: unsupported lens schema {payload['schema_version']!r}")
    return payload


class TextProjectionLens(nn.Module):
    """Map raw final CLS states to complete, soft text-conditioned evidence.

    The lens deliberately keeps all 768 final-layer dimensions. Top-k is used
    only by the audit script for presentation, never to remove training data.
    Buffers originate from frozen OpenAI CLIP and are never optimized.
    """

    def __init__(self, asset_path: str | Path) -> None:
        super().__init__()
        asset = load_lens_asset(asset_path)
        self.dataset = str(asset["dataset"])
        self.last_layer_index = int(asset["last_layer_index"])
        self.class_names = [str(value) for value in asset["abnormal_class_names"]]
        self.normal_class_name = str(asset["normal_class_name"])
        self.ln_eps = float(asset["ln_eps"])
        for name in (
            "ln_weight",
            "ln_bias",
            "visual_projection",
            "normal_text",
            "abnormal_text",
            "text_directions",
            "normal_mean",
            "normal_std",
            "normal_indices",
            "normal_prototypes",
        ):
            self.register_buffer(name, torch.as_tensor(asset[name]).clone(), persistent=True)
        if self.ln_weight.ndim != 1 or self.ln_weight.numel() != 768:
            raise ValueError("lens ln_weight must have shape [768]")
        if self.visual_projection.shape != (768, 512):
            raise ValueError("lens visual_projection must have shape [768,512]")
        if self.normal_text.shape != (512,):
            raise ValueError("lens normal_text must have shape [512]")
        if self.abnormal_text.ndim != 2 or self.abnormal_text.shape[1] != 512:
            raise ValueError("lens abnormal_text must have shape [C,512]")
        if self.text_directions.shape != (self.abnormal_text.shape[0], 768):
            raise ValueError("lens text_directions must have shape [C,768]")
        if len(self.class_names) != self.abnormal_text.shape[0]:
            raise ValueError("lens class-name count must equal text-direction count")
        if self.normal_indices.ndim != 1 or self.normal_prototypes.ndim != 2:
            raise ValueError("lens normality bank has invalid dimensions")
        if self.normal_prototypes.shape[1] != self.normal_indices.numel():
            raise ValueError("lens normal prototype width differs from normal indices")

    @property
    def input_width(self) -> int:
        return int(self.ln_weight.numel())

    @property
    def abnormal_class_count(self) -> int:
        return int(self.abnormal_text.shape[0])

    def forward(
        self,
        clip_feature: torch.Tensor,
        last_hidden: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full evidence, class routing, normal distance and contributions."""
        if clip_feature.ndim != 3 or clip_feature.shape[-1] != 512:
            raise ValueError(f"expected clip feature [B,T,512], got {tuple(clip_feature.shape)}")
        if last_hidden.shape != (*clip_feature.shape[:2], self.input_width):
            raise ValueError(
                f"expected raw final hidden [B,T,{self.input_width}], got {tuple(last_hidden.shape)}"
            )
        if temperature <= 0:
            raise ValueError("text routing temperature must be positive")
        post = F.layer_norm(
            last_hidden.float(),
            (self.input_width,),
            self.ln_weight.float(),
            self.ln_bias.float(),
            self.ln_eps,
        )
        centered = post - self.normal_mean.float()
        normalized_clip = F.normalize(clip_feature.float(), dim=-1, eps=1e-6)
        abnormal_margin = normalized_clip @ self.abnormal_text.float().t()
        normal_margin = normalized_clip @ self.normal_text.float()
        class_route = F.softmax((abnormal_margin - normal_margin.unsqueeze(-1)) / float(temperature), dim=-1)

        # [B,T,C,768]. All dimensions remain in the residual input.
        contributions = centered.unsqueeze(-2) * self.text_directions.float().unsqueeze(0).unsqueeze(0)
        normalized_contributions = F.layer_norm(contributions, (self.input_width,))
        text_evidence = torch.sum(class_route.unsqueeze(-1) * normalized_contributions, dim=-2)

        indices = self.normal_indices.long()
        z = post.index_select(-1, indices) - self.normal_mean.float().index_select(0, indices)
        z = z / self.normal_std.float().index_select(0, indices).clamp_min(1e-6)
        flat_z = z.reshape(-1, z.shape[-1])
        normal_distance = torch.cdist(flat_z, self.normal_prototypes.float()).amin(dim=-1).reshape(z.shape[:-1])
        return (
            text_evidence.to(dtype=clip_feature.dtype),
            class_route.to(dtype=clip_feature.dtype),
            normal_distance.to(dtype=clip_feature.dtype),
            contributions.to(dtype=clip_feature.dtype),
        )

"""Fixed, auditable temporal operators over selected CLIP coordinates.

The operators in this file have no learned parameters.  A contribution can
therefore always be traced back to an original CLIP coordinate and one named
temporal event (level, motion, or short/long-term departure).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


OPERATOR_NAMES = (
    "text_aligned_level",
    "text_aligned_velocity",
    "opposite_text_velocity",
    "text_aligned_short_long_shift",
    "persistent_text_aligned_level",
)


def check_odd_window(name: str, width: int) -> int:
    """Return a valid centered-window width shared by discovery and training."""
    value = int(width)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd number, got {width}")
    return value


def numpy_centered_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Valid-sample centered mean for a ``[T,K]`` video trajectory."""
    width = check_odd_window("temporal window", width)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("expected non-empty [time, channels] values")
    radius = width // 2
    cumulative = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.float64), values.astype(np.float64).cumsum(axis=0)], axis=0,
    )
    positions = np.arange(len(values))
    left, right = np.maximum(positions - radius, 0), np.minimum(positions + radius + 1, len(values))
    return ((cumulative[right] - cumulative[left]) / (right - left)[:, None]).astype(np.float32)


def numpy_temporal_dynamics(zscore: np.ndarray, short_window: int, long_window: int) -> np.ndarray:
    """Return signed velocity and short-minus-long change, shape ``[T,K,2]``."""
    if short_window >= long_window:
        raise ValueError("short temporal window must be smaller than long temporal window")
    values = np.asarray(zscore, dtype=np.float32)
    velocity = np.zeros_like(values)
    if len(values) > 1:
        velocity[1:] = values[1:] - values[:-1]
    short = numpy_centered_mean(values, short_window)
    long = numpy_centered_mean(values, long_window)
    return np.stack([velocity, short - long], axis=-1)


def masked_centered_mean(values: torch.Tensor, mask: torch.Tensor, width: int) -> torch.Tensor:
    """Masked centered mean for padded ``[B,T,K]`` trajectories.

    Only valid samples enter each local mean.  This vectorized rule is exactly
    the same as :func:`numpy_centered_mean`; a zero padding tail can never
    become an event.
    """
    width = check_odd_window("temporal window", width)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("masked_centered_mean expects values [B,T,K] and mask [B,T]")
    _batch, _time, _channels = values.shape
    radius = width // 2
    valid = mask.to(values.dtype).unsqueeze(1)
    numerators = F.avg_pool1d(values.transpose(1, 2) * valid, width, stride=1, padding=radius, count_include_pad=False)
    counts = F.avg_pool1d(valid, width, stride=1, padding=radius, count_include_pad=False).clamp_min(1e-6)
    return (numerators / counts).transpose(1, 2).masked_fill(~mask.unsqueeze(-1), 0.0)


def masked_temporal_dynamics(zscore: torch.Tensor, mask: torch.Tensor, short_window: int, long_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of :func:`numpy_temporal_dynamics` for batched videos."""
    if short_window >= long_window:
        raise ValueError("short temporal window must be smaller than long temporal window")
    velocity = torch.zeros_like(zscore)
    if zscore.shape[1] > 1:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        velocity[:, 1:] = (zscore[:, 1:] - zscore[:, :-1]).masked_fill(~pair_mask.unsqueeze(-1), 0.0)
    short = masked_centered_mean(zscore, mask, short_window)
    long = masked_centered_mean(zscore, mask, long_window)
    return velocity, (short - long).masked_fill(~mask.unsqueeze(-1), 0.0)

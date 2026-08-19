"""Read-only adapter for the bundled, unmodified VadCLIP baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .common import labels_for_dataset


def add_vadclip_source() -> Path:
    """Expose only the local bundled VadCLIP source; never modify it."""
    root = Path(__file__).resolve().parents[1] / "VadCLIP"
    source = root / "src"
    if not (source / "model.py").is_file():
        raise FileNotFoundError(f"bundled VadCLIP source is incomplete: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return root


def load_options(dataset: str):
    add_vadclip_source()
    if dataset == "xd":
        import xd_option as options
    elif dataset == "ucf":
        import ucf_option as options
    else:
        raise ValueError(f"unsupported dataset={dataset!r}")
    return options.parser.parse_args([])


def checkpoint_state(path: str | Path) -> dict:
    content = torch.load(path, map_location="cpu", weights_only=False)
    state = content.get("model_state_dict", content) if isinstance(content, dict) else content
    if not isinstance(state, dict):
        raise ValueError(f"{path}: no model state dictionary")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def build_frozen_baseline(dataset: str, checkpoint: str | Path, device: torch.device):
    add_vadclip_source()
    from model import CLIPVAD

    options = load_options(dataset)
    model = CLIPVAD(
        options.classes_num, options.embed_dim, options.visual_length, options.visual_width,
        options.visual_head, options.visual_layers, options.attn_window,
        options.prompt_prefix, options.prompt_postfix, str(device),
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), options


def chunk_lengths(length: int, maxlen: int) -> torch.Tensor:
    remaining = int(length)
    values = torch.zeros(int(length / maxlen) + 1, dtype=torch.int64)
    for index in range(len(values)):
        if index == 0 and length < maxlen:
            values[index] = length
        elif index == 0 and length > maxlen:
            values[index] = maxlen
            remaining -= maxlen
        elif remaining > maxlen:
            values[index] = maxlen
            remaining -= maxlen
        else:
            values[index] = remaining
    return values


@torch.no_grad()
def score_sequence(model, sequence: np.ndarray, maxlen: int, dataset: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact official VadCLIP chunking and logits1/logits2 score convention."""
    add_vadclip_source()
    from utils.tools import get_batch_mask

    if sequence.ndim != 2 or sequence.shape[1] != 512 or not len(sequence):
        raise ValueError(f"expected non-empty [T,512] feature, got {sequence.shape}")
    length = len(sequence)
    chunks: list[np.ndarray] = []
    for index in range(1 if length < maxlen else int(length / maxlen) + 1):
        part = sequence[index * maxlen:(index + 1) * maxlen]
        if len(part) < maxlen:
            part = np.pad(part, ((0, maxlen - len(part)), (0, 0)), mode="constant")
        chunks.append(part)
    visual = torch.from_numpy(np.stack(chunks)).to(device)
    lengths = chunk_lengths(length, maxlen).to(device)
    mask = get_batch_mask(lengths, maxlen).to(device)
    _text, logits1, logits2 = model(visual, mask, list(labels_for_dataset(dataset).values()), lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:length]
    probability1 = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy().astype(np.float32)
    all_probability = logits2.softmax(dim=-1).cpu().numpy().astype(np.float32)
    return probability1, (1.0 - all_probability[:, 0]).astype(np.float32), all_probability

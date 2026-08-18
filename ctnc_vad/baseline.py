"""Local adapter for the unmodified VadCLIP baseline used by CTNC-VAD."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .common import labels_for_dataset


def add_vadclip_source() -> Path:
    root = Path(__file__).resolve().parents[1] / "VadCLIP"
    source = root / "src"
    if not (source / "model.py").is_file():
        raise FileNotFoundError(f"local VadCLIP source is incomplete: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return root


def load_options(dataset: str):
    add_vadclip_source()
    if dataset == "xd":
        import xd_option as option_module
    elif dataset == "ucf":
        import ucf_option as option_module
    else:
        raise ValueError(f"unsupported dataset={dataset!r}")
    return option_module.parser.parse_args([])


def state_dict_from_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"{path}: no state dictionary found")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def build_frozen_baseline(dataset: str, checkpoint: str | Path, device: torch.device):
    add_vadclip_source()
    from model import CLIPVAD

    options = load_options(dataset)
    model = CLIPVAD(
        options.classes_num, options.embed_dim, options.visual_length, options.visual_width,
        options.visual_head, options.visual_layers, options.attn_window,
        options.prompt_prefix, options.prompt_postfix, str(device),
    )
    model.load_state_dict(state_dict_from_checkpoint(checkpoint), strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, options


def chunk_lengths(length: int, maxlen: int) -> torch.Tensor:
    """Exactly reproduce the official VadCLIP test chunk length construction."""
    remaining = int(length)
    result = torch.zeros(int(length / maxlen) + 1, dtype=torch.int64)
    for index in range(len(result)):
        if index == 0 and length < maxlen:
            result[index] = length
        elif index == 0 and length > maxlen:
            result[index] = maxlen
            remaining -= maxlen
        elif remaining > maxlen:
            result[index] = maxlen
            remaining -= maxlen
        else:
            result[index] = remaining
    return result


@torch.no_grad()
def score_sequence(model, sequence: np.ndarray, maxlen: int, dataset: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run frozen VadCLIP with its original logits1/logits2 inference protocol."""
    add_vadclip_source()
    from utils.tools import get_batch_mask

    if sequence.ndim != 2 or sequence.shape[1] != 512 or len(sequence) == 0:
        raise ValueError(f"expected non-empty [T,512] feature sequence, got {sequence.shape}")
    length = len(sequence)
    # Match ``utils.tools.process_split`` exactly, including its final padded
    # chunk when length is an exact multiple of ``maxlen``.
    chunk_count = 1 if length < maxlen else int(length / maxlen) + 1
    chunks: list[np.ndarray] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * maxlen
        part = sequence[start:start + maxlen]
        if len(part) < maxlen:
            part = np.pad(part, ((0, maxlen - len(part)), (0, 0)), mode="constant")
        chunks.append(part)
    visual = torch.from_numpy(np.stack(chunks)).to(device)
    lengths = chunk_lengths(length, maxlen).to(device)
    mask = get_batch_mask(lengths, maxlen).to(device)
    prompt_text = list(labels_for_dataset(dataset).values())
    _text, logits1, logits2 = model(visual, mask, prompt_text, lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:length]
    probability1 = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy().astype(np.float32)
    probability2_all = logits2.softmax(dim=-1).cpu().numpy().astype(np.float32)
    probability2 = (1.0 - probability2_all[:, 0]).astype(np.float32)
    return probability1, probability2, probability2_all

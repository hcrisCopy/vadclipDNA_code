"""Thin, local adapter around the unmodified VadCLIP baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .common import labels_for_dataset


def local_vadclip_root() -> Path:
    return Path(__file__).resolve().parents[1] / "VadCLIP"


def add_local_vadclip_source() -> Path:
    """Expose only this repository's unmodified VadCLIP source directory."""
    root = local_vadclip_root().resolve()
    source = root / "src"
    if not (source / "model.py").is_file():
        raise FileNotFoundError(f"local VadCLIP baseline is incomplete: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return root


def load_options(dataset: str = "xd"):
    add_local_vadclip_source()
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
        raise ValueError(f"{path}: checkpoint does not contain a model state dictionary")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def build_baseline(options, checkpoint_path: str | Path, device: torch.device):
    """Load the official 512D XD VadCLIP model from this repository only."""
    add_local_vadclip_source()
    from model import CLIPVAD

    model = CLIPVAD(
        options.classes_num,
        options.embed_dim,
        options.visual_length,
        options.visual_width,
        options.visual_head,
        options.visual_layers,
        options.attn_window,
        options.prompt_prefix,
        options.prompt_postfix,
        str(device),
    )
    model.load_state_dict(state_dict_from_checkpoint(checkpoint_path), strict=True)
    model.to(device).eval()
    return model


def score_sequence(
    model,
    sequence: np.ndarray,
    visual_length: int,
    device: torch.device,
    dataset: str = "xd",
) -> np.ndarray:
    """Return VadCLIP sigmoid(logits1), with the official 256-step chunking."""
    if sequence.ndim != 2 or sequence.shape[1] != 512:
        raise ValueError(f"expected [T,512] VadCLIP input, got {sequence.shape}")
    chunks: list[np.ndarray] = []
    lengths: list[int] = []
    for start in range(0, len(sequence), visual_length):
        piece = sequence[start:start + visual_length]
        lengths.append(len(piece))
        if len(piece) < visual_length:
            piece = np.pad(piece, ((0, visual_length - len(piece)), (0, 0)), mode="constant")
        chunks.append(piece[None])
    visual = torch.from_numpy(np.concatenate(chunks, axis=0)).to(device)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    with torch.no_grad():
        _text, logits1, _logits2 = model(visual, None, list(labels_for_dataset(dataset).values()), valid_lengths)
    return torch.sigmoid(logits1.reshape(-1)[:len(sequence)]).cpu().numpy().astype(np.float32)

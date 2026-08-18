"""Hidden-state datasets that preserve VadCLIP's temporal preprocessing contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    align_hidden,
    base_key,
    grouped_source_rows,
    is_normal_video,
    load_clip_feature,
    load_hidden,
    normal_label,
    read_source_csv,
    resample_for_train,
    resolve_path,
)


def select_circuit(hidden: np.ndarray, selected_layers: np.ndarray, selected_dimensions: np.ndarray) -> np.ndarray:
    if hidden.ndim != 3:
        raise ValueError(f"expected [T,L,D] hidden state, got {hidden.shape}")
    if np.any(selected_layers < 0) or np.any(selected_layers >= hidden.shape[1]):
        raise ValueError(f"selected layer is invalid for hidden state {hidden.shape}")
    if np.any(selected_dimensions < 0) or np.any(selected_dimensions >= hidden.shape[2]):
        raise ValueError(f"selected dimension is invalid for hidden state {hidden.shape}")
    return hidden[:, selected_layers, selected_dimensions].astype(np.float32, copy=False)


class HiddenBagDataset(Dataset):
    """One VAD video per item; no baseline score is used to form a frame label."""

    def __init__(
        self,
        dataset: str,
        source_csv: str,
        source_path_base: str,
        hidden_by_key: dict[str, Path],
        selected_layers: np.ndarray,
        selected_dimensions: np.ndarray,
        visual_length: int,
        training: bool,
        alignment: str,
        allow_missing_hidden: bool = False,
    ) -> None:
        self.dataset = dataset
        self.source_csv = Path(source_csv)
        self.source_path_base = Path(source_path_base)
        self.hidden_by_key = hidden_by_key
        self.selected_layers = np.asarray(selected_layers, dtype=np.int64)
        self.selected_dimensions = np.asarray(selected_dimensions, dtype=np.int64)
        self.visual_length = int(visual_length)
        self.training = bool(training)
        self.alignment = alignment
        self.normal = normal_label(dataset)
        groups = grouped_source_rows(read_source_csv(self.source_csv))
        self.items: list[dict[str, object]] = []
        self.skipped: list[str] = []
        for key, group in groups.items():
            hidden_path = hidden_by_key.get(key)
            if hidden_path is None:
                if allow_missing_hidden and self.training:
                    self.skipped.append(key)
                    continue
                raise FileNotFoundError(f"{key}: no hidden state found in the supplied manifest")
            row = group.iloc[0]
            self.items.append({
                "key": key,
                "label": str(row["label"]),
                "feature_paths": [resolve_path(str(value), self.source_path_base) for value in group["path"].astype(str)],
                "hidden_path": hidden_path,
            })
        if not self.items:
            raise ValueError("no videos remain after matching the source CSV and hidden manifest")

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        item = self.items[index]
        hidden = load_hidden(item["hidden_path"])
        feature = np.concatenate([load_clip_feature(path) for path in item["feature_paths"]], axis=0)
        hidden, _action = align_hidden(hidden, len(feature), self.alignment)
        circuit = select_circuit(hidden, self.selected_layers, self.selected_dimensions)
        last_hidden = hidden[:, -1, :].astype(np.float32, copy=False)
        return circuit, last_hidden, feature

    def __getitem__(self, index: int):
        item = self.items[index]
        circuit, last_hidden, feature = self._load(index)
        if self.training:
            circuit, valid_length = resample_for_train(circuit, self.visual_length)
            last_hidden, valid_last_length = resample_for_train(last_hidden, self.visual_length)
            if valid_length != valid_last_length:
                raise RuntimeError("circuit and final hidden resampling disagree")
            return (
                torch.from_numpy(circuit),
                torch.from_numpy(last_hidden),
                torch.tensor(valid_length, dtype=torch.int64),
                torch.tensor(0 if is_normal_video(self.dataset, str(item["label"])) else 1, dtype=torch.float32),
            )
        return {
            "circuit": torch.from_numpy(circuit),
            "last_hidden": torch.from_numpy(last_hidden),
            "length": int(len(feature)),
            "label": str(item["label"]),
            "key": str(item["key"]),
            "clip_feature": torch.from_numpy(feature),
            "hidden_path": Path(item["hidden_path"]),
        }

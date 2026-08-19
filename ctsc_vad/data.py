"""Datasets for raw, selected CLIP coordinates and frozen baseline scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_cached_probability
from .common import (
    align_hidden, grouped_source_rows, load_clip_feature, load_hidden, read_source_csv,
    resample_for_train, resolve_path, video_label_vector,
)


def extract_channels(hidden: np.ndarray, layers: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    return hidden[:, layers, dimensions].astype(np.float32, copy=False)


class ChannelBagDataset(Dataset):
    """One source video per item; video labels never come from baseline scores."""

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
        cached_probabilities: dict[str, Path] | None = None,
    ) -> None:
        self.dataset, self.source_path_base = dataset, Path(source_path_base)
        self.layers = np.asarray(selected_layers, dtype=np.int64)
        self.dimensions = np.asarray(selected_dimensions, dtype=np.int64)
        self.visual_length, self.training, self.alignment = int(visual_length), bool(training), alignment
        self.cached_probabilities = cached_probabilities or {}
        self.items: list[dict[str, object]] = []
        self.skipped: list[str] = []
        for key, group in grouped_source_rows(read_source_csv(source_csv)).items():
            hidden_path = hidden_by_key.get(key)
            if hidden_path is None:
                if allow_missing_hidden:
                    self.skipped.append(key)
                    continue
                raise FileNotFoundError(f"{key}: absent from supplied hidden manifest")
            row = group.iloc[0]
            self.items.append({
                "key": key, "label": str(row["label"]), "hidden_path": hidden_path,
                "feature_path": resolve_path(str(row["path"]), self.source_path_base),
            })
        if not self.items:
            raise ValueError("no source videos match the hidden manifest")

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        item = self.items[index]
        feature = load_clip_feature(item["feature_path"])
        hidden = align_hidden(load_hidden(item["hidden_path"]), len(feature), self.alignment)
        circuit = extract_channels(hidden, self.layers, self.dimensions)
        final_hidden = hidden[:, -1, :].astype(np.float32, copy=False)
        probability_path = self.cached_probabilities.get(str(item["key"]))
        probability = None if probability_path is None else load_cached_probability(probability_path, len(feature))
        return circuit, final_hidden, feature, probability

    def __getitem__(self, index: int):
        item = self.items[index]
        circuit, final_hidden, feature, probability = self._load(index)
        if self.training:
            circuit, length = resample_for_train(circuit, self.visual_length)
            final_hidden, final_length = resample_for_train(final_hidden, self.visual_length)
            if length != final_length or probability is None:
                raise RuntimeError("training circuit/final/baseline inputs are inconsistent")
            probability, probability_length = resample_for_train(probability, self.visual_length)
            if length != probability_length:
                raise RuntimeError("resampled baseline probabilities do not align")
            return (
                torch.from_numpy(circuit), torch.from_numpy(final_hidden), torch.from_numpy(probability),
                torch.tensor(length, dtype=torch.int64),
                torch.from_numpy(video_label_vector(self.dataset, str(item["label"]))[1:]),
            )
        return {
            "key": str(item["key"]), "label": str(item["label"]), "length": len(feature),
            "circuit": torch.from_numpy(circuit), "final_hidden": torch.from_numpy(final_hidden),
            "clip_feature": torch.from_numpy(feature),
        }

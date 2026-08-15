"""Dataset wrapper that preserves VadCLIP's XD temporal preprocessing exactly."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.utils.data as data

from .common import load_clip_feature, read_path_label_csv, resolve_recorded_path
from .vadclip import add_local_vadclip_source


class XDDNAFeatureDataset(data.Dataset):
    def __init__(self, csv_path: str, visual_length: int, input_width: int, test_mode: bool) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.frame = read_path_label_csv(self.csv_path)
        self.visual_length = int(visual_length)
        self.input_width = int(input_width)
        self.test_mode = bool(test_mode)
        add_local_vadclip_source()
        from utils.tools import process_feat, process_split

        self.process_feat = process_feat
        self.process_split = process_split

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        listed_path = str(row["path"])
        feature_path = resolve_recorded_path(listed_path, self.csv_path.parent)
        feature = load_clip_feature(feature_path)
        if feature.shape[1] != self.input_width:
            raise ValueError(f"{feature_path}: expected {self.input_width}D fused input, got {feature.shape}")
        if self.test_mode:
            feature, length = self.process_split(feature, self.visual_length)
            return torch.from_numpy(feature), str(row["label"]), int(length), listed_path
        feature, length = self.process_feat(feature, self.visual_length)
        return torch.from_numpy(feature), str(row["label"]), int(length)

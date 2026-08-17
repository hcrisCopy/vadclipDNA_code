"""Repository-local helpers for the NeuVAD-Lens VadCLIP experiment.

Only reusable data artifacts from xd_dna are imported. This package never
imports a model or code path from a sibling project, and every artifact it
writes is constrained to the sibling vadclipDNA_data directory.
"""
from __future__ import annotations

from pathlib import Path

from xd_dna.common import (
    atomic_save_npy,
    atomic_save_npz,
    atomic_torch_save,
    base_key,
    labels_for_dataset,
    load_clip_feature,
    load_hidden,
    load_json,
    manifest_hidden_paths,
    read_path_label_csv,
    relpath,
    resolve_recorded_path,
    save_json,
    set_seed,
    stage_dir,
    write_csv,
)


def default_output_root() -> Path:
    """Return the approved default experiment root."""
    return Path("../vadclipDNA_data/xd_neuvad_lens")


def required_lens_path(output_root: str | Path, lens_assets: str) -> Path:
    """Resolve an explicit or default lens asset."""
    if lens_assets:
        return Path(lens_assets).resolve()
    return (Path(output_root).resolve() / "lens" / "lens_assets.pt").resolve()

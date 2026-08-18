"""Create an audited VadCLIP input contract from DSANet-localized neurons."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..common import save_json
from .common import (
    CLIP_FEATURE_DIM,
    default_output_root,
    fdu_fingerprint,
    load_json_object,
    output_path,
    relative_metadata_path,
    remove_tree,
    sha256_file,
    validate_dsanet_fdu_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import DSANet FDU identities as an auditable VadCLIP transfer contract."
    )
    parser.add_argument("--dsanet-fdu-json", required=True, help="DSANet localization/fdu_indices.json.")
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--clean", action="store_true", help="Delete and rebuild only the transfer contract stage.")
    parser.add_argument("--no-resume", action="store_true", help="Rewrite the contract even when it already matches.")
    args = parser.parse_args()
    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")

    root = output_path(args.output_root)
    stage = root / "contract"
    if args.clean:
        remove_tree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    source = Path(args.dsanet_fdu_json).resolve()
    specification = load_json_object(source)
    fdus = validate_dsanet_fdu_spec(specification, source)
    target = stage / "dsanet_transfer_neurons.json"
    contract = {
        "format_version": 1,
        "method": "DSANet-localized DNA neuron transfer to VadCLIP",
        "dataset": "xd",
        "clip_model": "ViT-B/16",
        "token_pool": "cls",
        "unit_definition": "frozen CLIP ViT-B/16 visual block output CLS dimension",
        "source_project": "DSANet_DNA",
        "source_fdu_json": relative_metadata_path(source, target.parent),
        "source_fdu_json_sha256": sha256_file(source),
        "source_fdu_fingerprint": fdu_fingerprint(fdus),
        "critical_layers": [int(value) for value in specification.get("critical_layers", [])],
        "fdus": fdus,
        "neuron_width": len(fdus),
        "clip_dim": CLIP_FEATURE_DIM,
        "input_width": len(fdus) + CLIP_FEATURE_DIM,
        "concat_contract": {
            "order": "dsanet_fdu_zscore_then_vadclip_clip",
            "neuron_normalization": "XD train pure-normal (label A) mean/std only",
            "clip_normalization": "raw official VadCLIP 512D features",
            "temporal_alignment": "strict exact T match",
        },
    }
    if target.exists() and not args.no_resume:
        existing = load_json_object(target)
        if existing == contract:
            print(f"reused transfer contract: {target}", flush=True)
            return
        raise RuntimeError(f"{target}: transfer contract differs; use --clean or a new --output-root")
    save_json(target, contract)
    print(
        f"wrote {target}: {len(fdus)} DSANet-localized neurons, VadCLIP input width={contract['input_width']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

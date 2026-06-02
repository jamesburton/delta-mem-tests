"""Emit a pair of OSCAR-format rotation files where every layer's rotation
is the identity matrix.

Use case: testing the OSCAR pipeline (INT2 / INT4 quantized middle, sink +
recent windows, our cache fast paths) without any rotation effects. Run the
LoCoMo eval with these files and ``KV_CACHE_BACKEND=oscar`` and the OSCAR
rotation step is a no-op end-to-end. Quality comes purely from the
quant/dequant grid choice.

This is the raw-INT4-vs-rotated-INT4 control. Saw-INT4 and KIVI-4 papers
both report near-bf16 quality from the unrotated basis at 4-bit; OSCAR's
eigenbasis rotation is INT2-specific. The identity-rotation files let us
verify on our actual setup.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=36)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eye = torch.eye(args.head_dim, dtype=torch.float32)
    ones = torch.ones(args.head_dim, dtype=torch.float32)

    layers = {
        i: {"layer_id": i, "rotation": eye.clone(), "eigenvalues": ones.clone()}
        for i in range(args.num_layers)
    }
    for tag in ("k", "v"):
        objective = f"identity_{'qqt' if tag == 'k' else 'sst'}_r_h_pbr"
        blob = {
            "format_version": 1,
            "objective": objective,
            "source_grouping": "layer",
            "layers": layers,
        }
        path = out / f"{tag}_rotation_identity.pt"
        torch.save(blob, str(path))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

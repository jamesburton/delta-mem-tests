"""Validate a Strix-trained delta-mem adapter on the LOCAL 12 GB host.

Runs the three scenarios from ``LONG_CONTEXT_PLAN.md`` Option 1 step 2
against a checkpoint directory by invoking ``run.locomo_eval
--adapter-override`` as a subprocess (mirrors how every other eval in
this repo runs). Reads each output JSON for the delta/base ratio and
prints a pass/fail table.

Scenarios:

  1. **Anchor**     conv-26 / 10 q at ~17 k context — expects ratio >= 1.25
  2. **Extension**  conv-41 / 10 q at ~25 k context — expects ratio >= 1.20
  3. **Stretch**    synthetic conv-26 x2 at ~32 k     — expects ratio >= 1.10

OSCAR INT2 backend + GPQA-calibrated rotations are used (the production
combo). Designed to run on the local Windows/CUDA host (RTX 3060 12 GB);
the Strix Halo box is not needed for this script.

Usage:
    python -m strix.verify_checkpoint --ckpt path\\to\\checkpoint\\final
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROTATIONS = {
    "OSCAR_K_ROTATION_PATH": "data/oscar/rotations/instruct_gpqa/k_rotation_qqt_r_h_pbr.pt",
    "OSCAR_V_ROTATION_PATH": "data/oscar/rotations/instruct_gpqa/v_rotation_sst_r_h_pbr.pt",
    "OSCAR_DISABLE_DEQUANT_SHADOW": "1",
    "KV_CACHE_BACKEND": "oscar",
    "KV_CACHE_BITS": "2",
    "PYTHONIOENCODING": "utf-8",
}

SCENARIOS = [
    # (name, data_file, max_questions, threshold, builder)
    # builder: optional shell command that must run to produce data_file
    ("anchor_17k", "data/locomo_conv-26.json", 10, 1.25,
     [sys.executable, "-m", "run.build_context_sweep_data", "single", "conv-26"]),
    ("extension_25k", "data/locomo_conv-41.json", 10, 1.20,
     [sys.executable, "-m", "run.build_context_sweep_data", "single", "conv-41"]),
    ("stretch_32k", "data/locomo_conv-26_x2.json", 10, 1.10,
     [sys.executable, "-m", "run.build_context_sweep_data", "extend", "conv-26", "2"]),
]


def _ensure_data(builder: List[str], data_file: Path) -> bool:
    if data_file.exists():
        return True
    print(f"[verify] building {data_file} via: {' '.join(builder)}", flush=True)
    res = subprocess.run(builder, capture_output=False)
    return res.returncode == 0 and data_file.exists()


def _run_eval(ckpt: Path, data_file: Path, max_q: int,
              out_json: Path) -> Optional[float]:
    """Run locomo_eval and return the delta/base ratio (or None on failure)."""
    env = os.environ.copy()
    env.update(ROTATIONS)
    cmd = [
        sys.executable, "-m", "run.locomo_eval",
        "--kv-cache-backend", "oscar",
        "--kv-cache-bits", "2",
        "--eval-batch-size", "1",
        "--max-conversations", "1",
        "--max-questions-per-conversation", str(max_q),
        "--data-file", str(data_file),
        "--adapter-override", str(ckpt),
        "--output-json", str(out_json),
    ]
    print(f"[verify] running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, env=env, capture_output=False)
    if res.returncode != 0:
        print(f"[verify] eval failed (rc={res.returncode})", flush=True)
        return None
    if not out_json.exists():
        print(f"[verify] eval output missing: {out_json}", flush=True)
        return None
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    # locomo_eval scores typically expose summary scores; ratio = delta/base
    # We tolerate a few schema variants.
    summary = payload.get("summary") or payload.get("overall") or payload
    delta = summary.get("delta_score") or summary.get("delta") or summary.get("delta_mean")
    base = summary.get("base_score") or summary.get("base") or summary.get("base_mean")
    if delta is None or base is None or base == 0:
        ratio = summary.get("ratio")
        return float(ratio) if ratio is not None else None
    return float(delta) / float(base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="Path to the trained delta-mem adapter directory.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/verify_ckpt"),
                    help="Directory for per-scenario eval JSONs.")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    if not ckpt.is_dir():
        print(f"[verify] not a directory: {ckpt}", file=sys.stderr)
        return 1
    if not (ckpt / "delta_mem_adapter.pt").exists():
        print(f"[verify] missing delta_mem_adapter.pt under {ckpt}",
              file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check the rotations exist before burning eval cycles
    for key in ("OSCAR_K_ROTATION_PATH", "OSCAR_V_ROTATION_PATH"):
        if not Path(ROTATIONS[key]).exists():
            print(f"[verify] missing rotation file: {ROTATIONS[key]}",
                  file=sys.stderr)
            return 1

    results = []
    for name, data_str, max_q, threshold, builder in SCENARIOS:
        data_file = Path(data_str)
        if not _ensure_data(builder, data_file):
            results.append((name, threshold, None, "DATA_BUILD_FAILED"))
            continue
        out_json = args.out_dir / f"verify_{name}.json"
        ratio = _run_eval(ckpt, data_file, max_q, out_json)
        if ratio is None:
            results.append((name, threshold, None, "EVAL_FAILED"))
            continue
        status = "PASS" if ratio >= threshold else "FAIL"
        results.append((name, threshold, ratio, status))

    # Pass/fail summary table
    print("\n[verify] results")
    print(f"  {'scenario':<14} {'threshold':>10} {'ratio':>8}   status")
    print(f"  {'-'*14} {'-'*10} {'-'*8}   ------")
    overall_pass = True
    for name, thr, ratio, status in results:
        ratio_str = f"{ratio:.3f}" if ratio is not None else "  n/a "
        print(f"  {name:<14} {thr:>10.2f} {ratio_str:>8}   {status}")
        if status != "PASS":
            overall_pass = False
    print()
    if overall_pass:
        print("[verify] ALL SCENARIOS PASSED — checkpoint is good to ship.")
        return 0
    print("[verify] one or more scenarios failed; review per-scenario JSONs "
          f"under {args.out_dir}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())

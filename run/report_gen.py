"""Pure-logic generator for the LoCoMo reproduction report.

Given eval scores and run metadata, produce a markdown report with a clear
verdict against the paper's 1.20x within a ±0.05 tolerance band.
"""

from __future__ import annotations

from typing import Literal

Verdict = Literal["PASS", "OUT_OF_BAND", "REGRESSION"]


def verdict(*, our_ratio: float, paper_ratio: float, tolerance: float) -> Verdict:
    """Classify our reproduction outcome.

    - REGRESSION: our ratio is below 1.0 (delta-mem hurt rather than helped).
    - PASS: our ratio is within `tolerance` of the paper's.
    - OUT_OF_BAND: everything else (deviation > tolerance, in either direction).
    """
    if our_ratio < 1.0:
        return "REGRESSION"
    if abs(our_ratio - paper_ratio) <= tolerance:
        return "PASS"
    return "OUT_OF_BAND"


def render_report(
    *,
    our_ratio: float,
    paper_ratio: float,
    tolerance: float,
    our_delta_mem_score: float,
    our_frozen_score: float,
    peak_vram_gb: float,
    skipped_samples: list[tuple[int, str]],
    vendored_commit: str,
    eval_config: dict,
) -> str:
    """Render the LoCoMo reproduction report as markdown."""
    v = verdict(our_ratio=our_ratio, paper_ratio=paper_ratio, tolerance=tolerance)
    deviation = abs(our_ratio - paper_ratio)

    lines: list[str] = [
        "# LoCoMo reproduction — delta-Mem on Qwen3-4B-Instruct-2507",
        "",
        f"**Verdict:** {v}",
        "",
        "## Headline",
        "",
        f"- Our delta-mem-vs-frozen-backbone ratio: **{our_ratio:.2f}×**",
        f"- Paper's reported ratio: **{paper_ratio:.2f}×**",
        f"- Tolerance band: **±{tolerance:.2f}**",
        f"- Deviation from paper: **{deviation:.2f}**",
        "",
        "## Scores",
        "",
        f"- delta-mem score: **{our_delta_mem_score:.4f}**",
        f"- frozen backbone score: **{our_frozen_score:.4f}**",
        "",
        "## Run metadata",
        "",
        f"- Vendored delta-Mem commit: `{vendored_commit}`",
        f"- Peak VRAM: **{peak_vram_gb:.2f} GB**",
        "",
        "### Eval config",
        "",
    ]
    for k, val in eval_config.items():
        lines.append(f"- `{k}`: `{val}`")
    lines.append("")

    lines.append("## Asterisks")
    lines.append("")
    if skipped_samples:
        lines.append(f"{len(skipped_samples)} sample(s) skipped (recorded, not silently dropped):")
        lines.append("")
        for idx, reason in skipped_samples:
            lines.append(f"- sample #{idx}: {reason}")
    else:
        lines.append("- None. All samples evaluated.")
    lines.append("")

    if v == "OUT_OF_BAND":
        lines.append("## Investigation note")
        lines.append("")
        lines.append(
            f"Our ratio differs from the paper by {deviation:.2f}, which exceeds "
            f"the ±{tolerance:.2f} tolerance. Per the spec, this is a finding to "
            "investigate, not to smooth over. See the raw outputs under "
            "`report/raw/` for the per-sample breakdown."
        )
        lines.append("")
    elif v == "REGRESSION":
        lines.append("## Investigation note")
        lines.append("")
        lines.append(
            f"Our ratio of {our_ratio:.2f} is below 1.0 — delta-mem failed to "
            "improve over the frozen backbone in our run. Treat as a failure of "
            "the reproduction; investigate before declaring Tier 1 complete."
        )
        lines.append("")

    return "\n".join(lines) + "\n"

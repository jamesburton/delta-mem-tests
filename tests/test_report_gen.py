"""Unit tests for the LoCoMo reproduction report generator."""

from run.report_gen import render_report, verdict


def test_verdict_inside_tolerance_passes():
    assert verdict(our_ratio=1.20, paper_ratio=1.20, tolerance=0.05) == "PASS"
    assert verdict(our_ratio=1.16, paper_ratio=1.20, tolerance=0.05) == "PASS"
    assert verdict(our_ratio=1.24, paper_ratio=1.20, tolerance=0.05) == "PASS"


def test_verdict_outside_tolerance_is_oob():
    assert verdict(our_ratio=1.10, paper_ratio=1.20, tolerance=0.05) == "OUT_OF_BAND"
    assert verdict(our_ratio=1.30, paper_ratio=1.20, tolerance=0.05) == "OUT_OF_BAND"


def test_verdict_below_one_is_regression():
    assert verdict(our_ratio=0.95, paper_ratio=1.20, tolerance=0.05) == "REGRESSION"


def test_render_report_contains_required_sections():
    out = render_report(
        our_ratio=1.18,
        paper_ratio=1.20,
        tolerance=0.05,
        our_delta_mem_score=0.59,
        our_frozen_score=0.50,
        peak_vram_gb=10.5,
        skipped_samples=[(7, "OOM at 12k tokens"), (42, "OOM at 14k tokens")],
        vendored_commit="abc123def",
        eval_config={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "adapter": "declare-lab/delta-mem_qwen3_4b-instruct",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "max_seq_len": 8192,
        },
    )
    assert "# LoCoMo reproduction" in out
    assert "PASS" in out
    assert "1.18" in out
    assert "1.20" in out
    assert "±0.05" in out
    assert "abc123def" in out
    assert "Qwen/Qwen3-4B-Instruct-2507" in out
    assert "OOM at 12k tokens" in out
    assert "10.50" in out  # peak VRAM, two decimals


def test_render_report_marks_oob_clearly():
    out = render_report(
        our_ratio=1.10,
        paper_ratio=1.20,
        tolerance=0.05,
        our_delta_mem_score=0.55,
        our_frozen_score=0.50,
        peak_vram_gb=11.2,
        skipped_samples=[],
        vendored_commit="abc123",
        eval_config={"model": "x", "adapter": "y", "dtype": "bfloat16",
                     "attn_implementation": "sdpa", "max_seq_len": 8192},
    )
    assert "OUT_OF_BAND" in out
    # No silent papering-over: the deviation magnitude must be visible
    assert "0.10" in out  # 1.20 - 1.10


def test_render_report_marks_regression_clearly():
    out = render_report(
        our_ratio=0.95,
        paper_ratio=1.20,
        tolerance=0.05,
        our_delta_mem_score=0.50,
        our_frozen_score=0.53,
        peak_vram_gb=10.0,
        skipped_samples=[],
        vendored_commit="abc123",
        eval_config={"model": "x", "adapter": "y", "dtype": "bfloat16",
                     "attn_implementation": "sdpa", "max_seq_len": 8192},
    )
    # Symmetric with the OOB test: a REGRESSION must surface that delta-mem
    # FAILED to improve over the frozen backbone, and the offending ratio
    # must be visible — same "no silent papering-over" rule the spec mandates.
    assert "REGRESSION" in out
    assert "0.95" in out
    assert "below 1.0" in out

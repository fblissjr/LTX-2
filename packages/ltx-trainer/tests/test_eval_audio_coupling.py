"""Tests for the scalable audio→video coupling eval.

The pure tracking-slope core is tested directly (no I/O). The full harness is
tested with real ffmpeg round-trip stand-ins: a "LoRA" arm whose clips pulse at
each case's own BPM (should track → slope≈1) vs a "baseline" arm whose clips pulse
at a FIXED BPM regardless (should be flat → slope≈0), so the eval must report a
large positive delta and PASS — and the inverse must FAIL.
"""

from __future__ import annotations

import shutil

import pytest

from ltx_trainer.eval_audio_coupling import (
    CouplingEvalReport,
    evaluate_coupling,
    tracking_slope,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")


def test_tracking_slope_perfect_and_flat():
    exp = [60, 90, 120, 150]
    s_track, r2_track = tracking_slope(exp, exp)        # output == input
    assert abs(s_track - 1.0) < 1e-6 and r2_track > 0.99
    s_flat, _ = tracking_slope(exp, [100, 100, 100, 100])  # ignores input
    assert abs(s_flat) < 1e-6


def test_tracking_slope_degenerate_inputs():
    assert tracking_slope([100], [100]) == (0.0, 0.0)       # <2 points
    assert tracking_slope([90, 90], [50, 150]) == (0.0, 0.0)  # no input spread


def test_report_pass_fail_logic():
    good = CouplingEvalReport(coupling="beat_pulse", n_cases=4, lora_slope=0.95,
                              baseline_slope=0.1, base_preservation_ok=True)
    assert good.passed and abs(good.delta - 0.85) < 1e-6
    marginal = CouplingEvalReport(coupling="beat_pulse", n_cases=4, lora_slope=0.95,
                                  baseline_slope=0.8)  # delta 0.15 < threshold
    assert not marginal.passed
    broke_base = CouplingEvalReport(coupling="beat_pulse", n_cases=4, lora_slope=0.95,
                                    baseline_slope=0.1, base_preservation_ok=False)
    assert not broke_base.passed


@_needs_ffmpeg
def test_evaluate_coupling_lora_tracks_baseline_flat(tmp_path):
    from ltx_trainer.synthetic_av import ClipSpec, generate_beat_pulse_clip, write_clip

    w = h = 64
    fps = 25
    bpms = [80, 110, 140]

    def _clip(bpm, name):
        frames, audio, sr, _ = generate_beat_pulse_clip(ClipSpec(bpm=bpm, duration_s=6.0, fps=fps, width=w, height=h))
        out = tmp_path / name
        write_clip(frames, audio, sr, fps, out)
        return str(out)

    # LoRA arm pulses at each case's own BPM (tracks); baseline pulses at a fixed 100 (flat).
    cases = [{"expected": b, "lora_video": _clip(b, f"lora_{b}.mp4"),
              "baseline_video": _clip(100, f"base_{b}.mp4")} for b in bpms]
    report = evaluate_coupling(cases, coupling="beat_pulse", width=w, height=h, fps=fps)

    assert report.lora_slope > 0.7, f"LoRA arm should track input, got {report.lora_slope:.2f}"
    assert abs(report.baseline_slope) < 0.3, f"baseline should be flat, got {report.baseline_slope:.2f}"
    assert report.passed, format_fail(report)


def format_fail(r):
    return f"expected pass: lora_slope={r.lora_slope:.2f} baseline_slope={r.baseline_slope:.2f} delta={r.delta:.2f}"

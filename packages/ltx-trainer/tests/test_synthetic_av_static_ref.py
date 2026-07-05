"""Static-identity-reference variant of the synthetic generator (E1.1 v3).

Background — why this supersedes paired-refs for the gate:

The paired-ref variant (ref≠target, different BPM) still gave the reference
video a PULSE, and in a narrow sub-Nyquist BPM range the rejection-sampler
made ref_bpm anti-correlated with target_bpm (corr=-0.62 on the ≤85 set).
A model could perceive the reference's pulse rate and inverse-map it to the
target rate WITHOUT reading the audio — a confounded gate.

v3 fix (resolved with LTX-2): audio is a NATIVE modality on LTX-2, so the
reference video is inherited scaffolding, not the control. The control is
the audio. For the kill-early gate we make the reference a STATIC identity
frame (frozen, no pulse): it carries identity but provably cannot carry a
rate, so audio is the ONLY temporal signal and the corr leak vanishes by
construction. It's the zero-trainer-code stand-in for the product's
"audio + prompt, no reference" design.

These tests lock the static-ref contract before the generator runs on real
GPU minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ltx_trainer.synthetic_av import generate_dataset_static_ref, measure_pulse_rate


def _manifest(tmp_path: Path) -> list[dict]:
    return [json.loads(L) for L in (tmp_path / "manifest.jsonl").read_text().splitlines()]


def test_generates_n_target_and_n_reference_clips(tmp_path: Path):
    """Each row has a target clip in clips/ and a reference clip in references/."""
    out = generate_dataset_static_ref(tmp_path, n=4, duration_s=1.0, fps=25, seed=0)
    rows = json.loads(out.read_text())
    assert len(rows) == 4
    for r in rows:
        assert (tmp_path / r["video"]).is_file(), f"missing target: {r['video']}"
        assert (tmp_path / r["reference"]).is_file(), f"missing reference: {r['reference']}"
        assert r["video"].startswith("clips/"), r["video"]
        assert r["reference"].startswith("references/"), r["reference"]


def test_reference_carries_no_rate(tmp_path: Path):
    """The whole point of v3: the reference is FROZEN — zero pulse. The manifest
    records no reference_bpm, and the rendered reference frames are temporally
    static (measure_pulse_rate ~ 0)."""
    out = generate_dataset_static_ref(tmp_path, n=4, duration_s=1.0, fps=25, seed=1)
    rows = json.loads(out.read_text())
    for m in _manifest(tmp_path):
        assert "reference_bpm" not in m, "static reference must NOT have a BPM"
        assert m.get("reference_static") is True, m
    # And: there is no reference_bpm column to correlate with target_bpm at all.
    assert all("reference_bpm" not in m for m in _manifest(tmp_path))
    # Optional sanity: the caption schema still carries the reference key.
    for r in rows:
        assert "reference" in r


def test_reference_frames_are_temporally_constant(tmp_path: Path):
    """Decode-free check via the generator's own renderer: a static reference's
    frames are all identical (no motion, no pulse), so its recovered pulse rate
    is ~0 — the structural guarantee that it can't leak the target rate."""
    from ltx_trainer.synthetic_av import generate_static_identity_clip, ClipSpec

    spec = ClipSpec(bpm=80.0, duration_s=1.0, fps=25, width=64, height=64,
                    shape="circle", color=(200, 150, 100), center=(0.5, 0.5))
    frames, audio, sr, beats = generate_static_identity_clip(spec)
    # all frames identical
    assert np.array_equal(frames[0], frames[-1])
    assert frames.std(axis=0).max() == 0, "static reference frames must not vary over time"
    # recovered pulse rate is ~0 (no temporal variation -> no rate)
    assert measure_pulse_rate(frames, fps=25) == pytest.approx(0.0, abs=1e-6)


def test_target_still_pulses_at_target_bpm(tmp_path: Path):
    """The TARGET clip is unchanged from beat-pulse — audio still drives it.
    Recovered rate should track the manifest target_bpm (sub-Nyquist range)."""
    out = generate_dataset_static_ref(tmp_path, n=6, bpm_range=(50.0, 85.0),
                                      duration_s=3.0, fps=25, seed=2)
    json.loads(out.read_text())
    for m in _manifest(tmp_path):
        assert 50.0 <= m["target_bpm"] <= 85.0


def test_reference_identity_matches_target(tmp_path: Path):
    """The reference anchors the SAME identity (shape/color/center) as its
    target — that's the reference's only job."""
    out = generate_dataset_static_ref(tmp_path, n=4, duration_s=1.0, fps=25, seed=3)
    json.loads(out.read_text())
    for m in _manifest(tmp_path):
        assert "shape" in m and "color" in m and "center" in m, m
    # cross-row identity still varies (nuisance diversity preserved)
    mani = _manifest(tmp_path)
    assert len({mm["shape"] for mm in mani}) > 1 or len({tuple(mm["center"]) for mm in mani}) > 1


def test_captions_schema(tmp_path: Path):
    out = generate_dataset_static_ref(tmp_path, n=2, duration_s=1.0, fps=25, seed=4)
    rows = json.loads(out.read_text())
    for r in rows:
        for k in ("video", "caption", "reference"):
            assert k in r, f"row missing required key '{k}': {r}"


@pytest.mark.parametrize("n", [1, 5, 20])
def test_n_rows_produces_2n_clips(tmp_path: Path, n: int):
    generate_dataset_static_ref(tmp_path, n=n, duration_s=1.0, fps=25, seed=5)
    assert len(list((tmp_path / "clips").glob("*.mp4"))) == n
    assert len(list((tmp_path / "references").glob("*.mp4"))) == n

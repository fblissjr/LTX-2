"""Paired-reference variant of the synthetic generator (E1.1 v2).

Background — why this exists:

E1.1 v1 failed because the synthetic generator wrote each clip as its own
reference (`r["reference"] = r["video"]`). With ref = target, the model
could satisfy the training loss by copying the reference video; audio was
structurally non-load-bearing. Combined with a broad-suffix LoRA target
preset that touched audio + cross-modal layers, this trained the LoRA to
overwrite base's audio coupling with near-zero deltas (musubi-tuner
docs/ltx_2.md line 1971 documents this exact failure).

v2 fix: each row gets a SEPARATE reference clip with the SAME visual
identity (shape/color/center) but a DIFFERENT BPM. The audio is the only
signal that distinguishes target rate from reference rate, so the LoRA
MUST use audio to predict the target — or it fails the loss.

These tests lock the paired-ref contract before the generator runs on real
GPU minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltx_trainer.synthetic_av import generate_dataset_paired_refs


def test_generates_n_target_and_n_reference_clips(tmp_path: Path):
    """Each of n rows has both a target clip and a reference clip on disk."""
    out = generate_dataset_paired_refs(tmp_path, n=4, duration_s=1.0, fps=25, seed=0)
    rows = json.loads(out.read_text())
    assert len(rows) == 4
    for r in rows:
        assert (tmp_path / r["video"]).is_file(), f"missing target: {r['video']}"
        assert (tmp_path / r["reference"]).is_file(), f"missing reference: {r['reference']}"
        # Convention: targets in clips/, references in references/. Keeps
        # process_dataset.py + verify_training_data.py able to find each
        # via the same captions.json schema.
        assert r["video"].startswith("clips/"), r["video"]
        assert r["reference"].startswith("references/"), r["reference"]


def test_each_row_has_different_bpm_for_ref_vs_target(tmp_path: Path):
    """The whole point of v2: audio MUST be load-bearing. Ref BPM ≠ target BPM
    per row, so visual reference alone can't predict target rate."""
    out = generate_dataset_paired_refs(tmp_path, n=8, duration_s=1.0, fps=25, seed=1)
    manifest = [json.loads(L) for L in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    for m in manifest:
        assert m["target_bpm"] != m["reference_bpm"], (
            f"row {m['video']}: target_bpm={m['target_bpm']} == reference_bpm={m['reference_bpm']} — "
            "v2 contract violated; ref must differ from target so audio is load-bearing"
        )


def test_ref_and_target_share_visual_identity_per_row(tmp_path: Path):
    """Same shape / color / center per row — the LoRA shouldn't have to
    re-learn visual identity from audio. Only the RATE should require audio."""
    out = generate_dataset_paired_refs(tmp_path, n=4, duration_s=1.0, fps=25, seed=2)
    manifest = [json.loads(L) for L in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    for m in manifest:
        # Identity fields are recorded once per row (the same for target + ref).
        assert "shape" in m and "color" in m and "center" in m, m
    # And: across rows, identity VARIES (nuisance diversity preserved).
    shapes = {m["shape"] for m in manifest}
    centers = {tuple(m["center"]) for m in manifest}
    assert len(shapes) > 1 or len(centers) > 1, (
        "v2 must preserve cross-row visual diversity; otherwise the LoRA learns "
        "a fixed scene + audio→rate mapping, not a general scene+audio→rate skill"
    )


def test_captions_schema_matches_existing_dataset_pipeline(tmp_path: Path):
    """process_dataset.py expects `video`, `caption`, and (for IC-LoRA) `reference`
    keys in captions.json. Don't break that contract — v2 just populates
    `reference` with a different file than `video`."""
    out = generate_dataset_paired_refs(tmp_path, n=2, duration_s=1.0, fps=25, seed=3)
    rows = json.loads(out.read_text())
    for r in rows:
        for k in ("video", "caption", "reference"):
            assert k in r, f"row missing required key '{k}': {r}"


@pytest.mark.parametrize("n", [1, 5, 20])
def test_n_rows_produces_2n_clips(tmp_path: Path, n: int):
    """No surprise behaviour at edge counts."""
    generate_dataset_paired_refs(tmp_path, n=n, duration_s=1.0, fps=25, seed=4)
    target_clips = list((tmp_path / "clips").glob("*.mp4"))
    ref_clips = list((tmp_path / "references").glob("*.mp4"))
    assert len(target_clips) == n
    assert len(ref_clips) == n

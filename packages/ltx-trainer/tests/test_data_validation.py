"""Tests for the precomputed-dataset validator (ltx_trainer.data_validation).

The validator's whole job is to catch bad training data BEFORE a GPU run, so the
tests build synthetic .pt datasets on disk (CPU, no model) exercising each real
failure mode and assert the validator flags it — and passes clean data.

File naming mirrors PrecomputedDataset's pairing contract:
  latents/latent_N.pt  ↔  conditions/condition_N.pt
  audio_latents/latent_N.pt , reference_latents/latent_N.pt  (same rel name as latents)
"""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_trainer.data_validation import validate_dataset

EMBED_DIM = 16
PROMPT_LEN = 8


def _video(frames: int = 13, h: int = 8, w: int = 8, fps: int = 24, channels: int = 128) -> dict:
    return {
        "latents": torch.randn(channels, frames, h, w),
        "num_frames": frames,
        "height": h,
        "width": w,
        "fps": fps,
    }


def _audio(t: int = 121, channels: int = 8, mel: int = 16) -> dict:
    return {"latents": torch.randn(channels, t, mel), "num_frames": t}


def _conditions(with_audio: bool = True) -> dict:
    c = {
        "video_prompt_embeds": torch.randn(PROMPT_LEN, EMBED_DIM),
        "prompt_attention_mask": torch.ones(PROMPT_LEN, dtype=torch.bool),
    }
    if with_audio:
        c["audio_prompt_embeds"] = torch.randn(PROMPT_LEN, EMBED_DIM)
    return c


def _write_dataset(
    root: Path,
    n: int,
    *,
    with_audio: bool = True,
    with_reference: bool = True,
    n_audio: int | None = None,
    audio_frames=None,            # callable(i)->T or None for default
    cond_with_audio: bool = True,
    video_override=None,          # callable(i)->video dict
) -> None:
    for d in ["latents", "conditions"] + (["reference_latents"] if with_reference else []) + (
        ["audio_latents"] if with_audio else []
    ):
        (root / d).mkdir(parents=True, exist_ok=True)

    for i in range(n):
        v = video_override(i) if video_override else _video()
        torch.save(v, root / "latents" / f"latent_{i}.pt")
        torch.save(_conditions(cond_with_audio), root / "conditions" / f"condition_{i}.pt")
        if with_reference:
            torch.save(_video(frames=3), root / "reference_latents" / f"latent_{i}.pt")

    if with_audio:
        n_aud = n if n_audio is None else n_audio
        for i in range(n_aud):
            t = audio_frames(i) if audio_frames else 121
            torch.save(_audio(t=t), root / "audio_latents" / f"latent_{i}.pt")


def test_clean_dataset_passes(tmp_path):
    _write_dataset(tmp_path, 4, with_audio=True)
    report = validate_dataset(tmp_path, with_audio=True)
    assert report.ok, f"expected clean dataset OK, got errors: {report.all_errors()}"
    assert report.paired_count == 4
    assert report.audio_rate_median is not None


def test_count_mismatch_is_error_not_silent(tmp_path):
    """100 video / 3 audio is the silent-intersection footgun — must be a loud ERROR."""
    _write_dataset(tmp_path, 5, with_audio=True, n_audio=3)
    report = validate_dataset(tmp_path, with_audio=True)
    assert not report.ok
    assert any("intersection" in e.lower() or "differ" in e.lower() for e in report.all_errors())


def test_wrong_audio_shape_flagged(tmp_path):
    _write_dataset(tmp_path, 2, with_audio=True)
    # Clobber one audio file with wrong channel count (4 instead of 8).
    torch.save({"latents": torch.randn(4, 121, 16)}, tmp_path / "audio_latents" / "latent_0.pt")
    report = validate_dataset(tmp_path, with_audio=True)
    assert not report.ok
    assert any("audio channels" in e for e in report.all_errors())


def test_nan_latent_flagged(tmp_path):
    _write_dataset(tmp_path, 2, with_audio=True)
    bad = torch.randn(128, 13, 8, 8)
    bad[0, 0, 0, 0] = float("nan")
    torch.save({"latents": bad, "num_frames": 13, "height": 8, "width": 8, "fps": 24},
               tmp_path / "latents" / "latent_1.pt")
    report = validate_dataset(tmp_path, with_audio=True)
    assert not report.ok
    assert any("NaN" in e or "Inf" in e for e in report.all_errors())


def test_missing_audio_prompt_embeds_flagged(tmp_path):
    """with_audio but conditions lack audio_prompt_embeds — the strategy will refuse."""
    _write_dataset(tmp_path, 2, with_audio=True, cond_with_audio=False)
    report = validate_dataset(tmp_path, with_audio=True)
    assert not report.ok
    assert any("audio_prompt_embeds" in e for e in report.all_errors())


def test_audio_video_misalignment_warns(tmp_path):
    """One sample whose audio length doesn't match its video duration → outlier warning."""
    # 4 aligned samples (T=121) + force sample 3 to a wildly different audio length.
    def audio_frames(i: int) -> int:
        return 240 if i == 3 else 121
    _write_dataset(tmp_path, 4, with_audio=True, audio_frames=audio_frames)
    report = validate_dataset(tmp_path, with_audio=True, alignment_tolerance=0.10)
    assert report.ok  # shapes are valid; misalignment is a WARNING, not an error
    assert any("deviates" in w and "rate" in w for w in report.all_warnings())


def test_video_only_dataset_ok_without_audio(tmp_path):
    """with_audio=False must validate a video-only dataset (no audio dir required)."""
    _write_dataset(tmp_path, 3, with_audio=False)
    report = validate_dataset(tmp_path, with_audio=False)
    assert report.ok, f"errors: {report.all_errors()}"
    assert report.paired_count == 3

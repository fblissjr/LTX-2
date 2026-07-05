"""Tests for the reference-audio precompute helpers (the reference_audio_latents channel).

Pure/stubbed — no model, no GPU. They lock (a) the on-disk pairing convention
(a reference latent lands at the SAME relative path as the clip's video latent so
PrecomputedDataset pairs them) and (b) the latent-dict format, which must match the
audio_latents the AV precompute writes so AudioReferenceStrategy consumes it identically.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_trainer.reference_audio import (
    encode_reference_waveform,
    ensure_audio_channels,
    reference_output_path,
)


def test_reference_output_path_mirrors_video_latent_rel():
    out = reference_output_path("clips/clip_0001.mp4", Path("/data/reference_audio_latents"))
    assert out == Path("/data/reference_audio_latents/clips/clip_0001.pt")


def test_reference_output_path_absolute_with_data_root():
    out = reference_output_path(
        "/data/clips/clip_0007.mp4", Path("/data/reference_audio_latents"), data_root="/data"
    )
    assert out == Path("/data/reference_audio_latents/clips/clip_0007.pt")


class _StubProcessor:
    def waveform_to_mel(self, audio):  # noqa: ARG002 - encoder stub ignores the mel content
        return torch.zeros(1, 64, 100)


class _StubEncoder(torch.nn.Module):
    """Stands in for the audio VAE encoder: a param (so device/dtype resolve) +
    a fixed [B, C=8, T, F=16] output."""

    def __init__(self, t: int = 50):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.t = t

    def forward(self, mel):  # noqa: ARG002
        return torch.zeros(1, 8, self.t, 16)


def test_ensure_audio_channels_mono_to_stereo_is_dual_mono():
    mono = torch.randn(1, 1, 100)  # [batch, channels=1, samples]
    out = ensure_audio_channels(mono, 2)
    assert out.shape == (1, 2, 100)
    assert torch.equal(out[:, 0], out[:, 1])  # dual-mono: identical L/R


def test_ensure_audio_channels_noop_when_already_matching():
    stereo = torch.randn(1, 2, 100)
    out = ensure_audio_channels(stereo, 2)
    assert out.shape == (1, 2, 100)
    assert torch.equal(out, stereo)


def test_ensure_audio_channels_downmixes_when_too_many():
    quad = torch.randn(1, 4, 100)
    out = ensure_audio_channels(quad, 2)
    assert out.shape == (1, 2, 100)
    assert torch.equal(out[:, 0], out[:, 1])  # mean-downmix, then duplicated


def test_encode_reference_waveform_format_matches_audio_latents():
    out = encode_reference_waveform(
        _StubEncoder(t=50), _StubProcessor(), torch.randn(1, 32000), sampling_rate=16000  # 2.0s mono @16k
    )
    assert set(out) == {"latents", "num_time_steps", "frequency_bins", "duration"}
    assert tuple(out["latents"].shape) == (8, 50, 16)  # [C, T, F], batch squeezed
    assert out["num_time_steps"] == 50
    assert out["frequency_bins"] == 16
    assert abs(out["duration"] - 2.0) < 1e-6


# --- channel augmentation (the anti-shortcut for session fingerprints) ----------


def _aug(waveform, seed: int):
    from ltx_trainer.reference_audio import augment_reference_waveform

    gen = torch.Generator().manual_seed(seed)
    return augment_reference_waveform(waveform, 16000, generator=gen)


def test_channel_aug_preserves_shape_and_stays_finite():
    wav = torch.randn(1, 16000) * 0.1
    out = _aug(wav, seed=0)
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()
    # bounded: EQ/gain jitter must not explode the signal
    assert out.abs().max() < wav.abs().max() * 4 + 1e-3


def test_channel_aug_is_deterministic_per_seed_and_varies_across_seeds():
    """Re-running a precompute must reproduce byte-identical variants (the eval and the
    training data must not drift between runs), while different variant seeds must produce
    genuinely different channel renditions (otherwise the K-stack is K copies and the
    anti-shortcut does nothing)."""
    wav = torch.randn(2, 16000) * 0.1
    a1, a2 = _aug(wav, seed=7), _aug(wav, seed=7)
    assert torch.equal(a1, a2)
    b = _aug(wav, seed=8)
    assert not torch.equal(a1, b)


def test_channel_aug_actually_changes_the_signal():
    wav = torch.randn(1, 16000) * 0.1
    out = _aug(wav, seed=0)
    assert not torch.allclose(out, wav, atol=1e-4)

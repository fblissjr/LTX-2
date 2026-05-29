"""Tests for the reference-audio precompute helpers (the reference_audio_latents channel).

Pure/stubbed — no model, no GPU. They lock (a) the on-disk pairing convention
(a reference latent lands at the SAME relative path as the clip's video latent so
PrecomputedDataset pairs them) and (b) the latent-dict format, which must match the
audio_latents the AV precompute writes so AudioReferenceStrategy consumes it identically.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ltx_trainer.reference_audio import encode_reference_waveform, reference_output_path


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


def test_encode_reference_waveform_format_matches_audio_latents():
    out = encode_reference_waveform(
        _StubEncoder(t=50), _StubProcessor(), torch.randn(1, 32000), sampling_rate=16000  # 2.0s mono @16k
    )
    assert set(out) == {"latents", "num_time_steps", "frequency_bins", "duration"}
    assert tuple(out["latents"].shape) == (8, 50, 16)  # [C, T, F], batch squeezed
    assert out["num_time_steps"] == 50
    assert out["frequency_bins"] == 16
    assert abs(out["duration"] - 2.0) < 1e-6

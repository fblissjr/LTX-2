"""Trainer must not load + must offload validation-only codecs.

Background: training reads pre-encoded latents (video VAE + Gemma + audio VAE
all run UPFRONT in process_dataset.py). At train time, the audio VAE decoder
and vocoder are touched ONLY by validation sampling. Leaving them loaded +
GPU-resident through training wastes ~700 MB – 1 GB — fatal on a tight 24 GB
4090 where the int8 22B base alone is ~22 GB.

Two contracts checked here:

  1. `_validation_will_run(cfg)` — gates codec loading. False when validation
     is disabled (no interval), even if the training strategy `requires_audio`.

  2. `_offload_frozen_codecs(trainer)` — moves the loaded video VAE, audio
     VAE, and vocoder to CPU symmetrically. The old code offloaded only the
     video VAE; the audio side leaked GPU memory.
"""

from __future__ import annotations

from types import SimpleNamespace

from ltx_trainer.trainer import _offload_frozen_codecs, _validation_will_run


def test_validation_off_blocks_codec_load():
    """interval=None / 0 ⇒ validation won't run ⇒ codecs are pure waste."""
    assert _validation_will_run(SimpleNamespace(interval=None)) is False
    assert _validation_will_run(SimpleNamespace(interval=0)) is False
    assert _validation_will_run(SimpleNamespace(interval=100)) is True


def test_offload_moves_audio_codecs_to_cpu():
    """The old code offloaded the video VAE but left audio_vae + vocoder
    resident on GPU. This is the bug fix: symmetric offload."""
    moved: list[tuple[str, str]] = []

    class _FakeMod:
        def __init__(self, name):
            self._name = name

        def to(self, device):
            moved.append((self._name, device))
            return self

    trainer = SimpleNamespace(
        _vae_decoder=_FakeMod("video_dec"),
        _vae_encoder=_FakeMod("video_enc"),
        _audio_vae=_FakeMod("audio_dec"),
        _vocoder=_FakeMod("vocoder"),
    )
    _offload_frozen_codecs(trainer)
    assert {n for n, _ in moved} == {"video_dec", "video_enc", "audio_dec", "vocoder"}
    assert all(d == "cpu" for _, d in moved)


def test_offload_skips_none_codecs():
    """Codecs may be None when their load was gated off — offload must not crash."""
    trainer = SimpleNamespace(_vae_decoder=None, _vae_encoder=None, _audio_vae=None, _vocoder=None)
    _offload_frozen_codecs(trainer)  # no raise

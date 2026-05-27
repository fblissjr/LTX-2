"""Unit tests for audio support in the video-to-video (IC-LoRA) training strategy.

These tests prove the PLUMBING of audio-guided IC-LoRA training — that the
strategy builds the right Modality / loss-mask / target shapes for each
`audio_mode` — using synthetic tensors. They do NOT prove the model learns
anything (that needs a real train run). No model, no GPU, no checkpoint.

The three audio modes (per internal/trainer_audio_iclora_plan.md):
  condition    — audio = clean context (the guide), video = noised target,
                 loss on video only. "Audio guides video generation." PRIMARY.
  generate     — audio noised + audio loss (joint AV gen; mirrors text_to_video).
  continuation — audio prefix clean + tail noised (loss on tail), video all clean
                 (no video loss). Training analog of the video->audio inversion.

First tests in packages/ltx-trainer — see internal/trainer_audio_iclora_plan.md
section "TDD".
"""

from __future__ import annotations

import pytest
import torch

from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.training_strategies.video_to_video import (
    VideoToVideoConfig,
    VideoToVideoStrategy,
)

# Audio latent layout: [B, C=8, T, F=16]; patchify -> [B, T, C*F=128].
AUDIO_CHANNELS = 8
AUDIO_MEL_BINS = 16
EMBED_DIM = 32
PROMPT_LEN = 8


class _FixedSigmaSampler:
    """Deterministic timestep sampler: every sample gets the same sigma.

    Lets tests assert exact per-token timesteps (0.0 for clean/conditioning
    tokens, SIGMA for noised/target tokens).
    """

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def sample_for(self, latents: torch.Tensor) -> torch.Tensor:
        batch = latents.shape[0]
        return torch.full((batch,), self.sigma, dtype=latents.dtype, device=latents.device)


def _video_latents(batch: int, frames: int, h: int, w: int) -> torch.Tensor:
    return torch.randn(batch, 128, frames, h, w)


def _audio_latents(batch: int, t: int) -> torch.Tensor:
    return torch.randn(batch, AUDIO_CHANNELS, t, AUDIO_MEL_BINS)


def _make_batch(
    *,
    batch: int = 1,
    frames: int = 5,
    h: int = 8,
    w: int = 8,
    ref_frames: int = 2,
    audio_t: int | None = None,
    fps: float = 24.0,
) -> dict:
    b: dict = {
        "latents": {
            "latents": _video_latents(batch, frames, h, w),
            "num_frames": torch.tensor([frames] * batch),
            "height": torch.tensor([h] * batch),
            "width": torch.tensor([w] * batch),
            "fps": torch.tensor([fps] * batch),
        },
        "ref_latents": {
            "latents": _video_latents(batch, ref_frames, h, w),
            "num_frames": torch.tensor([ref_frames] * batch),
            "height": torch.tensor([h] * batch),
            "width": torch.tensor([w] * batch),
        },
        "conditions": {
            "video_prompt_embeds": torch.randn(batch, PROMPT_LEN, EMBED_DIM),
            "audio_prompt_embeds": torch.randn(batch, PROMPT_LEN, EMBED_DIM),
            "prompt_attention_mask": torch.ones(batch, PROMPT_LEN, dtype=torch.bool),
        },
    }
    if audio_t is not None:
        b["audio_latents"] = {
            "latents": _audio_latents(batch, audio_t),
            "num_frames": torch.tensor([audio_t] * batch),
        }
    return b


def _strategy(**cfg) -> VideoToVideoStrategy:
    return VideoToVideoStrategy(VideoToVideoConfig(**cfg))


# --- Test 5/6: config-level plumbing -----------------------------------------


def test_requires_audio_tracks_with_audio():
    assert _strategy().requires_audio is False
    assert _strategy(with_audio=True).requires_audio is True


def test_get_data_sources_includes_audio_when_enabled():
    assert "audio_latents" not in _strategy().get_data_sources().values()
    sources = _strategy(with_audio=True).get_data_sources()
    assert sources.get("audio_latents") == "audio_latents"
    # video reference sources still present
    assert "ref_latents" in sources.values()


# --- Test 1: with_audio=False locks current behavior --------------------------


def test_no_audio_is_video_only():
    strat = _strategy(with_audio=False)
    out = strat.prepare_training_inputs(_make_batch(), _FixedSigmaSampler())
    assert isinstance(out, ModelInputs)
    assert out.audio is None
    assert out.audio_targets is None
    assert out.audio_loss_mask is None


# --- Test 2: generate mode ----------------------------------------------------


def test_generate_mode_audio_noised_and_in_loss():
    strat = _strategy(with_audio=True, audio_mode="generate", first_frame_conditioning_p=0.0)
    out = strat.prepare_training_inputs(_make_batch(audio_t=20), _FixedSigmaSampler(0.5))
    assert out.audio is not None
    assert out.audio_targets is not None
    # All audio tokens are noised targets -> all in the loss mask, all timesteps = sigma.
    assert out.audio_loss_mask.all()
    assert torch.allclose(out.audio.timesteps, torch.full_like(out.audio.timesteps, 0.5))


# --- Test 3: condition mode (audio guides video) ------------------------------


def test_condition_mode_audio_clean_not_in_loss():
    strat = _strategy(with_audio=True, audio_mode="condition", first_frame_conditioning_p=0.0)
    batch = _make_batch(audio_t=20)
    clean_audio = strat._audio_patchifier.patchify(batch["audio_latents"]["latents"].clone())
    out = strat.prepare_training_inputs(batch, _FixedSigmaSampler(0.5))

    assert out.audio is not None
    # Audio is clean context: timesteps all 0, latent equals the un-noised input.
    assert torch.allclose(out.audio.timesteps, torch.zeros_like(out.audio.timesteps))
    assert torch.allclose(out.audio.latent, clean_audio)
    # Audio contributes NO loss (it's the guide, not a target).
    assert out.audio_targets is None
    assert not out.audio_loss_mask.any()
    # Video target loss is unchanged: with first_frame_conditioning_p=0, every
    # target token is in the loss mask (ref tokens excluded).
    ref_len = out.ref_seq_len
    assert out.video_loss_mask[:, ref_len:].all()
    assert not out.video_loss_mask[:, :ref_len].any()


# --- Test 4: continuation mode ------------------------------------------------


def test_continuation_mode_prefix_clean_video_frozen():
    strat = _strategy(
        with_audio=True, audio_mode="continuation", audio_prefix_seconds=2.0,
        first_frame_conditioning_p=0.0,
    )
    out = strat.prepare_training_inputs(_make_batch(frames=13, audio_t=121, fps=24.0), _FixedSigmaSampler(0.5))
    assert out.audio is not None

    n_clean = VideoToVideoStrategy._audio_prefix_frame_count(
        audio_seq_len=121, num_video_latent_frames=13, fps=24.0, prefix_seconds=2.0
    )
    assert 0 < n_clean < 121
    # Audio: prefix clean (timestep 0, no loss), tail noised (timestep sigma, in loss).
    assert torch.allclose(out.audio.timesteps[:, :n_clean], torch.zeros(1, n_clean))
    assert torch.allclose(out.audio.timesteps[:, n_clean:], torch.full((1, 121 - n_clean), 0.5))
    assert not out.audio_loss_mask[:, :n_clean].any()
    assert out.audio_loss_mask[:, n_clean:].all()
    assert out.audio_targets is not None
    # Video is fully frozen context: no video loss anywhere.
    assert not out.video_loss_mask.any()


# --- Test 7: seconds -> audio-latent-frame math -------------------------------


def test_continuation_full_prefix_raises():
    """A prefix covering the whole clip leaves nothing to generate AND freezes the
    video -> a silent zero-gradient step. Fail loud at input-prep instead."""
    strat = _strategy(with_audio=True, audio_mode="continuation", audio_prefix_seconds=999.0)
    with pytest.raises(ValueError, match="prefix"):
        strat.prepare_training_inputs(_make_batch(frames=13, audio_t=121, fps=24.0), _FixedSigmaSampler())


def test_with_audio_missing_audio_prompt_embeds_raises():
    """with_audio=True but no audio text-encoder embeddings -> clear error, not a
    deep None-context failure inside the model."""
    strat = _strategy(with_audio=True, audio_mode="condition")
    batch = _make_batch(audio_t=20)
    del batch["conditions"]["audio_prompt_embeds"]
    with pytest.raises(ValueError, match="audio_prompt_embeds"):
        strat.prepare_training_inputs(batch, _FixedSigmaSampler())


# --- compute_loss (the extracted _masked_velocity_loss + two-stream sum) ---


def _loss_inputs(*, ref_seq_len, video_loss_mask, video_targets, audio_targets=None, audio_loss_mask=None):
    # compute_loss reads only targets/masks/ref_seq_len, not the Modality objects.
    return ModelInputs(
        video=None, audio=None,
        video_targets=video_targets, audio_targets=audio_targets,
        video_loss_mask=video_loss_mask, audio_loss_mask=audio_loss_mask,
        ref_seq_len=ref_seq_len,
    )


def test_compute_loss_video_only_masked_value():
    """video-only: masked MSE on the target portion (ref tokens excluded)."""
    strat = _strategy(with_audio=False)
    ref, tgt, c = 2, 3, 4
    video_pred = torch.cat([torch.zeros(1, ref, c), torch.ones(1, tgt, c)], dim=1)  # target=1
    video_targets = torch.zeros(1, tgt, c)
    video_loss_mask = torch.tensor([[False, False, True, True, True]])
    loss = strat.compute_loss(video_pred, None, _loss_inputs(
        ref_seq_len=ref, video_loss_mask=video_loss_mask, video_targets=video_targets))
    assert torch.allclose(loss, torch.tensor([1.0]))  # mean((1-0)^2) over target


def test_compute_loss_adds_audio_term_when_targets_present():
    strat = _strategy(with_audio=True, audio_mode="generate")
    ref, tgt, c = 1, 2, 4
    video_pred = torch.cat([torch.zeros(1, ref, c), torch.ones(1, tgt, c)], dim=1)
    inputs = _loss_inputs(
        ref_seq_len=ref, video_loss_mask=torch.tensor([[False, True, True]]),
        video_targets=torch.zeros(1, tgt, c),
        audio_targets=torch.zeros(1, 2, c), audio_loss_mask=torch.tensor([[True, True]]),
    )
    audio_pred = torch.full((1, 2, c), 2.0)  # audio err = 4
    loss = strat.compute_loss(video_pred, audio_pred, inputs)
    assert torch.allclose(loss, torch.tensor([5.0]))  # 1 (video) + 4 (audio)


def test_compute_loss_skips_audio_when_targets_none():
    """condition mode: audio_targets is None → no audio term (video loss only)."""
    strat = _strategy(with_audio=True, audio_mode="condition")
    ref, tgt, c = 1, 2, 4
    video_pred = torch.cat([torch.zeros(1, ref, c), torch.ones(1, tgt, c)], dim=1)
    loss = strat.compute_loss(video_pred, torch.full((1, 2, c), 9.0), _loss_inputs(
        ref_seq_len=ref, video_loss_mask=torch.tensor([[False, True, True]]),
        video_targets=torch.zeros(1, tgt, c), audio_targets=None,
        audio_loss_mask=torch.tensor([[False, False]])))
    assert torch.allclose(loss, torch.tensor([1.0]))


def test_compute_loss_fully_masked_stream_is_zero_not_nan():
    """continuation freezes the video → empty video loss mask → 0, not NaN."""
    strat = _strategy(with_audio=False)
    ref, tgt, c = 0, 3, 4
    video_pred = torch.ones(1, tgt, c)
    loss = strat.compute_loss(video_pred, None, _loss_inputs(
        ref_seq_len=ref, video_loss_mask=torch.zeros(1, tgt, dtype=torch.bool),
        video_targets=torch.zeros(1, tgt, c)))
    assert torch.isfinite(loss).all() and torch.allclose(loss, torch.tensor([0.0]))


@pytest.mark.parametrize(
    ("audio_seq_len", "num_latent_frames", "fps", "prefix_seconds", "expected"),
    [
        (121, 13, 24.0, 0.0, 0),       # zero prefix -> no clean tokens
        (121, 13, 24.0, 999.0, 121),   # prefix beyond clip -> clamp to full
        (121, 13, 24.0, 2.0, 60),      # pixel=(13-1)*8+1=97, dur=97/24, rate=121/dur, *2 -> 60
        (100, 13, 24.0, 0.0, 0),
    ],
)
def test_audio_prefix_frame_count(audio_seq_len, num_latent_frames, fps, prefix_seconds, expected):
    n = VideoToVideoStrategy._audio_prefix_frame_count(
        audio_seq_len=audio_seq_len,
        num_video_latent_frames=num_latent_frames,
        fps=fps,
        prefix_seconds=prefix_seconds,
    )
    assert n == expected

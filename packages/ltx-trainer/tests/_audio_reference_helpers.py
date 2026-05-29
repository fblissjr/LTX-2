"""Shared synthetic-tensor fixtures for the audio_reference strategy tests.

Both the plumbing tests (test_audio_reference_strategy.py) and the
real-architecture forward/backward smoke (test_audio_reference_forward.py) need
the same deterministic sampler, synthetic-latent builders, and latent-layout
constants. They live here so the two suites can't drift apart.

Audio latent layout: [B, C=8, T, F=16]; patchify -> [B, T, C*F=128].
"""

from __future__ import annotations

import torch

AUDIO_CHANNELS = 8
AUDIO_MEL_BINS = 16
EMBED_DIM = 32
PROMPT_LEN = 8


class FixedSigmaSampler:
    """Deterministic timestep sampler: every sample gets the same sigma.

    Lets tests assert exact per-token timesteps (0.0 for clean/conditioning
    tokens, SIGMA for noised/target tokens).
    """

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def sample_for(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full((latents.shape[0],), self.sigma, dtype=latents.dtype, device=latents.device)


def video_latents(batch: int, frames: int, h: int, w: int) -> torch.Tensor:
    return torch.randn(batch, 128, frames, h, w)


def audio_latents(batch: int, t: int) -> torch.Tensor:
    return torch.randn(batch, AUDIO_CHANNELS, t, AUDIO_MEL_BINS)


def make_batch(
    *,
    batch: int = 1,
    frames: int = 5,
    h: int = 8,
    w: int = 8,
    audio_t: int = 20,
    ref_audio_t: int = 12,
    fps: float = 24.0,
    prompt_mask_dtype: torch.dtype = torch.bool,
) -> dict:
    """Synthetic batch for the audio_reference strategy (target + reference audio).

    ``prompt_mask_dtype`` defaults to bool for the plumbing tests; the real-arch
    forward smoke passes int32 because the model's _prepare_attention_mask does
    (mask-1)*max, which rejects bool (the real trainer feeds a converted mask).
    """
    return {
        "latents": {
            "latents": video_latents(batch, frames, h, w),
            "num_frames": torch.tensor([frames] * batch),
            "height": torch.tensor([h] * batch),
            "width": torch.tensor([w] * batch),
            "fps": torch.tensor([fps] * batch),
        },
        "audio_latents": {
            "latents": audio_latents(batch, audio_t),
            "num_frames": torch.tensor([audio_t] * batch),
        },
        "reference_audio_latents": {
            "latents": audio_latents(batch, ref_audio_t),
            "num_frames": torch.tensor([ref_audio_t] * batch),
        },
        "conditions": {
            "video_prompt_embeds": torch.randn(batch, PROMPT_LEN, EMBED_DIM),
            "audio_prompt_embeds": torch.randn(batch, PROMPT_LEN, EMBED_DIM),
            "prompt_attention_mask": torch.ones(batch, PROMPT_LEN, dtype=prompt_mask_dtype),
        },
    }

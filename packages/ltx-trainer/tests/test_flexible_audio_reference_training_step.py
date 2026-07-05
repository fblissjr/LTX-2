"""End-to-end (GPU-free) training-step check for the audio-reference FlexibleStrategy.

Upstream ships no FlexibleStrategy tests, and the parity test only checks reference POSITIONS.
This drives the full training-input path on synthetic latents — prepare_training_inputs ->
compute_loss — so the audio-reference recipe (audio-only, is_generated, a single lipdub-negative
reference condition) is proven to actually execute and produce a finite loss, and the reference is
shown to materially enter the model input (it extends the audio sequence at negative RoPE). This is
the GPU-free twin of "the reference is load-bearing"; a real forward/backward on the 22B is the
separate feasibility check.
"""

import torch

from ltx_trainer.timestep_samplers import UniformTimestepSampler
from ltx_trainer.training_strategies.flexible import (
    FlexibleStrategy,
    FlexibleStrategyConfig,
    ModalityConfig,
    ReferenceConditionConfig,
)

_AUDIO_CHANNELS = 8
_AUDIO_MEL_BINS = 16
_PATCH_DIM = _AUDIO_CHANNELS * _AUDIO_MEL_BINS  # 128
_TARGET_FRAMES = 8
_REF_FRAMES = 6
_BATCH = 1
_PROMPT_LEN = 12
_PROMPT_DIM = 32  # arbitrary — context is only stored, not used by compute_loss


def _audio_reference_strategy(with_reference: bool) -> FlexibleStrategy:
    conditions = []
    if with_reference:
        conditions.append(
            ReferenceConditionConfig(
                latents_dir="reference_audio_latents",
                audio_positions_mode="lipdub_negative",
                probability=1.0,
            )
        )
    config = FlexibleStrategyConfig(
        name="flexible",
        audio=ModalityConfig(is_generated=True, latents_dir="audio_latents", conditions=conditions),
    )
    return FlexibleStrategy(config)


def _synthetic_batch(with_reference: bool) -> dict:
    batch = {
        "audio_latents": {"latents": torch.randn(_BATCH, _AUDIO_CHANNELS, _TARGET_FRAMES, _AUDIO_MEL_BINS)},
        "conditions": {
            "audio_prompt_embeds": torch.randn(_BATCH, _PROMPT_LEN, _PROMPT_DIM),
            "prompt_attention_mask": torch.ones(_BATCH, _PROMPT_LEN, dtype=torch.bool),
        },
    }
    if with_reference:
        batch["reference_audio_latents"] = {
            "latents": torch.randn(_BATCH, _AUDIO_CHANNELS, _REF_FRAMES, _AUDIO_MEL_BINS)
        }
    return batch


def test_audio_reference_training_step_runs_and_loss_is_finite():
    strategy = _audio_reference_strategy(with_reference=True)
    inputs = strategy.prepare_training_inputs(_synthetic_batch(with_reference=True), UniformTimestepSampler())

    # Audio-only: no video modality, audio present and enabled.
    assert inputs.video is None
    assert inputs.audio is not None and inputs.audio.enabled

    # Reference is PREPENDED: combined audio sequence = reference + target tokens.
    combined_seq = inputs.audio.latent.shape[1]
    assert combined_seq == _REF_FRAMES + _TARGET_FRAMES
    assert inputs.audio_targets.shape[1] == _TARGET_FRAMES  # loss only on the target portion

    # The prepended reference tokens sit at strictly-negative RoPE positions (lipdub convention);
    # the target tokens are non-negative.
    positions = inputs.audio.positions  # [B, 1, combined_seq, 2]
    ref_positions = positions[:, :, :_REF_FRAMES, :]
    target_positions = positions[:, :, _REF_FRAMES:, :]
    assert (ref_positions < 0).all()
    assert (target_positions >= 0).all()

    # compute_loss runs over the combined-length prediction and returns a finite per-element loss.
    audio_pred = torch.randn(_BATCH, combined_seq, _PATCH_DIM)
    loss = strategy.compute_loss(video_pred=None, audio_pred=audio_pred, inputs=inputs)
    assert loss.shape == (_BATCH,)
    assert torch.isfinite(loss).all()


def test_reference_condition_materially_extends_the_audio_input():
    """Load-bearing at the input level: removing the reference shortens the audio sequence by
    exactly the reference token count (the reference is not decorative — it enters the model)."""
    with_ref = _audio_reference_strategy(with_reference=True).prepare_training_inputs(
        _synthetic_batch(with_reference=True), UniformTimestepSampler()
    )
    without_ref = _audio_reference_strategy(with_reference=False).prepare_training_inputs(
        _synthetic_batch(with_reference=False), UniformTimestepSampler()
    )
    assert with_ref.audio.latent.shape[1] - without_ref.audio.latent.shape[1] == _REF_FRAMES

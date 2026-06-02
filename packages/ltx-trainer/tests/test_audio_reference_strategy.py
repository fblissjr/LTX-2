"""Unit tests for the audio-reference IC-LoRA training strategy.

Plumbing tests (synthetic tensors — no model, no GPU, no checkpoint). They prove
the strategy builds the right Modality / loss-mask / target shapes for the
"reference audio (an attribute exemplar, e.g. a pitch) -> generate AV at that
attribute" task:

  - Reference audio is appended to the audio stream as CLEAN context (the IC
    reference), mirroring ltx_core's AudioConditionByReferenceLatent (target
    tokens lead, reference tokens trail, reference held clean).
  - The target is AV: video is the generated target (in the loss) and the
    target-audio portion is noised + in the loss.
  - The reference-audio tokens are excluded from the loss.

They do NOT prove the model learns anything (that needs a real train run).
"""

from __future__ import annotations

import pytest
import torch
from _audio_reference_helpers import FixedSigmaSampler, make_batch

from ltx_trainer.training_strategies.audio_reference import (
    REFERENCE_ROPE_GAP,
    AudioReferenceConfig,
    AudioReferenceStrategy,
)
from ltx_trainer.training_strategies.base_strategy import ModelInputs


def _strategy(**cfg) -> AudioReferenceStrategy:
    return AudioReferenceStrategy(AudioReferenceConfig(**cfg))


# --- config-level plumbing ----------------------------------------------------


def test_requires_audio_always_true():
    assert _strategy().requires_audio is True


def test_data_sources_include_target_and_reference_audio():
    sources = _strategy().get_data_sources()
    assert sources.get("latents") == "latents"
    assert sources.get("conditions") == "conditions"
    assert sources.get("audio_latents") == "audio_latents"
    assert sources.get("reference_audio_latents") == "reference_audio_latents"


# --- the core: audio = [target (noised) | reference (clean)] -------------------


def test_audio_is_target_then_clean_reference():
    strat = _strategy(first_frame_conditioning_p=0.0)
    batch = make_batch(audio_t=20, ref_audio_t=12)
    ref_clean = strat._audio_patchifier.patchify(batch["reference_audio_latents"]["latents"].clone())
    out = strat.prepare_training_inputs(batch, FixedSigmaSampler(0.5))

    assert isinstance(out, ModelInputs)
    assert out.audio is not None
    # Audio sequence length = target (20) + reference (12).
    assert out.audio.latent.shape[1] == 20 + 12

    # Target portion (first 20): noised -> timesteps = sigma, in the loss.
    assert torch.allclose(out.audio.timesteps[:, :20], torch.full((1, 20), 0.5))
    assert out.audio_loss_mask[:, :20].all()

    # Reference portion (last 12): clean -> timesteps 0, NOT in the loss, and the
    # latent equals the (un-noised) reference patchified input.
    assert torch.allclose(out.audio.timesteps[:, 20:], torch.zeros(1, 12))
    assert not out.audio_loss_mask[:, 20:].any()
    assert torch.allclose(out.audio.latent[:, 20:, :], ref_clean)


def test_audio_targets_full_length_zero_on_reference():
    """audio_targets spans the full audio sequence; reference rows are zero (masked)."""
    strat = _strategy(first_frame_conditioning_p=0.0)
    out = strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=12), FixedSigmaSampler(0.5))
    assert out.audio_targets is not None
    assert out.audio_targets.shape[1] == 20 + 12
    # Reference rows carry no target signal.
    assert torch.allclose(out.audio_targets[:, 20:, :], torch.zeros_like(out.audio_targets[:, 20:, :]))


def test_reference_positions_match_inference_negative_convention():
    """Reference audio tokens sit at strictly-negative positions ending JUST below the
    target timeline's 0 — the exact convention inference uses
    (ltx_pipelines.lipdub.patchify_lipdub_audio_reference_latent: shift by the reference
    end-bound + 0.04). Train/inference must agree on this offset, else the LoRA sees a
    different reference<->target geometry at generation time. The small-gap assertion
    guards against an arbitrary large offset (e.g. the old -(max+1.0))."""
    strat = _strategy(first_frame_conditioning_p=0.0)
    out = strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=12), FixedSigmaSampler())
    positions = out.audio.positions  # [B, 1, T_tgt + T_ref, 2]
    assert positions.shape[2] == 20 + 12
    ref_time = positions[:, 0, 20:, :]
    assert (ref_time < 0).all()  # entirely out of the target timeline
    # Pin the LITERAL gap, not the REFERENCE_ROPE_GAP constant: asserting against the
    # constant would be tautological (the computed offset IS -constant), so it would pass
    # for any value. The literal locks the actual train<->inference contract — inference
    # subtracts the same 0.04 gap, so any drift in the constant breaks generation AND
    # these assertions. The first line ties the constant to the literal the mirror requires.
    assert REFERENCE_ROPE_GAP == 0.04
    assert ref_time.max().item() == pytest.approx(-0.04)


def test_reference_negative_offset_holds_across_lengths():
    """The offset is computed from the reference's OWN length, so a shorter or longer reference
    must STILL end exactly -REFERENCE_ROPE_GAP below 0. This is the invariant the inference-side
    reference-window / multi-slice feature relies on: trimming changes the ref token count but must
    not change the train<->inference geometry. Locks it across lengths, not just the fixed case above."""
    strat = _strategy(first_frame_conditioning_p=0.0)
    for ref_t in (4, 12, 30, 60):
        out = strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=ref_t), FixedSigmaSampler())
        positions = out.audio.positions
        assert positions.shape[2] == 20 + ref_t, f"ref_t={ref_t}: wrong seq length"
        ref_time = positions[:, 0, 20:, :]
        assert (ref_time < 0).all(), f"ref_t={ref_t}: reference not entirely negative"
        assert ref_time.max().item() == pytest.approx(-REFERENCE_ROPE_GAP), f"ref_t={ref_t}: end-bound drifted"


# --- the AV target: video is generated, in the loss ---------------------------


def test_video_is_generated_target_in_loss():
    strat = _strategy(first_frame_conditioning_p=0.0)
    out = strat.prepare_training_inputs(make_batch(), FixedSigmaSampler(0.5))
    assert out.video is not None and out.video.enabled
    assert out.video_targets is not None
    # No video reference here -> every video token is a generated target in the loss.
    assert out.video_loss_mask.all()
    assert out.ref_seq_len in (None, 0)


# --- compute_loss: video target + masked target-audio (reference excluded) ----


def test_compute_loss_sums_video_and_target_audio_only():
    strat = _strategy()
    # video: 3 target tokens, pred=1 target=0 -> video loss 1.0
    video_pred = torch.ones(1, 3, 4)
    video_targets = torch.zeros(1, 3, 4)
    video_loss_mask = torch.ones(1, 3, dtype=torch.bool)
    # audio: 2 target + 2 reference. target err=4 (pred 2, target 0), reference masked.
    audio_pred = torch.full((1, 4, 4), 2.0)
    audio_targets = torch.zeros(1, 4, 4)
    audio_loss_mask = torch.tensor([[True, True, False, False]])
    inputs = ModelInputs(
        video=None,
        audio=None,
        video_targets=video_targets,
        audio_targets=audio_targets,
        video_loss_mask=video_loss_mask,
        audio_loss_mask=audio_loss_mask,
        ref_seq_len=None,
    )
    loss = strat.compute_loss(video_pred, audio_pred, inputs)
    assert torch.allclose(loss, torch.tensor([5.0]))  # 1 (video) + 4 (target audio)


def test_missing_reference_audio_raises():
    strat = _strategy()
    batch = make_batch()
    del batch["reference_audio_latents"]
    with pytest.raises((KeyError, ValueError)):
        strat.prepare_training_inputs(batch, FixedSigmaSampler())


# --- reference dropout (trains the unconditional path for CFG-on-reference) ----


def test_reference_dropout_defaults_to_zero():
    assert AudioReferenceConfig().reference_dropout_p == 0.0


def test_reference_dropout_zero_always_appends_reference():
    strat = _strategy(first_frame_conditioning_p=0.0, reference_dropout_p=0.0)
    out = strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=12), FixedSigmaSampler(0.5))
    assert out.audio.latent.shape[1] == 20 + 12              # target + reference
    assert out.audio.positions[:, 0, 20:, 1].max() < 0       # reference at negative positions


def test_reference_dropout_one_drops_reference_unconditional():
    # p=1.0 -> the no-reference (unconditional) path: audio is target-only, all in the loss,
    # no negative reference positions. This is the branch CFG-on-reference extrapolates from.
    strat = _strategy(first_frame_conditioning_p=0.0, reference_dropout_p=1.0)
    out = strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=12), FixedSigmaSampler(0.5))
    assert out.audio.latent.shape[1] == 20                   # target only, reference dropped
    assert out.audio_loss_mask.shape[1] == 20
    assert out.audio_loss_mask.all()                         # all target tokens stay in the loss
    assert out.audio_targets.shape[1] == 20
    assert (out.audio.positions[:, 0, :, 1] >= 0).all()      # no negative reference positions


def test_reference_dropout_is_stochastic():
    strat = _strategy(first_frame_conditioning_p=0.0, reference_dropout_p=0.5)
    torch.manual_seed(0)
    seqlens = {
        strat.prepare_training_inputs(make_batch(audio_t=20, ref_audio_t=12), FixedSigmaSampler(0.5)).audio.latent.shape[1]
        for _ in range(40)
    }
    assert seqlens == {20, 32}   # both the dropped (20) and the kept (20+12) paths occur

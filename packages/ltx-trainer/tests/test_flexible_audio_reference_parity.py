"""Train/inference RoPE parity for the ``lipdub_negative`` audio-reference convention.

The FlexibleStrategy ``ReferenceConditionConfig(audio_positions_mode="lipdub_negative")`` path
must place audio reference tokens at exactly the positions the SHIPPED inference uses
(``ltx_pipelines.lipdub.patchify_lipdub_audio_reference_latent`` with ``negative_positions=True``,
which the ComfyUI / LipDub audio-reference path relies on). A train-time offset that drifts from
inference silently degrades the IC-LoRA, so this is locked two ways:

- literal pins: the gap constant, reference tokens strictly negative, ending at ``-gap``;
- a cross-parity check that imports the REAL upstream patchify and asserts byte-equal positions
  (catches upstream drift the literal pin cannot).

The test drives the strategy's reference-conditioning geometry directly — the exact internal
contract being locked — rather than the full training pipeline, mirroring the previous
validation-sampler parity lock.
"""

from pathlib import Path

import pytest
import torch
import yaml

from ltx_trainer.training_strategies.flexible import (
    REFERENCE_ROPE_GAP,
    FlexibleStrategy,
    FlexibleStrategyConfig,
    ModalityConfig,
    ReferenceConditionConfig,
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "audio_reference_ic_lora.yaml"

_AUDIO_CHANNELS = 8
_AUDIO_MEL_BINS = 16
_REF_FRAMES = 10
_TARGET_SEQ = 20


def _make_audio_reference_strategy() -> FlexibleStrategy:
    """Audio-only FlexibleStrategy whose audio target carries a lipdub-negative reference."""
    config = FlexibleStrategyConfig(
        name="flexible",
        audio=ModalityConfig(
            is_generated=True,
            latents_dir="audio_latents",
            conditions=[
                ReferenceConditionConfig(
                    latents_dir="reference_audio_latents",
                    audio_positions_mode="lipdub_negative",
                    probability=1.0,
                )
            ],
        ),
    )
    return FlexibleStrategy(config)


def _apply_reference_and_get_ref_positions() -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reference-conditioning geometry; return (ref_positions, ref_vae_latents)."""
    device = torch.device("cpu")
    strategy = _make_audio_reference_strategy()
    config = strategy.config.audio.conditions[0]

    ref_vae_latents = torch.randn(1, _AUDIO_CHANNELS, _REF_FRAMES, _AUDIO_MEL_BINS)
    batch = {"reference_audio_latents": {"latents": ref_vae_latents}}

    # Minimal target-stream tensors (audio patchified dim = channels * mel_bins = 128).
    noisy = torch.randn(1, _TARGET_SEQ, _AUDIO_CHANNELS * _AUDIO_MEL_BINS)
    positions = strategy._get_audio_positions(num_time_steps=_TARGET_SEQ, batch_size=1, device=device)
    timesteps = torch.zeros(1, _TARGET_SEQ)
    loss_mask = torch.ones(1, _TARGET_SEQ, dtype=torch.bool)
    targets = torch.randn(1, _TARGET_SEQ, _AUDIO_CHANNELS * _AUDIO_MEL_BINS)

    _, combined_positions, _, _, _ = strategy._apply_reference_condition(
        noisy, positions, timesteps, loss_mask, targets, batch, config, "audio"
    )
    # Reference is prepended: [cond | target]. Recover the reference slice.
    ref_seq = strategy._audio_patchifier.patchify(ref_vae_latents).shape[1]
    ref_positions = combined_positions[:, :, :ref_seq, :]
    return ref_positions, ref_vae_latents


def test_reference_positions_are_negative_ending_at_the_locked_gap():
    ref_positions, _ = _apply_reference_and_get_ref_positions()
    # The gap constant is load-bearing (must match the inference constant).
    assert REFERENCE_ROPE_GAP == 0.04
    # Every reference coordinate is strictly negative; the last upper-bound sits at -gap.
    assert (ref_positions < 0).all()
    # FP32: -0.04 is not exactly representable, so approx (the byte-equal test below is the
    # exact lock against inference).
    assert ref_positions.max().item() == pytest.approx(-REFERENCE_ROPE_GAP)


def test_reference_positions_match_shipped_lipdub_inference_byte_equal():
    lipdub = pytest.importorskip("ltx_pipelines.lipdub")
    ref_positions, ref_vae_latents = _apply_reference_and_get_ref_positions()
    _, upstream_positions = lipdub.patchify_lipdub_audio_reference_latent(
        ref_vae_latents, negative_positions=True, device=torch.device("cpu")
    )
    assert torch.equal(ref_positions, upstream_positions)


def test_shipped_audio_reference_config_pins_lipdub_negative():
    """The shipped training config must keep the lipdub-negative convention — regenerating it
    from the stock a2a schema (positive positions) would silently break inference parity."""
    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    strat = FlexibleStrategyConfig(**raw["training_strategy"])
    ref = strat.audio.conditions[0]
    assert isinstance(ref, ReferenceConditionConfig)
    assert ref.audio_positions_mode == "lipdub_negative"

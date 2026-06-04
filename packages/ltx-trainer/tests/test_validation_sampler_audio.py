"""Unit tests for the validation sampler's audio-input paths (the scripted render lane).

Pure CPU tensor plumbing — no model, no GPU, no checkpoint. They lock the two
GenerationConfig audio inputs the scripted eval needs:

  - ``input_audio_latents`` — a real clip's audio latents injected as CLEAN conditioning
    (whole stream frozen via denoise_mask=0; the audio->video coupling eval).
  - ``reference_audio_latents`` — an in-context reference appended ``[target | ref]`` at
    NEGATIVE RoPE positions (the audio-reference IC-LoRA swap eval; the trainer twin of
    the ComfyUI guide node).

The reference geometry MUST byte-match both the training strategy and the upstream
inference convention — the same train/inference parity the strategy tests lock. The
denoise loop itself needs no changes (per-token ``timesteps = sigma * denoise_mask`` +
the clean-state re-imposition already pin mask=0 tokens); these tests cover the state
preparation that loop consumes.
"""

from __future__ import annotations

import pytest
import torch

from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape
from ltx_trainer.training_strategies.audio_reference import REFERENCE_ROPE_GAP
from ltx_trainer.validation_sampler import (
    apply_input_audio_conditioning,
    extend_audio_state_with_reference,
    strip_audio_reference,
)

AUDIO_CHANNELS = 8
AUDIO_MEL_BINS = 16


def _patchifier() -> AudioPatchifier:
    return AudioPatchifier(patch_size=1)


def _audio_state(t: int):
    """A fresh audio LatentState with t time-steps (the sampler's create_initial_state shape)."""
    tools = AudioLatentTools(
        patchifier=_patchifier(),
        target_shape=AudioLatentShape(batch=1, channels=AUDIO_CHANNELS, frames=t, mel_bins=AUDIO_MEL_BINS),
    )
    return tools.create_initial_state(device=torch.device("cpu"), dtype=torch.float32)


def _vae_latents(t: int) -> torch.Tensor:
    return torch.randn(1, AUDIO_CHANNELS, t, AUDIO_MEL_BINS)


# --- input_audio_latents: clean in-place conditioning --------------------------


def test_input_audio_conditioning_freezes_the_whole_stream():
    state = _audio_state(20)
    latents = _vae_latents(20)
    out = apply_input_audio_conditioning(state, latents, _patchifier())

    expected = _patchifier().patchify(latents).to(out.latent.dtype)
    assert torch.equal(out.clean_latent, expected)
    assert torch.equal(out.latent, expected)
    assert (out.denoise_mask == 0).all()  # whole stream conditioned -> noiser + loop hold it clean
    assert torch.equal(out.positions, state.positions)  # in-place: positions untouched


def test_input_audio_conditioning_rejects_length_mismatch():
    state = _audio_state(20)
    with pytest.raises(ValueError, match="token"):
        apply_input_audio_conditioning(state, _vae_latents(12), _patchifier())


def test_input_audio_conditioning_accepts_unbatched_latents():
    state = _audio_state(20)
    out = apply_input_audio_conditioning(state, _vae_latents(20)[0], _patchifier())
    assert out.latent.shape == state.latent.shape


# --- reference_audio_latents: [target | ref] at negative RoPE -------------------


def test_reference_appends_target_then_ref_with_zero_ref_mask():
    state = _audio_state(20)
    extended, ref_seq_len = extend_audio_state_with_reference(state, _vae_latents(12), _patchifier())

    assert ref_seq_len == 12
    assert extended.latent.shape[1] == 20 + 12  # target leads, reference trails
    # Target block untouched; reference block clean + mask-0 (the IC reference contract).
    assert torch.equal(extended.latent[:, :20], state.latent)
    assert torch.equal(extended.denoise_mask[:, :20], state.denoise_mask)
    assert (extended.denoise_mask[:, 20:] == 0).all()
    assert torch.equal(extended.clean_latent[:, 20:], extended.latent[:, 20:])


def test_reference_positions_are_negative_ending_at_the_locked_gap():
    state = _audio_state(20)
    extended, _ = extend_audio_state_with_reference(state, _vae_latents(12), _patchifier())

    # positions: [B, 1, seq, 2]; target block first, untouched
    assert torch.equal(extended.positions[:, :, :20], state.positions)
    ref_time = extended.positions[:, 0, 20:, :]
    assert (ref_time < 0).all()
    assert ref_time.max().item() == pytest.approx(-REFERENCE_ROPE_GAP)


def test_reference_positions_match_upstream_lipdub_function():
    """CROSS-PARITY: the sampler's reference positions must equal what the real inference
    function computes — the same lock the strategy carries
    (test_audio_reference_strategy.py). A sampler that drifts from lipdub would rank
    checkpoints under a geometry the shipped inference never runs."""
    lipdub = pytest.importorskip("ltx_pipelines.lipdub")

    state = _audio_state(20)
    ref_latents = _vae_latents(12)
    extended, _ = extend_audio_state_with_reference(state, ref_latents, _patchifier())

    _, upstream_positions = lipdub.patchify_lipdub_audio_reference_latent(
        ref_latents, negative_positions=True, device=torch.device("cpu")
    )
    assert torch.equal(extended.positions[:, :, 20:].float(), upstream_positions)


def test_strip_audio_reference_roundtrips_the_target():
    state = _audio_state(20)
    extended, ref_seq_len = extend_audio_state_with_reference(state, _vae_latents(12), _patchifier())
    stripped = strip_audio_reference(extended, ref_seq_len)

    assert torch.equal(stripped.latent, state.latent)
    assert torch.equal(stripped.denoise_mask, state.denoise_mask)
    assert torch.equal(stripped.positions, state.positions)
    assert torch.equal(stripped.clean_latent, state.clean_latent)


# --- GenerationConfig validation -------------------------------------------------


def test_config_rejects_both_audio_inputs_at_once():
    from ltx_trainer.validation_sampler import GenerationConfig

    config = GenerationConfig(
        prompt="a person speaking",
        input_audio_latents=_vae_latents(20),
        reference_audio_latents=_vae_latents(12),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate_audio_inputs()


def test_config_audio_inputs_require_generate_audio():
    from ltx_trainer.validation_sampler import GenerationConfig

    config = GenerationConfig(
        prompt="a person speaking",
        generate_audio=False,
        reference_audio_latents=_vae_latents(12),
    )
    with pytest.raises(ValueError, match="generate_audio"):
        config.validate_audio_inputs()


# --- eval_utils: loading precomputed audio-latent .pt files ---------------------


def test_load_audio_latents_file_handles_precompute_dict_and_k_stacks(tmp_path):
    """The precompute writes ``{latents: [C,T,F], ...}`` (targets) and ``{latents: [K,C,T,F],
    variants: K}`` (stacked references). The eval loader must accept both, select a variant
    from a stack, and reject malformed shapes — so sweep scripts can point at the exact files
    training consumed."""
    from ltx_trainer.eval_utils import load_audio_latents_file

    plain = {"latents": torch.randn(8, 88, 16), "duration": 3.5}
    p1 = tmp_path / "plain.pt"
    torch.save(plain, p1)
    out = load_audio_latents_file(p1)
    assert torch.equal(out, plain["latents"])

    stack = {"latents": torch.randn(3, 8, 88, 16), "variants": 3}
    p2 = tmp_path / "stack.pt"
    torch.save(stack, p2)
    assert torch.equal(load_audio_latents_file(p2, variant=2), stack["latents"][2])

    bare = torch.randn(8, 88, 16)
    p3 = tmp_path / "bare.pt"
    torch.save(bare, p3)
    assert torch.equal(load_audio_latents_file(p3), bare)

    with pytest.raises(ValueError, match="variant"):
        load_audio_latents_file(p2, variant=7)
    p4 = tmp_path / "bad.pt"
    torch.save({"latents": torch.randn(88, 16)}, p4)
    with pytest.raises(ValueError, match="shape"):
        load_audio_latents_file(p4)

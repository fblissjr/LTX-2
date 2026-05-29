"""Forward/backward smoke for `audio_reference` on the REAL LTX architecture.

Builds a TINY real-architecture LTX AudioVideo model (random weights, CPU — no
checkpoint, no GPU) and runs ONE forward + compute_loss + backward with the
audio_reference strategy's ModelInputs. This answers the *architectural*
feasibility question for an audio-only IC-LoRA:

  Does the LTX joint-AV transformer ACCEPT the audio-reference shape — an audio
  sequence = [target | reference] that is LONGER than the video, the reference
  at distinct (negative) RoPE positions, and AV cross-attention with a
  target-video <-> (target+reference)-audio length mismatch — and backprop a
  finite loss with gradients reaching the audio stream (the LoRA surface)?

It does NOT test scale/VRAM (that's the real-22B smoke) nor that the attribute
is learned (that's the data-backed train). It's the cheap, deterministic proof
that the mechanism is architecturally sound — and a regression guard on the
position/concat convention in the strategy.
"""

from __future__ import annotations

import torch
from _audio_reference_helpers import EMBED_DIM, FixedSigmaSampler, make_batch

from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_trainer.training_strategies.audio_reference import (
    AudioReferenceConfig,
    AudioReferenceStrategy,
)

# Tiny audio lengths for the real-arch forward (target leads, reference trails).
AUDIO_T = 10
REF_AUDIO_T = 6


def _batch(b: int = 1, frames: int = 3, h: int = 8, w: int = 8, fps: float = 24.0) -> dict:
    # int32 prompt mask: the model's _prepare_attention_mask does (mask-1)*max,
    # which rejects bool. The real trainer feeds a converted (non-bool) mask.
    return make_batch(
        batch=b,
        frames=frames,
        h=h,
        w=w,
        audio_t=AUDIO_T,
        ref_audio_t=REF_AUDIO_T,
        fps=fps,
        prompt_mask_dtype=torch.int32,
    )


def _tiny_av_model():
    """A minimal real-architecture LTX AudioVideo model (V2/22B-style: caption
    projection lives in the text encoder, so none is built here)."""
    transformer = {
        # tiny dims
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 2,
        "cross_attention_dim": EMBED_DIM,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 16,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": EMBED_DIM,
        "attention_type": "pytorch",  # SDPA -> CPU-safe
        "caption_proj_before_connector": True,  # V2: no in-transformer caption proj
        # equality-checked keys (LTXModelConfigurator.from_config)
        "dropout": 0.0,
        "attention_bias": True,
        "num_vector_embeds": None,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "use_linear_projection": False,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "double_self_attention": False,
        "upcast_attention": False,
        "standardization_norm": "rms_norm",
        "norm_elementwise_affine": False,
        "qk_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "use_audio_video_cross_attention": True,
        "share_ff": False,
        "av_cross_ada_norm": True,
        "use_middle_indices_grid": True,
    }
    model = LTXModelConfigurator.from_config({"transformer": transformer})
    # AdaLN-zero init: real DiT-style models zero their modulation tables so residual
    # blocks start near-identity. RANDOM init of these tables blows an untrained
    # forward to NaN (independent of this strategy — a plain text_to_video forward
    # NaNs the same way). Zeroing them makes the random-weight forward numerically
    # sane, which is what lets this CPU smoke assert a finite loss.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scale_shift_table" in name:
                p.zero_()
    return model


def test_audio_reference_forward_backward_runs_on_real_ltx_arch():
    torch.manual_seed(0)
    model = _tiny_av_model().float()
    model.train()

    strat = AudioReferenceStrategy(AudioReferenceConfig(first_frame_conditioning_p=0.0))
    inputs = strat.prepare_training_inputs(_batch(), FixedSigmaSampler(0.5))

    video_pred, audio_pred = model(video=inputs.video, audio=inputs.audio, perturbations=None)

    # The audio prediction spans target + reference; the reference is masked out of the loss.
    assert audio_pred.shape[1] == AUDIO_T + REF_AUDIO_T

    loss = strat.compute_loss(video_pred, audio_pred, inputs)
    assert loss.shape == (1,)  # per-element [B]
    loss = loss.mean()
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()

    # Gradient must reach the AUDIO stream (where the LoRA attaches) and be finite —
    # i.e. the audio path is differentiably connected to the loss and trainable.
    audio_grads = [
        p.grad
        for name, p in model.named_parameters()
        if "audio" in name and p.requires_grad and p.grad is not None
    ]
    assert audio_grads, "no gradient reached any audio-stream parameter"
    assert all(torch.isfinite(g).all() for g in audio_grads), "non-finite audio gradient"


def test_reference_audio_is_load_bearing():
    """The claim that actually matters: the loss must DEPEND on the reference
    audio tokens, i.e. ∂loss/∂(reference) != 0. Shapes can pass and the forward
    run while the reference is *decorative* — if something masked it out of the
    target's attention, no gradient would flow back through it. The reference is
    held clean and EXCLUDED from the loss, so a non-zero gradient w.r.t. it can
    only arise if the target genuinely attends to it. This is the trainer-side
    twin of the data eval's "remove the reference -> output stops tracking", and
    it's checkable on the finite tiny model (no real weights needed)."""
    torch.manual_seed(0)
    model = _tiny_av_model().float()
    model.train()
    strat = AudioReferenceStrategy(AudioReferenceConfig(first_frame_conditioning_p=0.0))

    batch = _batch()
    ref = batch["reference_audio_latents"]["latents"].clone().requires_grad_(True)
    batch["reference_audio_latents"]["latents"] = ref

    inputs = strat.prepare_training_inputs(batch, FixedSigmaSampler(0.5))
    video_pred, audio_pred = model(video=inputs.video, audio=inputs.audio, perturbations=None)
    loss = strat.compute_loss(video_pred, audio_pred, inputs).mean()
    assert torch.isfinite(loss)
    loss.backward()

    assert ref.grad is not None, "reference audio got NO gradient — it is decorative, not load-bearing"
    assert torch.isfinite(ref.grad).all(), "non-finite gradient w.r.t. reference audio"
    assert ref.grad.abs().sum() > 0, "zero gradient w.r.t. reference audio — the target does not attend to it"

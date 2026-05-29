"""Audio-reference IC-LoRA training strategy.

The "audio is the controller" paradigm (Shape 3 / transfer): an in-context
reference *audio* clip carries an attribute (e.g. a pitch), and the model
generates an AV target that adopts that attribute. Content comes from the
caption (kept attribute-free); the *attribute* is read from the reference audio.
This is the only shape where the audio reference is load-bearing — matched
content would push control into the caption (the "seesaw"); reference and target
therefore share the attribute and DIFFER in content.

Mechanics (mirrors ltx_core.conditioning.types.reference_audio_cond):
- The audio stream is ``[target (noised) | reference (clean)]`` — target tokens
  lead and stay in the loss; reference tokens trail, held clean (denoise_mask 0)
  at distinct (negative) RoPE positions so the model reads them as context.
- The target is AV: the video is the generated target (first-frame conditioning
  optional) and the target-audio portion is noised + in the loss. The video
  follows the (reference-influenced) audio because LTX is jointly trained.
- Loss = video target loss + masked target-audio loss; reference audio excluded.

No video reference (this is audio-only-reference). The strategy is agnostic to
what the reference audio *is* (a tone, a voice) — that is a data decision.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class AudioReferenceConfig(TrainingStrategyConfigBase):
    """Configuration for the audio-reference IC-LoRA training strategy."""

    name: Literal["audio_reference"] = "audio_reference"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the target's first frame during training",
        ge=0.0,
        le=1.0,
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for the precomputed TARGET audio latents",
    )

    reference_audio_latents_dir: str = Field(
        default="reference_audio_latents",
        description=(
            "Directory name for the precomputed REFERENCE audio latents (the in-context "
            "attribute carrier, e.g. a voiced tone at the target pitch)"
        ),
    )

    reference_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 keeps the reference audio fully clean; <1.0 partially noises it.",
    )


class AudioReferenceStrategy(TrainingStrategy):
    """Audio-reference IC-LoRA: reference audio steers an AV target's audio attribute."""

    config: AudioReferenceConfig

    def __init__(self, config: AudioReferenceConfig):
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Always — both the target audio and the reference audio are audio latents."""
        return True

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.audio_latents_dir: "audio_latents",
            self.config.reference_audio_latents_dir: "reference_audio_latents",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        if "reference_audio_latents" not in batch:
            raise ValueError(
                "AudioReferenceStrategy requires a 'reference_audio_latents' data source "
                "(the in-context reference audio). It is missing from the batch — check "
                "reference_audio_latents_dir and that the data was precomputed."
            )

        # --- Video target (generated; no video reference) -----------------------
        latents = batch["latents"]
        video_latents = self._video_patchifier.patchify(latents["latents"])
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        sigmas = timestep_sampler.sample_for(video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        video_noise = torch.randn_like(video_latents)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise
        noisy_video = torch.where(video_conditioning_mask.unsqueeze(-1), video_latents, noisy_video)
        video_targets = video_noise - video_latents
        video_loss_mask = ~video_conditioning_mask

        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )
        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            sigma=sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # --- Audio: [target (noised) | reference (clean)] -----------------------
        target_audio = self._audio_patchifier.patchify(batch["audio_latents"]["latents"])
        ref_audio = self._audio_patchifier.patchify(batch["reference_audio_latents"]["latents"])
        target_audio_len = target_audio.shape[1]
        ref_audio_len = ref_audio.shape[1]

        # Hold the reference clean by reference_strength (1.0 -> fully clean).
        ref_keep = self.config.reference_strength
        if ref_keep < 1.0:
            ref_noise = torch.randn_like(ref_audio)
            ref_audio = ref_keep * ref_audio + (1 - ref_keep) * ref_noise

        audio_noise = torch.randn_like(target_audio)
        noisy_target_audio = (1 - sigmas_expanded) * target_audio + sigmas_expanded * audio_noise

        # Sequence: target leads (in the loss), reference trails (clean context).
        audio_latent = torch.cat([noisy_target_audio, ref_audio], dim=1)
        audio_targets = torch.cat([audio_noise - target_audio, torch.zeros_like(ref_audio)], dim=1)

        # Conditioning mask: target = noised target (False), reference = clean (True).
        audio_conditioning_mask = torch.zeros(
            batch_size, target_audio_len + ref_audio_len, dtype=torch.bool, device=device
        )
        audio_conditioning_mask[:, target_audio_len:] = True
        audio_loss_mask = ~audio_conditioning_mask

        audio_timesteps = self._create_per_token_timesteps(audio_conditioning_mask, sigmas.squeeze())

        target_audio_positions = self._get_audio_positions(
            num_time_steps=target_audio_len, batch_size=batch_size, device=device, dtype=dtype
        )
        # Reference at distinct, strictly-negative positions so the model reads it as
        # out-of-timeline context. This MUST match the inference convention exactly
        # (ltx_pipelines.lipdub.patchify_lipdub_audio_reference_latent with
        # negative_positions=True): shift by the reference's own end-bound plus a small
        # 0.04 gap, so the reference ends just below the target timeline's 0. RoPE is
        # absolute — a different train-time offset would give the LoRA a reference<->target
        # geometry it never sees at generation time.
        ref_audio_positions = self._get_audio_positions(
            num_time_steps=ref_audio_len, batch_size=batch_size, device=device, dtype=dtype
        )
        aud_dur = ref_audio_positions[:, :, -1, 1].max()
        ref_audio_positions = ref_audio_positions - aud_dur - 0.04
        audio_positions = torch.cat([target_audio_positions, ref_audio_positions], dim=2)

        audio_modality = Modality(
            enabled=True,
            latent=audio_latent,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=None,  # no video reference; audio reference is masked, not offset
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Video target loss + masked target-audio loss. Reference audio is excluded
        by ``audio_loss_mask`` (False on the trailing reference tokens). Returns [B,].

        Video stays in the loss even when ``target_modules`` are audio-only: the
        seemingly-wasted video term is not a real cost, because the video stream's
        forward+backward is already required for the *audio* gradient (audio_pred
        depends on video activations via video->audio cross-attention). The only true
        cost lever would be dropping video entirely (audio-only output), which trades
        away the AV product shape — a deliberate fallback, not the default."""
        video_loss = self._masked_velocity_loss(video_pred, inputs.video_targets, inputs.video_loss_mask)

        if audio_pred is None or inputs.audio_targets is None or inputs.audio_loss_mask is None:
            return video_loss

        audio_loss = self._masked_velocity_loss(audio_pred, inputs.audio_targets, inputs.audio_loss_mask)
        return video_loss + audio_loss

"""Video-to-video training strategy for IC-LoRA.
This strategy implements training with reference video conditioning where:
- Reference latents (clean) are concatenated with target latents (noised)
- Video coordinates handle both reference and target sequences
- Loss is computed only on the target portion
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
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)

# Audio-guidance training modes. See internal/trainer_audio_iclora_plan.md.
#   condition    — audio is clean context (the guide), video is the noised
#                  target, loss on video only. "Audio guides video." PRIMARY.
#   generate     — audio noised + audio loss (joint AV gen; mirrors text_to_video).
#   continuation — audio prefix clean + tail noised (loss on tail), video fully
#                  clean (no video loss). Training analog of the video->audio inversion.
AudioMode = Literal["condition", "generate", "continuation"]


class VideoToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for video-to-video (IC-LoRA) training strategy."""

    name: Literal["video_to_video"] = "video_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for latents of reference videos",
    )

    with_audio: bool = Field(
        default=False,
        description=(
            "Train with audio in the loop so the IC-LoRA can learn an audio->video "
            "guidance relationship (LTX 2.3 is a joint audio-video model). When False "
            "the strategy behaves exactly as the original video-only IC-LoRA."
        ),
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for precomputed audio latents when with_audio is True",
    )

    audio_mode: AudioMode = Field(
        default="condition",
        description=(
            "How audio participates. 'condition': audio is clean guidance, loss on "
            "video (audio guides video generation). 'generate': joint AV generation "
            "(audio noised + in loss). 'continuation': audio prefix clean + tail "
            "generated, video frozen (video->audio inversion analog)."
        ),
    )

    audio_prefix_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "For audio_mode='continuation': seconds of audio kept clean as the seed; "
            "the rest is the generation target. Ignored by other modes."
        ),
    )


class VideoToVideoStrategy(TrainingStrategy):
    """Video-to-video training strategy for IC-LoRA.
    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates handle both reference and target sequences
    - Loss is computed only on the target portion
    Attributes:
        reference_downscale_factor: The inferred downscale factor of reference videos.
            This is computed from the first batch and cached for metadata export.
    """

    config: VideoToVideoConfig
    reference_downscale_factor: int | None

    def __init__(self, config: VideoToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Video-to-video configuration
        """
        super().__init__(config)
        self.reference_downscale_factor = None  # Will be inferred from first batch

    @property
    def requires_audio(self) -> bool:
        """Load audio VAE + vocoder only when training with audio in the loop."""
        return self.config.with_audio

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents.

        With audio enabled, also pull precomputed audio latents (produced by
        ``scripts/process_dataset.py --with-audio`` into ``audio_latents/``).
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }
        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"
        return sources

    def prepare_training_inputs(  # noqa: PLR0915
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for IC-LoRA training with reference videos."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Get dimensions
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Infer reference downscale factor from dimension ratios
        # This allows training with downscaled reference videos for efficiency
        reference_downscale_factor = self._infer_reference_downscale_factor(
            target_height=height,
            target_width=width,
            ref_height=ref_height,
            ref_width=ref_width,
        )

        # Cache the scale factor for metadata export (only on first batch)
        if self.reference_downscale_factor is None:
            self.reference_downscale_factor = reference_downscale_factor
        elif self.reference_downscale_factor != reference_downscale_factor:
            raise ValueError(
                f"Inconsistent reference downscale factor across batches. "
                f"First batch had factor={self.reference_downscale_factor}, "
                f"but current batch has factor={reference_downscale_factor}. "
                f"All training samples must use the same reference/target resolution ratio."
            )

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        target_latents = self._video_patchifier.patchify(target_latents)
        ref_latents = self._video_patchifier.patchify(ref_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        # Video-to-video uses only video embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]
        device = target_latents.device
        dtype = target_latents.dtype

        # Create conditioning mask
        # Reference tokens are always conditioning (timestep=0)
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # continuation mode freezes the whole video as clean context (the model
        # only generates the audio tail). Every target token becomes clean →
        # zero video loss. Reuses the existing per-token conditioning machinery.
        if self.config.with_audio and self.config.audio_mode == "continuation":
            target_conditioning_mask = torch.ones_like(target_conditioning_mask)

        # Combined conditioning mask
        conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

        # Sample noise and sigmas for target
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Apply noise to target
        noisy_target = (1 - sigmas_expanded) * target_latents + sigmas_expanded * noise

        # For first frame conditioning in target, use clean latents
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Targets for loss computation
        targets = noise - target_latents

        # Concatenate reference (clean) and target (noisy)
        combined_latents = torch.cat([ref_latents, noisy_target], dim=1)

        # Create per-token timesteps
        timesteps = self._create_per_token_timesteps(conditioning_mask, sigmas.squeeze())

        # Generate positions for reference and target separately, then concatenate
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Scale reference positions to match target coordinate space
        # This maps ref positions from (0, ref_H, ref_W) to (0, target_H, target_W)
        # Position tensor shape: [B, 3, seq_len, 2] where dim 1 is (time, height, width)
        if reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= reference_downscale_factor  # width axis
            # Time axis (index 0) remains unchanged

        target_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Concatenate positions along sequence dimension
        positions = torch.cat([ref_positions, target_positions], dim=2)

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas,
            timesteps=timesteps,
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: only compute loss on non-conditioning target tokens
        # Reference tokens: all False (no loss)
        # Target tokens: True where not conditioning
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        # Audio: optional second modality that the LoRA learns to be guided by.
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        if self.config.with_audio:
            audio_prompt_embeds = conditions.get("audio_prompt_embeds")
            if audio_prompt_embeds is None:
                raise ValueError(
                    "with_audio=True but conditions['audio_prompt_embeds'] is missing or None. "
                    "The audio stream needs its text-encoder embeddings — ensure the loaded model "
                    "has an audio connector and the embedding processor populated audio_prompt_embeds."
                )
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                num_video_latent_frames=num_frames,
                fps=fps,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            ref_seq_len=ref_seq_len,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Masked velocity loss, summed over active streams. Returns [B,].

        Video loss is on the target portion only (reference tokens excluded by
        ref_seq_len). When audio has target tokens (generate/continuation), its
        masked loss is added; in `condition` mode audio is pure context so
        ``audio_targets`` is None and no audio term is added. The clamp makes a
        fully-masked stream contribute 0 (e.g. video in continuation mode).
        """
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        video_loss = (target_pred - inputs.video_targets).pow(2)
        video_mask = target_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_mask).mean(dim=[-2, -1]) / video_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

        if (
            not self.config.with_audio
            or audio_pred is None
            or inputs.audio_targets is None
            or inputs.audio_loss_mask is None
        ):
            return video_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        audio_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
        audio_loss = audio_loss.mul(audio_mask).mean(dim=[-2, -1]) / audio_mask.mean(dim=[-2, -1]).clamp(min=1e-8)
        return video_loss + audio_loss

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        num_video_latent_frames: int,
        fps: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor | None, Tensor]:
        """Build the audio Modality for the configured audio_mode.

        Returns ``(audio_modality, audio_targets, audio_loss_mask)``. The audio
        conditioning mask (True = clean context token, False = noised target)
        encodes the mode:
          condition    → all True  (clean guide, no audio loss → targets None)
          generate     → all False (everything generated)
          continuation → first ``audio_prefix_seconds`` clean, rest generated
        """
        audio_latents = self._audio_patchifier.patchify(batch["audio_latents"]["latents"])  # [B, T, C*F]
        audio_seq_len = audio_latents.shape[1]

        audio_conditioning_mask = torch.zeros(batch_size, audio_seq_len, dtype=torch.bool, device=device)
        if self.config.audio_mode == "condition":
            audio_conditioning_mask[:] = True
        elif self.config.audio_mode == "continuation":
            n_clean = self._audio_prefix_frame_count(
                audio_seq_len=audio_seq_len,
                num_video_latent_frames=num_video_latent_frames,
                fps=fps,
                prefix_seconds=self.config.audio_prefix_seconds,
            )
            if n_clean >= audio_seq_len:
                raise ValueError(
                    f"audio_mode='continuation' with audio_prefix_seconds="
                    f"{self.config.audio_prefix_seconds} keeps all {audio_seq_len} audio frames "
                    "as clean seed, leaving nothing to generate. Combined with the frozen video "
                    "this is a zero-gradient step. Reduce audio_prefix_seconds below the clip length."
                )
            if n_clean > 0:
                audio_conditioning_mask[:, :n_clean] = True
        # else "generate": all False (all noised target)

        sigmas_expanded = sigmas.view(-1, 1, 1)
        audio_noise = torch.randn_like(audio_latents)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise
        # Conditioning tokens keep the clean latent; only target tokens are noised.
        noisy_audio = torch.where(audio_conditioning_mask.unsqueeze(-1), audio_latents, noisy_audio)

        audio_timesteps = self._create_per_token_timesteps(audio_conditioning_mask, sigmas.squeeze())
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss only on the noised (target) tokens. All-clean (condition mode) →
        # no targets, so the loss term is skipped entirely in compute_loss.
        audio_loss_mask = ~audio_conditioning_mask
        audio_targets = audio_noise - audio_latents if bool(audio_loss_mask.any()) else None
        return audio_modality, audio_targets, audio_loss_mask

    @staticmethod
    def _audio_prefix_frame_count(
        audio_seq_len: int,
        num_video_latent_frames: int,
        fps: float,
        prefix_seconds: float,
    ) -> int:
        """How many leading audio-latent frames cover ``prefix_seconds``.

        Maps seconds onto the audio-latent time axis using the VIDEO clock so
        the kept audio prefix and the video it should match share one timeline:
        derive clip duration from the video latent-frame count (latent→pixel via
        the temporal VAE factor, then /fps), and scale by the audio frame rate
        ``audio_seq_len / duration``. Clamped to ``[0, audio_seq_len]``.
        """
        if prefix_seconds <= 0 or audio_seq_len <= 0 or fps <= 0:
            return 0
        time_scale = VIDEO_SCALE_FACTORS.time
        pixel_frames = (num_video_latent_frames - 1) * time_scale + 1
        duration_s = pixel_frames / fps
        if duration_s <= 0:
            return 0
        rate = audio_seq_len / duration_s
        return min(audio_seq_len, max(0, round(prefix_seconds * rate)))

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata for checkpoint files."""
        metadata: dict[str, Any] = {}
        # Always include reference_downscale_factor for IC-LoRAs so inference
        # pipelines know the expected scale factor for reference videos.
        if self.reference_downscale_factor is not None:
            metadata["reference_downscale_factor"] = self.reference_downscale_factor
        return metadata

    @staticmethod
    def _infer_reference_downscale_factor(
        target_height: int,
        target_width: int,
        ref_height: int,
        ref_width: int,
    ) -> int:
        """Infer the reference downscale factor from target and reference dimensions."""
        # If dimensions match, no scaling needed
        if target_height == ref_height and target_width == ref_width:
            return 1

        # Calculate scale factors for each dimension
        if target_height % ref_height != 0 or target_width % ref_width != 0:
            raise ValueError(
                f"Target dimensions ({target_height}x{target_width}) must be exact multiples "
                f"of reference dimensions ({ref_height}x{ref_width})"
            )

        scale_h = target_height // ref_height
        scale_w = target_width // ref_width

        if scale_h != scale_w:
            raise ValueError(
                f"Reference scale must be uniform. Got height scale {scale_h} and width scale {scale_w}. "
                f"Target: {target_height}x{target_width}, Reference: {ref_height}x{ref_width}"
            )

        if scale_h < 1:
            raise ValueError(
                f"Reference dimensions ({ref_height}x{ref_width}) cannot be larger than "
                f"target dimensions ({target_height}x{target_width})"
            )

        return scale_h

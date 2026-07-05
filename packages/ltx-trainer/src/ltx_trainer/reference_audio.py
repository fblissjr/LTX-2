"""Reference-audio precompute helpers (the ``reference_audio_latents`` channel).

`AudioReferenceStrategy` reads an in-context reference audio and concatenates it
(clean) into the audio stream. This module encodes a reference waveform (e.g. a
voiced tone at the target pitch) into a latent in the SAME format the AV precompute
writes for ``audio_latents`` (a ``.pt`` of ``{latents:[C,T,F], num_time_steps,
frequency_bins, duration}``), and maps it to the SAME relative path as the clip's
video latent so :class:`ltx_trainer.datasets.PrecomputedDataset` pairs the sources.

Mirrors the audio path in ``scripts/process_videos.py`` (intentionally a small,
self-contained reimplementation so the working video+audio precompute is untouched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ltx_core.types import Audio


def reference_output_path(
    video_path: str | Path, output_dir: str | Path, data_root: str | Path | None = None
) -> Path:
    """Reference latent path = ``output_dir / <clip rel>.pt`` — the same relative path
    the clip's video latent uses, so PrecomputedDataset pairs them by path.

    Args:
        video_path: The clip's path from the manifest (relative to data_root, or absolute).
        output_dir: The ``reference_audio_latents`` directory.
        data_root: Root to relativize an absolute ``video_path`` against.
    """
    vp = Path(video_path)
    if data_root is not None and vp.is_absolute():
        vp = vp.relative_to(Path(data_root))
    return Path(output_dir) / vp.with_suffix(".pt")


def build_audio_processor(encoder: torch.nn.Module):
    """Build the AudioProcessor matching the encoder's mel/sample-rate config."""
    from ltx_core.model.audio_vae import AudioProcessor

    return AudioProcessor(
        target_sample_rate=encoder.sample_rate,
        mel_bins=encoder.mel_bins,
        mel_hop_length=encoder.mel_hop_length,
        n_fft=encoder.n_fft,
    )


def ensure_audio_channels(waveform: torch.Tensor, want: int) -> torch.Tensor:
    """Match the waveform's channel count to what the audio VAE expects.

    The LTX audio VAE encodes a 2-channel (stereo) mel — its first conv has
    ``in_channels=2``. Mono speech/tones therefore must be widened to that many
    channels: duplicating mono into N identical channels ("dual-mono") is the
    correct way to feed mono content into a stereo-trained VAE (the decoder just
    yields identical L/R). ``>want`` channels are downmixed by averaging.

    Args:
        waveform: ``[..., channels, samples]`` (channels at dim -2).
        want: target channel count (e.g. ``encoder.in_channels``).
    """
    ch = waveform.shape[-2]
    if ch == want:
        return waveform
    if ch > want:
        waveform = waveform.mean(dim=-2, keepdim=True)
        ch = 1
    if ch == 1:
        return waveform.repeat_interleave(want, dim=-2)
    # 1 < ch < want: tile up then trim to exactly ``want``.
    reps = (want + ch - 1) // ch
    return waveform.repeat_interleave(reps, dim=-2)[..., :want, :]


def augment_reference_waveform(
    waveform: torch.Tensor,
    sampling_rate: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Random channel augmentation for a reference waveform (the anti-shortcut transform).

    Reference/target pairs cut from the same recording session share channel fingerprints
    beyond the voice — room tone, mic response, level. An IC-LoRA reads the controlled
    attribute from whichever channel co-varies with it at lowest cost, so a constant session
    fingerprint is a learnable shortcut ("match the room, not the voice"). Randomizing the
    reference's channel per variant (gain jitter, peaking EQ, light noise) makes those
    nuisance factors unreliable while leaving the voice attribute intact — ST-DRC's
    reference-augmentation recipe applied to audio.

    Deterministic given ``generator`` (re-running a precompute reproduces byte-identical
    variants). Waveform is ``[channels, samples]`` (or batched); shape is preserved.
    """
    import torchaudio.functional as taf

    def _unif(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1, generator=generator).item()

    out = waveform
    # Gain jitter +-4 dB (level is the cheapest session fingerprint).
    out = taf.gain(out, gain_db=_unif(-4.0, 4.0))
    # 1-2 random peaking-EQ bands (mic/room coloration), center log-uniform 200 Hz - 6 kHz,
    # clamped below Nyquist for short sample rates.
    n_bands = 1 + int(torch.rand(1, generator=generator).item() < 0.5)
    for _ in range(n_bands):
        center = 10.0 ** _unif(2.301, 3.778)  # ~200 .. ~6000 Hz
        center = min(center, sampling_rate / 2 * 0.9)
        out = taf.equalizer_biquad(out, sampling_rate, center_freq=center, gain=_unif(-6.0, 6.0), Q=_unif(0.7, 2.0))
    # Light broadband noise at -45 .. -35 dB relative to signal RMS (ambience floor).
    rms = out.pow(2).mean().sqrt().clamp_min(1e-8)
    noise_level = rms * (10.0 ** (_unif(-45.0, -35.0) / 20.0))
    out = out + torch.randn(out.shape, generator=generator, dtype=out.dtype) * noise_level
    return out


def encode_reference_waveform(
    encoder: torch.nn.Module,
    processor: Any,
    waveform: torch.Tensor,
    sampling_rate: int,
) -> dict[str, Any]:
    """Encode a reference waveform into a latent dict, format-identical to the
    ``audio_latents`` the AV precompute produces.

    Args:
        encoder: Audio VAE encoder (provides device/dtype via its parameters).
        processor: AudioProcessor with ``waveform_to_mel``.
        waveform: ``[channels, samples]`` or ``[batch, channels, samples]``.
        sampling_rate: Sample rate of ``waveform``.

    Returns:
        ``{"latents": [C, T, F], "num_time_steps": int, "frequency_bins": int, "duration": float}``.
    """
    param = next(encoder.parameters())
    waveform = waveform.to(device=param.device, dtype=param.dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [C, samples] -> [1, C, samples]
    # LTX audio VAE wants a 2-channel mel; widen mono refs to match (dual-mono).
    waveform = ensure_audio_channels(waveform, getattr(encoder, "in_channels", 2))

    duration = waveform.shape[-1] / sampling_rate
    mel = processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sampling_rate)).to(dtype=param.dtype)
    latents = encoder(mel)  # [1, C, T, F]
    _, _channels, time_steps, freq_bins = latents.shape
    return {
        "latents": latents.squeeze(0),  # [C, T, F] — drop batch
        "num_time_steps": int(time_steps),
        "frequency_bins": int(freq_bins),
        "duration": float(duration),
    }

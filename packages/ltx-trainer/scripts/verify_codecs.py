#!/usr/bin/env python
"""Round-trip diagnostic for the codecs the training pipeline relies on:
video VAE, audio VAE (+ optional vocoder), and the Gemma text encoder. Confirms
each can encode→decode and that shapes obey the model's relationships — so we
trust the latents we train on and the outputs we eval.

Works with ComfyUI's SPLIT model layout (separate VAE files), which is what's on
disk in practice — pass each component's path. Needs a CUDA GPU. The VAEs are
LOSSY, so this gates on SHAPE + finiteness (deterministic facts) and reports
reconstruction error informationally — it does not assert bit-equality.

VAEs are loaded in float32: ComfyUI's "bf16" VAE files carry fp32 biases, so a
bf16 forward pass raises a dtype mismatch (validated on disk).

    uv run python packages/ltx-trainer/scripts/verify_codecs.py \
        --video-vae <models/vae/LTX2_video_vae_bf16.safetensors> \
        --audio-vae <models/vae/LTX2_audio_vae_bf16.safetensors> \
        [--vocoder <path>] [--gemma <gemma dir>]

Code map (what each component is + where): internal/audio_iclora_codec_map.md.
Each stage is independent + guarded, so one failure doesn't hide the others.
"""

from __future__ import annotations

import argparse
import sys

import torch

VIDEO_F, VIDEO_HW = 9, 64  # smallest valid: F=1+8k, HW div-by-32
EXP_VIDEO_LATENT = (1, 128, 2, 2, 2)


def _ok(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return cond


def check_video(video_vae: str, device: str) -> bool:
    from ltx_trainer.model_loader import load_video_vae_decoder, load_video_vae_encoder

    print("\n# Video VAE round-trip")
    try:
        enc = load_video_vae_encoder(video_vae, device=device, dtype=torch.float32)
        dec = load_video_vae_decoder(video_vae, device=device, dtype=torch.float32)
        x = torch.rand(1, 3, VIDEO_F, VIDEO_HW, VIDEO_HW, device=device) * 2 - 1
        with torch.inference_mode():
            z = enc(x)
            xr = dec(z, generator=torch.Generator(device).manual_seed(0))
        ok = True
        ok &= _ok("encode shape", tuple(z.shape) == EXP_VIDEO_LATENT, f"{tuple(z.shape)} (expect {EXP_VIDEO_LATENT})")
        ok &= _ok("latent finite", bool(torch.isfinite(z).all()))
        ok &= _ok("decode shape", tuple(xr.shape) == tuple(x.shape), f"{tuple(xr.shape)}")
        ok &= _ok("no meta buffers left", not any(b.is_meta for _, b in enc.named_buffers()))
        print(f"  [info] mean|abs recon err = {float((xr.float() - x.float()).abs().mean()):.4f} "
              "(random input is OOD for the VAE; informational only)")
        return ok
    except Exception as e:  # noqa: BLE001 - diagnostic: report, don't crash
        print(f"  [FAIL] video VAE round-trip errored: {type(e).__name__}: {e}")
        return False


def check_audio(audio_vae: str, vocoder: str | None, device: str) -> bool:
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_core.types import Audio
    from ltx_trainer.model_loader import load_audio_vae_decoder, load_audio_vae_encoder

    print("\n# Audio VAE round-trip")
    try:
        aenc = load_audio_vae_encoder(audio_vae, device=device, dtype=torch.float32)
        adec = load_audio_vae_decoder(audio_vae, device=device, dtype=torch.float32)
        proc = AudioProcessor(aenc.sample_rate, aenc.mel_bins, aenc.mel_hop_length, aenc.n_fft).to(device)
        sr = aenc.sample_rate
        wav = torch.rand(2, sr, device=device) * 2 - 1  # 1s stereo
        mel = proc.waveform_to_mel(Audio(wav.unsqueeze(0), sr))
        with torch.inference_mode():
            z = aenc(mel.float())
            mel_r = adec(z)
        ok = True
        ok &= _ok("encode channels==8", z.shape[1] == 8, f"{tuple(z.shape)}")
        ok &= _ok("encode mel_bins==16", z.shape[-1] == 16)
        ok &= _ok("encode T ≈ 25/s", abs(z.shape[2] - 25) <= 3, f"T={z.shape[2]} (~25 for 1s)")
        ok &= _ok("latent finite", bool(torch.isfinite(z).all()))
        ok &= _ok("decode mel shape", mel_r.shape == mel.shape, f"{tuple(mel_r.shape)}")
        if vocoder:
            from ltx_trainer.model_loader import load_vocoder

            voc = load_vocoder(vocoder, device=device)
            with torch.inference_mode():
                wav_r = voc(mel_r)
            print(f"  [info] vocoder out {tuple(wav_r.shape)} @ {getattr(voc, 'output_sampling_rate', '?')} Hz")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] audio VAE round-trip errored: {type(e).__name__}: {e}")
        return False


def check_gemma(gemma_path: str, device: str) -> bool:
    from ltx_trainer.model_loader import load_text_encoder

    print("\n# Gemma text encode")
    try:
        enc = load_text_encoder(gemma_path, device=device, dtype=torch.bfloat16)
        hidden, mask = enc.encode("a glowing shape pulsing on a dark background")
        return _ok("returns per-layer hidden states + mask",
                   isinstance(hidden, tuple) and len(hidden) > 1 and mask.dim() == 2,
                   f"{len(hidden)} layers, last {tuple(hidden[-1].shape)}, mask {tuple(mask.shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] Gemma encode errored: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-vae", default=None, help="Video VAE safetensors (ComfyUI split file)")
    ap.add_argument("--audio-vae", default=None, help="Audio VAE safetensors (ComfyUI split file)")
    ap.add_argument("--vocoder", default=None, help="Vocoder checkpoint (optional; for the audio decode tail)")
    ap.add_argument("--gemma", default=None, help="Gemma model dir (optional; checks text encode)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not (args.video_vae or args.audio_vae or args.gemma):
        ap.error("pass at least one of --video-vae / --audio-vae / --gemma")

    results = {}
    if args.video_vae:
        results["video VAE"] = check_video(args.video_vae, args.device)
    if args.audio_vae:
        results["audio VAE"] = check_audio(args.audio_vae, args.vocoder, args.device)
    if args.gemma:
        results["gemma"] = check_gemma(args.gemma, args.device)

    print("\n=== codec verification ===")
    for k, v in results.items():
        print(f"  {k:12s}: {'OK' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

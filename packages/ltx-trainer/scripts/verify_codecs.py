#!/usr/bin/env python
"""Round-trip diagnostic for the codecs the training pipeline relies on:
video VAE, audio VAE (+ vocoder), and the Gemma text encoder. Confirms each can
encode→decode and that shapes obey the model's relationships — so we trust the
latents we train on and the outputs we eval (not just understand the code).

Needs the LTX-2 safetensors checkpoint + a CUDA GPU (the conv stacks are heavy).
The VAEs are LOSSY, so this gates on SHAPE + finiteness (deterministic facts) and
REPORTS reconstruction error (informational) — it does not assert bit-equality.

    uv run python packages/ltx-trainer/scripts/verify_codecs.py \
        --checkpoint /path/to/ltx2.safetensors [--gemma /path/to/gemma] [--device cuda]

Code map (what each component is + where): internal/audio_iclora_codec_map.md.
Each stage is independent + guarded, so one failure doesn't hide the others.
"""

from __future__ import annotations

import argparse
import sys

import torch

# expected video latent relationship: F'=(F-1)//8+1, H'=H//32, W'=W//32
VIDEO_F, VIDEO_HW = 9, 64  # smallest valid: F=1+8k, HW div-by-32
EXP_VIDEO_LATENT = (1, 128, 2, 2, 2)


def _ok(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return cond


def check_video(ckpt: str, device: str) -> bool:
    from ltx_trainer.model_loader import load_video_vae_decoder, load_video_vae_encoder

    print("\n# Video VAE round-trip")
    try:
        enc = load_video_vae_encoder(ckpt, device=device, dtype=torch.bfloat16)
        dec = load_video_vae_decoder(ckpt, device=device, dtype=torch.bfloat16)
        x = (torch.rand(1, 3, VIDEO_F, VIDEO_HW, VIDEO_HW, device=device) * 2 - 1).to(torch.bfloat16)
        with torch.inference_mode():
            z = enc(x)
            xr = dec(z, generator=torch.Generator(device).manual_seed(0))
        ok = True
        ok &= _ok("encode shape", tuple(z.shape) == EXP_VIDEO_LATENT, f"{tuple(z.shape)} (expect {EXP_VIDEO_LATENT})")
        ok &= _ok("latent finite", bool(torch.isfinite(z).all()))
        ok &= _ok("decode shape", tuple(xr.shape) == tuple(x.shape), f"{tuple(xr.shape)}")
        err = float((xr.float() - x.float()).abs().mean())
        print(f"  [info] mean|abs reconstruction err = {err:.4f} (lossy VAE; informational)")
        return ok
    except Exception as e:  # noqa: BLE001 - diagnostic: report, don't crash the whole run
        print(f"  [FAIL] video VAE round-trip errored: {type(e).__name__}: {e}")
        return False


def check_audio(ckpt: str, device: str) -> bool:
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_core.types import Audio
    from ltx_trainer.model_loader import (
        load_audio_vae_decoder,
        load_audio_vae_encoder,
        load_vocoder,
    )

    print("\n# Audio VAE (+ vocoder) round-trip")
    try:
        aenc = load_audio_vae_encoder(ckpt, device=device, dtype=torch.float32)
        adec = load_audio_vae_decoder(ckpt, device=device, dtype=torch.bfloat16)
        voc = load_vocoder(ckpt, device=device)
        proc = AudioProcessor(aenc.sample_rate, aenc.mel_bins, aenc.mel_hop_length, aenc.n_fft).to(device)
        sr = aenc.sample_rate
        wav = torch.rand(2, sr, device=device) * 2 - 1  # 1s stereo
        mel = proc.waveform_to_mel(Audio(wav.unsqueeze(0), sr))
        with torch.inference_mode():
            z = aenc(mel.float())
            mel_r = adec(z.to(torch.bfloat16))
            wav_r = voc(mel_r)
        ok = True
        ok &= _ok("encode channels==8", z.shape[1] == 8, f"{tuple(z.shape)}")
        ok &= _ok("encode mel_bins==16", z.shape[-1] == 16)
        ok &= _ok("encode T ≈ 25/s", abs(z.shape[2] - 25) <= 3, f"T={z.shape[2]} (~25 for 1s)")
        ok &= _ok("latent finite", bool(torch.isfinite(z).all()))
        mel_err = float((mel_r.float() - mel.float()).abs().mean()) if mel_r.shape == mel.shape else -1.0
        print(f"  [info] mel-domain recon err = {mel_err:.4f} (compare in mel, not waveform)")
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
        ok = _ok("returns per-layer hidden states + mask",
                 isinstance(hidden, tuple) and len(hidden) > 1 and mask.dim() == 2,
                 f"{len(hidden)} layers, last {tuple(hidden[-1].shape)}, mask {tuple(mask.shape)}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] Gemma encode errored: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="LTX-2 safetensors (video+audio VAE + vocoder)")
    ap.add_argument("--gemma", default=None, help="Gemma model dir (optional; checks text encode)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results = {"video VAE": check_video(args.checkpoint, args.device),
               "audio VAE": check_audio(args.checkpoint, args.device)}
    if args.gemma:
        results["gemma"] = check_gemma(args.gemma, args.device)

    print("\n=== codec verification ===")
    for k, v in results.items():
        print(f"  {k:12s}: {'OK' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

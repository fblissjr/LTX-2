#!/usr/bin/env python
"""Verify reference-audio encode parity for the audio-reference IC-LoRA eval.

The trained audio-reference LoRA touches only ``audio_attn`` and learned to attend to
reference-audio tokens encoded one specific way: ``reference_audio.encode_reference_waveform``
(resample to the VAE's own sample rate -> dual-mono widen -> log-mel -> encoder -> ``[C, T, F]``
latent -> patchify to ``[T, C*F]`` tokens). If the eval encodes the reference even slightly
differently, the LoRA sees off-distribution tokens and produces garbage -- which looks nothing
like a "the reference does nothing" null. This harness measures the two ways the encode can drift,
entirely inside ltx-core (no ComfyUI runtime):

  1. VAE weights: training used the audio VAE *inside the full LTX-2.3 checkpoint*; the ComfyUI
     eval loads a *standalone* audio VAE file. Different weights -> different latents.
  2. Preprocessing: the VAE is 16 kHz. The AudioProcessor resamples to its own target rate
     internally, so a wrong resample target is a (near-lossless for a tone) double-resample --
     this quantifies how much that actually matters.

It encodes the same reference tone every relevant way and reports shape + max-abs-diff + relative
L2 + cosine similarity, and can write the canonical latent as a plain tensor file so the eval can
load it directly and skip re-encoding entirely (the parity-guaranteed path).

Run (paths are examples -- pass your own):
  uv run --group dev python packages/ltx-trainer/scripts/check_reference_encode_parity.py \
    --train-model-path /path/to/ltx-2.3-22b-distilled.safetensors \
    --eval-vae-path <comfyui_models>/vae/LTX2_audio_vae_bf16.safetensors \
    --reference /path/to/pitch_gate_tone.wav \
    --save-latent /path/to/ref_tone_canonical.pt
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from ltx_trainer.model_loader import load_audio_vae_encoder
from ltx_trainer.reference_audio import build_audio_processor, encode_reference_waveform


def _patchify(latent: torch.Tensor) -> torch.Tensor:
    """[C, T, F] -> [T, C*F] reference tokens (matches the ComfyUI node + lipdub layout)."""
    c, t, f = latent.shape
    return latent.permute(1, 0, 2).reshape(t, c * f)


def _compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """Report parity stats between two latents (upcast to fp32, align on the common time length)."""
    a = a.float().cpu()
    b = b.float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: {tuple(a.shape)} vs {tuple(b.shape)}")
        # align on the shorter time axis so we can still report value drift
        t = min(a.shape[1], b.shape[1])
        a, b = a[:, :t], b[:, :t]
    diff = (a - b).abs()
    denom = b.norm().item() or 1.0
    rel_l2 = (a - b).norm().item() / denom
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    scale = b.abs().mean().item()
    print(
        f"  [{name}] shape={tuple(a.shape)}  max|d|={diff.max():.4g}  mean|d|={diff.mean():.4g}  "
        f"(ref mean|x|={scale:.4g})  relL2={rel_l2:.4g}  cos={cos:.6f}"
    )


def _load_vae(model_path: str, device: torch.device):
    """Load the audio VAE encoder (fp32) + its matching processor + config, once per model.

    Kept separate from :func:`_encode` so a single loaded encoder can serve many encodes (the
    train VAE drives the canonical encode AND the resample-sensitivity probes); reloading the
    checkpoint's VAE per encode would be the bulk of the runtime.
    """
    encoder = load_audio_vae_encoder(model_path, device=device, dtype=torch.float32)
    processor = build_audio_processor(encoder).to(device)
    cfg = {
        "sample_rate": encoder.sample_rate,
        "mel_bins": encoder.mel_bins,
        "mel_hop_length": encoder.mel_hop_length,
        "n_fft": encoder.n_fft,
        "in_channels": getattr(encoder, "in_channels", None),
    }
    return encoder, processor, cfg


@torch.inference_mode()
def _encode(encoder, processor, wav: torch.Tensor, sr: int, pre_resample_to: int | None = None) -> torch.Tensor:
    """Encode ``wav`` the trainer's way with an already-loaded encoder/processor -> ``[C, T, F]``.

    ``pre_resample_to`` optionally resamples the waveform BEFORE the encode (the AudioProcessor
    then resamples again to the VAE's own rate) -- used to model an eval node that resamples to
    the wrong target first.
    """
    w, s = wav, sr
    if pre_resample_to is not None and pre_resample_to != sr:
        w = torchaudio.functional.resample(wav, sr, pre_resample_to)
        s = pre_resample_to
    return encode_reference_waveform(encoder, processor, w, s)["latents"].float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-model-path", required=True, help="Checkpoint training encoded the reference with (full LTX-2.3 .safetensors).")
    ap.add_argument("--eval-vae-path", default=None, help="Standalone audio VAE the ComfyUI eval loads (optional; tests VAE-weights parity).")
    ap.add_argument("--reference", required=True, type=Path, help="Reference tone WAV to encode.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-latent", default=None, type=Path, help="Write the canonical [C,T,F] latent here (.pt) for a re-encode-free eval.")
    args = ap.parse_args()
    device = torch.device(args.device)

    wav, sr = torchaudio.load(str(args.reference))
    print(f"reference: {args.reference.name}  waveform={tuple(wav.shape)} @ {sr} Hz  ({wav.shape[-1] / sr:.2f}s)")

    # --- ground truth: training's VAE (from the full checkpoint), the trainer's exact encode ---
    enc_train, proc_train, cfg = _load_vae(args.train_model_path, device)
    latent_train = _encode(enc_train, proc_train, wav, sr)
    print(f"\ntraining audio VAE config: {cfg}")
    print(f"canonical reference latent [C,T,F] = {tuple(latent_train.shape)}; tokens [T, C*F] = {tuple(_patchify(latent_train).shape)}")
    print(f"  stats: mean={latent_train.mean():.4g}  std={latent_train.std():.4g}  min={latent_train.min():.4g}  max={latent_train.max():.4g}")

    print("\n=== parity checks (vs the canonical training encode) ===")

    # Axis 1: VAE weights -- standalone eval VAE vs training's in-checkpoint VAE.
    if args.eval_vae_path:
        try:
            enc_eval, proc_eval, cfg_eval = _load_vae(args.eval_vae_path, device)
            latent_eval = _encode(enc_eval, proc_eval, wav, sr)
            if cfg_eval != cfg:
                print(f"  [eval-VAE config] DIFFERS: {cfg_eval}")
            _compare("VAE-weights: eval standalone vs training in-checkpoint", latent_train, latent_eval)
        except Exception as e:  # noqa: BLE001 -- diagnostic tool, report and continue
            print(f"  [VAE-weights] could not load/encode with --eval-vae-path: {type(e).__name__}: {e}")

    # Axis 2: preprocessing sensitivity -- wrong resample target (e.g. a node defaulting to 44100).
    # Reuses the already-loaded train encoder; only the waveform's pre-resample changes.
    for wrong in (44100, 48000):
        latent_wrong = _encode(enc_train, proc_train, wav, sr, pre_resample_to=wrong)
        _compare(f"preprocess: pre-resampled to {wrong} Hz then encoded", latent_train, latent_wrong)

    if args.save_latent:
        torch.save({"samples": latent_train.unsqueeze(0)}, args.save_latent)  # [1, C, T, F]
        print(f"\nwrote canonical latent -> {args.save_latent}  (load this directly to skip eval re-encode)")

    print(
        "\nverdict guide: cos ~1.0 / relL2 ~0 => that axis is NOT the garbage cause. "
        "cos well below 1 or relL2 large => off-distribution reference; fix that encode path "
        "(or feed the saved canonical latent directly)."
    )


if __name__ == "__main__":
    main()

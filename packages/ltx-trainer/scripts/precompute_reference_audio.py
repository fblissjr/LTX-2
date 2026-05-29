#!/usr/bin/env python
"""Precompute the ``reference_audio_latents`` channel for audio-reference IC-LoRA training.

Reads a JSONL manifest pairing each clip (video) with a reference audio WAV (e.g. a
voiced tone at the target pitch), encodes each WAV through the LTX audio VAE encoder
(same encoder + dtype as the ``audio_latents`` AV precompute), and writes the latent to
``<output-dir>/<clip-rel>.pt`` — the same relative path as the clip's video latent, so
``PrecomputedDataset`` pairs them. The .pt format matches ``audio_latents`` exactly, so
``AudioReferenceStrategy`` consumes the reference identically to the target audio.

Example:
    uv run python packages/ltx-trainer/scripts/precompute_reference_audio.py \
        --model-path <ltx2_checkpoint>.safetensors \
        --manifest <dataset>/manifest.jsonl \
        --data-root <dataset> \
        --output-dir <dataset>/precomputed/reference_audio_latents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from ltx_trainer import logger
from ltx_trainer.model_loader import load_audio_vae_encoder
from ltx_trainer.reference_audio import (
    build_audio_processor,
    encode_reference_waveform,
    reference_output_path,
)


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="LTX-2 checkpoint (.safetensors) holding the audio VAE")
    ap.add_argument("--manifest", required=True, type=Path, help="JSONL: one row per clip with video + reference keys")
    ap.add_argument("--data-root", required=True, type=Path, help="Root the manifest paths are relative to")
    ap.add_argument("--output-dir", required=True, type=Path, help="Destination reference_audio_latents/ directory")
    ap.add_argument("--video-key", default="video", help="Manifest field with the clip (video) path")
    ap.add_argument("--reference-key", default="reference_audio", help="Manifest field with the reference WAV path")
    ap.add_argument("--device", default="cuda", help="Device for the audio VAE encoder")
    ap.add_argument("--overwrite", action="store_true", help="Re-encode even if the output .pt already exists")
    args = ap.parse_args()

    device = torch.device(args.device)
    encoder = load_audio_vae_encoder(args.model_path, device=device, dtype=torch.bfloat16)
    processor = build_audio_processor(encoder)

    rows = _read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Encoding {len(rows)} reference audio clips -> {args.output_dir}")

    n_done = 0
    for row in rows:
        dst = reference_output_path(row[args.video_key], args.output_dir, data_root=args.data_root)
        if dst.is_file() and not args.overwrite:
            continue

        ref_path = Path(row[args.reference_key])
        if not ref_path.is_absolute():
            ref_path = args.data_root / ref_path
        waveform, sample_rate = torchaudio.load(str(ref_path))

        with torch.inference_mode():
            out = encode_reference_waveform(encoder, processor, waveform, sample_rate)

        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": out["latents"].cpu().contiguous(),
                "num_time_steps": out["num_time_steps"],
                "frequency_bins": out["frequency_bins"],
                "duration": out["duration"],
            },
            dst,
        )
        n_done += 1

    logger.info(f"Done: wrote {n_done} reference latents ({len(rows) - n_done} skipped/existing) to {args.output_dir}")


if __name__ == "__main__":
    main()

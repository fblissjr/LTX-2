"""Scripted audio-reference eval sweep on the unified (FlexibleStrategy) architecture.

Renders ``<references> x <seeds>`` audio-reference generations from a LoRA-fused base by driving the
trainer's :class:`ValidationRunner` with ``reference`` audio conditions set to
``audio_positions_mode: lipdub_negative`` — inference-parity with the shipped LipDub / ComfyUI
audio-reference path (locked byte-equal in ``tests/test_flexible_audio_reference_parity.py``).

The swap contract: with a FIXED seed and prompt, only the reference varies across an arm group, so
differences in the output are attributable to the reference. Each ``(reference, seed)`` arm is one
``ValidationSample`` carrying that seed (``ValidationRunner`` uses ``sample.seed``).

Model load + LoRA fusion mirror the canonical ltx-pipelines inference path exactly:
``SingleGPUModelBuilder(..., model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP).lora(path, strength,
LTXV_LORA_COMFY_RENAMING_MAP)`` — the trainer saves adapters in that ComfyUI (``diffusion_model.``)
format, so the LoRA renaming map matches.

Two things this script does that differ from the pre-refactor lane, called out deliberately:
  1. Reference audio is encoded from FILES by ``ValidationRunner`` (its audio VAE), not read from
     precomputed ``.pt`` latents. Ensure the eval VAE precision matches the training encode.
  2. Generation reuses upstream's ``ValidationRunner`` loop (not a bespoke sampler), so it stays in
     step with upstream.

This renders from the full model; run it on the GPU box against a real checkpoint. The pure
config-building half (:func:`build_sweep_config`) is unit-tested; the render is not (no in-repo GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ltx_trainer import logger
from ltx_trainer.config import ReferenceConditionConfig, ValidationConfig, ValidationSample
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


def build_sweep_config(
    *,
    prompt: str,
    negative_prompt: str,
    references: list[Path],
    seeds: list[int],
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    inference_steps: int,
    guidance_scale: float,
    stg_scale: float,
    stg_blocks: list[int],
    stg_mode: str,
) -> ValidationConfig:
    """Build the sweep as a ``ValidationConfig``: one ``reference`` audio sample per (ref, seed).

    Each sample pins ``audio_positions_mode='lipdub_negative'`` so the eval geometry matches how the
    IC-LoRA was trained. Pure/deterministic (no I/O) so it is unit-testable.
    """
    samples = [
        ValidationSample(
            prompt=prompt,
            seed=seed,
            conditions=[
                ReferenceConditionConfig(audio=str(ref), audio_positions_mode="lipdub_negative")
            ],
        )
        for ref in references
        for seed in seeds
    ]
    return ValidationConfig(
        samples=samples,
        negative_prompt=negative_prompt,
        video_dims=(width, height, num_frames),
        frame_rate=frame_rate,
        seed=seeds[0],
        inference_steps=inference_steps,
        interval=1,  # not used by a standalone run(); kept > 0 for clarity
        guidance_scale=guidance_scale,
        stg_scale=stg_scale,
        stg_blocks=stg_blocks,
        stg_mode=stg_mode,
        generate_audio=True,
        generate_video=False,
        skip_initial_validation=False,
    )


def load_lora_fused_transformer(
    *,
    model_path: Path,
    lora_path: Path,
    lora_strength: float,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build a LoRA-fused LTX transformer, mirroring the ltx-pipelines inference path."""
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    builder = SingleGPUModelBuilder(
        model_path=str(model_path),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).lora(str(lora_path), lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)
    return builder.build(device=device, dtype=dtype)


def _collect_references(inputs: list[str]) -> list[Path]:
    """Expand file/dir args into a sorted, de-duplicated list of audio files."""
    refs: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            refs.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in _AUDIO_EXTENSIONS))
        elif p.is_file():
            refs.append(p)
        else:
            raise FileNotFoundError(f"Reference audio path does not exist: {p}")
    # De-dup preserving order.
    seen: set[Path] = set()
    unique = [r for r in refs if not (r in seen or seen.add(r))]
    if not unique:
        raise ValueError("No reference audio files found in the provided --reference-audio inputs.")
    return unique


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="Base LTX-2 .safetensors checkpoint.")
    parser.add_argument("--text-encoder-path", type=Path, required=True, help="Gemma text-encoder directory.")
    parser.add_argument("--lora", type=Path, required=True, help="Trained audio-reference IC-LoRA .safetensors.")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument(
        "--reference-audio", nargs="+", required=True, help="Reference audio file(s) and/or directory(ies)."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds; fixed across refs = the swap.")
    parser.add_argument("--prompt", required=True, help="Neutral caption (the reference is the controller).")
    parser.add_argument("--negative-prompt", default="worst quality, distorted, noisy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=81, help="frames %% 8 == 1; sets audio duration.")
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--stg-blocks", nargs="+", type=int, default=[29])
    parser.add_argument("--stg-mode", default="stg_av")
    parser.add_argument("--load-text-encoder-in-8bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    references = _collect_references(args.reference_audio)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_sweep_config(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        references=references,
        seeds=args.seeds,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_mode=args.stg_mode,
    )
    # sample_idx -> (reference, seed), matching build_sweep_config's exact (ref, seed) nesting.
    arms = [(ref, seed) for ref in references for seed in args.seeds]

    n_arms = len(config.samples)
    logger.info(f"Sweep: {len(references)} references x {len(args.seeds)} seeds = {n_arms} arms -> {args.output_dir}")

    logger.info("Loading LoRA-fused transformer...")
    transformer = load_lora_fused_transformer(
        model_path=args.model_path,
        lora_path=args.lora,
        lora_strength=args.lora_strength,
        device=device,
    )

    logger.info("Building ValidationRunner (loads text encoder, encodes references, loads decoders)...")
    runner = ValidationRunner(
        config=config,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
    )

    progress = TrainingProgress(enabled=True, total_steps=1)
    with progress:
        results = runner.run(
            transformer=transformer,
            step=0,
            output_dir=args.output_dir,
            device=device,
            progress=progress,
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample_idx, path in results:
            ref, seed = arms[sample_idx]
            manifest.write(
                json.dumps(
                    {
                        "arm": f"{ref.stem}_seed{seed}",
                        "reference": str(ref),
                        "seed": seed,
                        "prompt": args.prompt,
                        "output": str(path),
                    }
                )
                + "\n"
            )
    logger.info(f"Wrote {len(results)} arms + manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

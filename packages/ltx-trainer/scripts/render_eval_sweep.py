#!/usr/bin/env python
"""Scripted render sweep for audio-input evals — the automated eval lane.

Loads the model ONCE, then renders every (audio input x seed) arm of a sweep and writes
an mp4 (+ wav) per arm plus a manifest.jsonl. Built for the audio-reference IC-LoRA
swap eval: fixed prompt, fixed seed ACROSS references — swap only the reference and
score what changes (seed variance is large, so 3-5 seeds per config; that volume is why
this exists instead of hand-driving a UI).

Two mutually exclusive audio lanes (see GenerationConfig in validation_sampler.py):
  --reference-audio  in-context reference at negative RoPE; audio+video both GENERATED
                     (the audio-reference IC-LoRA shape; the trained LoRA reads it)
  --input-audio      audio stream FROZEN to the clip; only video generated
                     (the audio->video coupling shape)

Inputs are precompute-format .pt files ([C,T,F] latents, dicts, or [K,C,T,F] variant
stacks — see eval_utils.load_audio_latents_file), so a sweep can point at the exact
files training consumed.

NOTE on regimes: this is the automated lane. Manual ComfyUI evals continue, and when the
two regimes disagree about a checkpoint ranking, the ComfyUI path is the shipped regime
and arbitrates.

Usage (audio-reference swap eval, distilled base, 3 seeds x 2 references):
  render_eval_sweep.py \
    --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors \
    --text-encoder-path /path/to/gemma \
    --lora-path /path/to/audio_ref_lora.safetensors \
    --reference-audio refA.pt refB.pt \
    --seeds 0,1,2 \
    --prompt "a person speaking" \
    --out-dir outputs/swap_eval

The checkpoint must be the COMBINED file (embedded vae config) — a stripped standalone
VAE file builds phantom timestep-conditioning params and crashes decode on meta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Sibling-script import (scripts/ is not a package; inference.py owns the LoRA loader).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_lora_weights  # noqa: E402

from ltx_trainer.eval_utils import load_audio_latents_file  # noqa: E402
from ltx_trainer.model_loader import load_embeddings_processor, load_model  # noqa: E402
from ltx_trainer.progress import StandaloneSamplingProgress  # noqa: E402
from ltx_trainer.validation_sampler import (  # noqa: E402
    CachedPromptEmbeddings,
    GenerationConfig,
    ValidationSampler,
)
from ltx_trainer.video_utils import save_video  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scripted render sweep for audio-input evals.")
    ap.add_argument("--checkpoint", required=True, help="COMBINED model checkpoint (.safetensors)")
    ap.add_argument("--text-encoder-path", required=True, help="Gemma model directory")
    ap.add_argument("--lora-path", default=None, help="optional LoRA checkpoint to apply")
    lane = ap.add_mutually_exclusive_group(required=True)
    lane.add_argument("--reference-audio", nargs="+", default=None, help="reference-audio .pt files (IC reference lane)")
    lane.add_argument("--input-audio", nargs="+", default=None, help="input-audio .pt files (frozen-stream lane)")
    ap.add_argument("--variant", type=int, default=0, help="variant index for [K,C,T,F] reference stacks")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (fixed across inputs = the swap contract)")
    ap.add_argument("--prompt", default="a person speaking", help="keep neutral/attribute-free for reference evals")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=25.0)
    ap.add_argument("--num-inference-steps", type=int, default=8, help="8 = the distilled schedule")
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="1.0 = distilled (CFG off)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def _cache_prompt_embeddings(
    text_encoder, processor, prompt: str, negative_prompt: str, device: str, need_negative: bool
) -> CachedPromptEmbeddings:
    """Encode the (constant) prompt once so Gemma can be freed before the sweep loop.

    The negative encode is skipped at guidance_scale == 1.0 (the distilled regime) —
    mirroring the sampler's own _encode_prompts. NOTE: inference.py's on-the-fly path never
    loads an embeddings_processor at all (a latent bug there); caching is both the fix and
    the cheaper design for a constant-prompt sweep.
    """
    text_encoder.to(device)
    processor.to(device)
    pos_hs, pos_mask = text_encoder.encode(prompt)
    pos = processor.process_hidden_states(pos_hs, pos_mask)
    neg = None
    if need_negative:
        neg_hs, neg_mask = text_encoder.encode(negative_prompt)
        neg = processor.process_hidden_states(neg_hs, neg_mask)
    return CachedPromptEmbeddings(
        video_context_positive=pos.video_encoding.cpu(),
        audio_context_positive=pos.audio_encoding.cpu(),
        video_context_negative=neg.video_encoding.cpu() if neg is not None else None,
        audio_context_negative=(
            neg.audio_encoding.cpu() if neg is not None and neg.audio_encoding is not None else None
        ),
    )


def main() -> None:
    args = _parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    input_paths = [Path(p) for p in (args.reference_audio or args.input_audio)]
    lane = "reference_audio" if args.reference_audio else "input_audio"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load every audio input up front (fail fast on a bad path/shape, before model load).
    audio_inputs = {p: load_audio_latents_file(p, variant=args.variant) for p in input_paths}

    components = load_model(
        checkpoint_path=args.checkpoint,
        device="cpu",  # load to CPU; the sampler moves pieces to the device as needed
        dtype=torch.bfloat16,
        with_video_vae_encoder=False,  # no image/video conditioning in this lane
        with_video_vae_decoder=True,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
        text_encoder_path=args.text_encoder_path,
    )
    transformer = components.transformer

    # Encode the constant prompt ONCE on the device (GPU is otherwise empty at this point),
    # then free Gemma + the processor — the whole sweep runs without them.
    processor = load_embeddings_processor(args.checkpoint, device="cpu", dtype=torch.bfloat16)
    cached = _cache_prompt_embeddings(
        components.text_encoder,
        processor,
        args.prompt,
        args.negative_prompt,
        args.device,
        need_negative=args.guidance_scale != 1.0,
    )
    components.text_encoder = None
    del processor
    torch.cuda.empty_cache()

    if args.lora_path is not None:
        transformer = load_lora_weights(transformer, args.lora_path)

    sampler = ValidationSampler(
        transformer=transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=None,
        audio_decoder=components.audio_vae_decoder,
        vocoder=components.vocoder,
    )
    audio_sample_rate = components.vocoder.output_sampling_rate

    manifest_path = out_dir / "manifest.jsonl"
    n_arms = len(audio_inputs) * len(seeds)
    print(f"Sweep: {len(audio_inputs)} {lane} inputs x {len(seeds)} seeds = {n_arms} arms -> {out_dir}")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for path, latents in audio_inputs.items():
            for seed in seeds:
                arm = f"{path.stem}_v{args.variant}_seed{seed}"
                config = GenerationConfig(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                    generate_audio=True,
                    cached_embeddings=cached,
                    **{f"{lane}_latents": latents},
                )
                with StandaloneSamplingProgress(num_steps=args.num_inference_steps) as progress:
                    sampler._sampling_context = progress
                    video, audio = sampler.generate(config=config, device=args.device)

                video_path = out_dir / f"{arm}.mp4"
                save_video(
                    video_tensor=video,
                    output_path=video_path,
                    fps=args.frame_rate,
                    audio=audio,
                    audio_sample_rate=audio_sample_rate if audio is not None else None,
                )
                manifest.write(
                    json.dumps(
                        {
                            "arm": arm,
                            "output": video_path.name,
                            "lane": lane,
                            "audio_input": str(path),
                            "variant": args.variant,
                            "seed": seed,
                            "lora": args.lora_path,
                            "checkpoint": args.checkpoint,
                            "prompt": args.prompt,
                            "steps": args.num_inference_steps,
                            "guidance_scale": args.guidance_scale,
                            "size": [args.width, args.height, args.num_frames, args.frame_rate],
                        }
                    )
                    + "\n"
                )
                manifest.flush()
                print(f"  [done] {arm}")

    print(f"\nSweep complete: {n_arms} arms, manifest at {manifest_path}")


if __name__ == "__main__":
    main()

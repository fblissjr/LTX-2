Last updated: 2026-05-29

# Audio→video IC-LoRA — trainer-side notes (experimental)

Experimental notes on the trainer changes in this LTX-2 fork that support training a
small audio-conditioned IC-LoRA on the 22B distilled model **on a single 24 GB
consumer GPU**. This is the training half of a two-repo effort; the data design,
ComfyUI inference, and eval live in the companion repo (a ComfyUI custom-node pack —
its `docs/experimental/audio_iclora_method_notes.md` is the top-level writeup and is
honest about what does and doesn't work).

Voice here matches that writeup: observed facts or explicit guesses, no hype. We
have **not** demonstrated a working audio IC-LoRA. This doc describes the machinery
that lets someone try.

## The one thing that matters most: it fits on a 4090

The 22B distilled LTX-2 model does not fit on 24 GB for LoRA training the naive way.
What makes it fit:

- **Block-swap** (`packages/ltx-trainer/src/ltx_trainer/block_swap.py`). Most
  transformer blocks are wrapped in a `StreamingBlockWrapper` and streamed between
  CPU and GPU during the forward/backward pass; only a handful stay resident.
  `attach_block_swap(..., block_swap_blocks=36)` was the configuration we ran.
- **int8 quantization** (optimum-quanto) of the base weights.
- **Gradient checkpointing** + an 8-bit optimizer.

Observed envelope on a 4090: VRAM ~21.7 GB at attach dropping to ~8.9 GB resident
after the swap setup, ~17 GB peak during training, ~43 min for 300 steps (~8.7
s/step). These are single-run observations, not benchmarks.

A wrinkle worth knowing: block-swap serializes a spurious `.block.` segment into the
saved LoRA key names (an artifact of the wrapper). `strip_block_swap_prefix()` is
called in the save path so checkpoints load in ComfyUI without a converter. (The
companion repo also ships a standalone converter as a fallback.)

## What "audio guides video" means in the strategy

Training reuses the existing `video_to_video` IC-LoRA strategy in
`audio_mode: condition`:

- The audio latent is fed **clean** (as context, conditioning timestep 0); the loss
  is on the **video**. So the gradient flows through the audio→video cross-modal
  path while the model learns to reconstruct video that matches the frozen audio.
- This is **not** a separate "audio reference" strategy. The in-context reference is
  a *video* (or, in our final dataset, a static frame); the audio enters through the
  model's native audio stream, not as the IC reference. A true audio-reference
  variant (audio as the in-context signal, LipDub-style) would need a different data
  path and is not built here.
- Config: `packages/ltx-trainer/configs/ltx2_audio_coupling_ic_lora.yaml`. The
  load-bearing choice is `target_modules`: **audio-side modules only** (audio
  self-attn, audio cross-attn, audio FFN, and the audio→video bridge), explicitly
  **not** the broad `to_k/to_q/to_v/to_out.0` set. The why is in the config's own
  header comment and the companion repo's notes: a broad set on an audio-video
  checkpoint spends adapter capacity on video self-attn and was observed to *degrade*
  the base model's working audio coupling.

## Synthetic data generators (`synthetic_av.py`)

Procedural, CPU-only, no model. Three dataset builders, in the order we learned to
need them (the companion repo's notes explain the reasoning and the two leaks we hit):

- `generate_dataset` — v1, reference = target. **Known-broken for this task** (lets
  the model copy the reference and ignore audio). Kept for the integration smoke
  test; do not train a real run on it.
- `generate_dataset_paired_refs` — v2, reference ≠ target with different BPM. Better,
  but in a narrow BPM range the reference/target BPMs become correlated (a measured
  leak). Superseded.
- `generate_dataset_static_ref` — v3, frozen identity reference (no pulse, no audio).
  The reference can't carry a rate, so audio is the only time-varying signal. This is
  what we trained.

Helpers: `generate_beat_pulse_clip` (the target — a shape pulsing on a click track),
`generate_static_identity_clip` (the frozen reference). `measure_pulse_rate` recovers
BPM from per-frame brightness, so the coupling is in principle measurable.

A representability caveat baked into the data defaults: the video VAE compresses ~8×
in time, so a per-beat pulse aliases above ~`fps × 3.75` ≈ 94 BPM at 25 fps. The v3
default BPM range is sub-Nyquist (≤85) for that reason.

## Data validation before you burn GPU

`verify_training_data.py` runs on the precomputed dataset (output of
`process_dataset.py --with-audio`) and checks source counts, pairing, and shapes —
catches an integration gap before a training run rather than after.

## Honest status

- The machinery runs: data → precompute → validate → train → save → load.
- Whether the resulting LoRA actually tightens audio→video coupling **beyond the
  base model's already-present native reactivity is not demonstrated.** Every
  inference render so far is confounded (the base is already audio-reactive). The
  companion repo's notes lay out the controlled eval that would settle it and why
  the chosen synthetic task (beat→pulse) is a poor mechanism-prover.
- If forking: the block-swap fit and the data-independence discipline are the
  reusable parts. The open work is a clean eval, an audio-as-IC-reference node, and
  probably a task the base model doesn't already do.

## Pointers (code)

- `block_swap.py` — `attach_block_swap`, `StreamingBlockWrapper`, `strip_block_swap_prefix`
- `synthetic_av.py` — the three `generate_dataset*` builders + clip renderers
- `data_validation.py` / `scripts/verify_training_data.py` — pre-train validator
- `configs/ltx2_audio_coupling_ic_lora.yaml` — the trained config (audio-only targets)
- `trainer.py` — training loop (reads precomputed latents; no VAE in the loop)

This is a fork of Lightricks' LTX-2 training code; these notes cover only the
audio-IC-LoRA-specific additions.

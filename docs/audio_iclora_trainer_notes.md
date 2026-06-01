Last updated: 2026-06-01

# Audio-only IC-LoRA — trainer-side notes (experimental)

Notes on the trainer changes in this LTX-2 fork that support training an audio-only IC-LoRA on the
22B distilled model **on a single 24 GB consumer GPU**. "Audio-only" means the in-context reference
is an audio clip: no image, no video, no init frame. The model still generates audio and video
jointly; only the reference is audio. This is the training half of a two-repo effort; the ComfyUI
inference nodes, the data build, and the eval live in the companion node pack (ComfyUI-AudioLoopHelper,
`docs/audio_iclora/`). Released checkpoints + the full model card:
[fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context](https://huggingface.co/fbjr/LTX-2.3-22b-IC-LoRA-Audio-Only-Context).

Voice here is observed-facts-or-explicit-guesses, no hype. Short version of the result: the audio
reference **does** steer generation (the generated speech and the speaker's mannerisms follow it), but
it has not been through a controlled quantitative eval, so treat it as a working-looking proof of
concept, not a benchmarked capability.

## It fits on a 4090

The 22B distilled model does not fit on 24 GB for LoRA training the naive way. What makes it fit:

- **Block-swap** (`block_swap.py`): most transformer blocks are wrapped in a `StreamingBlockWrapper`
  and streamed CPU<->GPU during forward/backward; a handful stay resident. We ran `block_swap_blocks=36`.
- **int8 quantization** (optimum-quanto) of the base weights.
- **Gradient checkpointing** + an 8-bit optimizer.

Observed at 512x512x121: ~22 GB at attach dropping to ~9 GB resident after the swap setup, ~20.6 GB
peak during training, ~11.5 s/step. `batch_size > 1` OOMs at this resolution (block-swap is near its
floor), so single-sample steps are the ceiling. A wrinkle: block-swap serializes a spurious `.block.`
segment into saved LoRA key names; `strip_block_swap_prefix()` runs in the save path so checkpoints
load in ComfyUI without a converter.

## The `audio_reference` strategy (audio AS the in-context reference)

`training_strategies/audio_reference.py`. The audio stream is `[target (noised) | reference (clean)]`:
the target audio leads and stays in the loss; the reference is appended **clean** at distinct
**negative** RoPE positions as out-of-timeline context (the LipDub / ID-LoRA convention), and is
excluded from the loss. The target is audio+video; the video is generated from noise and follows the
(reference-influenced) audio through the joint model. Train/inference RoPE parity is load-bearing and
locked by a test.

**Two cuts (the only difference between the two released checkpoints), set by `lora.target_modules`:**
- **audio-only:** `audio_attn1/2`, `audio_ff`. The reference shapes the generated audio; the video
  follows via the frozen base coupling (subtler video effect, smallest footprint on the base).
- **cross-modal:** the above **plus** `audio_to_video_attn` / `video_to_audio_attn`. The bridge is the
  only path the audio reference reaches the video stream, so adapting it couples the reference into the
  video more strongly. Example config: `configs/ltx2_audio_reference.yaml`.

(There is also an older `video_to_video` `audio_mode: condition` path for coupling experiments; the
audio-only-reference work above supersedes it for this task.)

## Observability, and the metric that fooled us

Built so we are not blind the way the first run was (`metrics.py` + the trainer loop):
- **Held-out val loss** every `validation.holdout_interval` steps on a disjoint, held-out-by-identity
  set. Train falling while val flattens/rises = overfitting. Fed to a `ConvergenceMonitor` that can
  early-stop (`optimization.early_stop_on_convergence`).
- **Reference-attribution gap (`ref_gap`)**: loss with the correct reference minus loss with a wrong
  one, under paired noise so the difference isolates the reference. Positive = the reference helped.
- Always-on `metrics.jsonl` (the durable record; do not rely on W&B alone, and note W&B SDK 0.24.0
  has a `finish()` hang — keep it off until upgraded).

**The lesson worth carrying:** for an *identity* task, `ref_gap` reads ~0 at every noise level **by
construction**, and that is not a verdict that the model failed. `ref_gap` measures whether the
reference helps *reconstruct the target*, and the target video already shows the face, so the model
never needs the reference to reconstruct it (the "leak"). The clean test is **generation from noise**
(swap only the reference and watch the output), which is the companion repo's eval, not a
reconstruction-loss number. `ref_gap` is still useful as an overfit probe and would be informative for
an attribute the video *can't* leak (e.g. pitch, or audio->audio).

## Recipe (data-independent discipline)

The seesaw: for the audio reference to control an attribute (not the caption), the reference and target
must **differ in content and share only that attribute**, caption neutral. For identity: same speaker,
different clip, neutral caption. Reference is a fixed 3.5 s window from a different clip of the same
speaker, appended clean. Precompute targets with `process_dataset.py --with-audio` and the reference
windows separately; train with `audio_reference`. To adapt to a new attribute, change only the pairing.

## Honest status / next

- Machinery runs end to end: data -> precompute -> validate -> train (held-out + ref_gap) -> save -> load.
- The reference visibly steers generation at default strength; **no controlled quantitative eval yet.**
- The leading next lever is training-side: the target leaks the attribute, so the model has little
  pressure to use the reference. Mask the target face region and/or bias timesteps toward high sigma to
  force reliance, then re-eval with the swap protocol.

## Pointers (code)

- `training_strategies/audio_reference.py` — the strategy; `block_swap.py` — the 4090 fit.
- `metrics.py` (`ConvergenceMonitor`, `emit_if_fresh`, `paired_difference`), `lr_schedulers.py` (cosine + warmup),
  trainer held-out/ref-gap/early-stop loop.
- `scripts/reference_gap_by_sigma.py` — sigma-resolved ref-gap canary; `scripts/replay_metrics_to_wandb.py` — re-log metrics.jsonl.
- `configs/ltx2_audio_reference.yaml` — example config.

### Earlier work (superseded)

The first audio-coupling experiments used procedural synthetic data (a shape pulsing on a click track,
`synthetic_av.py`) and the `video_to_video` `audio_mode: condition` path, to test whether a LoRA could
tighten audio->video coupling beyond the base model's native reactivity. That task was a poor
mechanism-prover (the base is already audio-reactive) and the result was confounded; the audio-only
in-context-reference work above replaced it.

This is a fork of Lightricks' LTX-2 training code; these notes cover only the audio-IC-LoRA additions.

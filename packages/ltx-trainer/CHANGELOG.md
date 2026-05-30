# Changelog

All notable changes to the LTX trainer (this fork) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses semantic versioning.

## [Unreleased]

### Added

- Training telemetry (`metrics.py`): an always-on `metrics.jsonl` curve written every
  optimization step (loss, EMA loss, gradient norm, lr, step-time, per-sigma-bucket losses) —
  durable and greppable even when the Rich progress bar is the only live display and stdout is
  redirected (the gap that left an unattended run without a curve). Adds a `ConvergenceMonitor`
  (under-fit / plateau / converged / reference-decorative; overfit is accepted forward-compatibly,
  pending a held-out-loss forward) that logs warnings during the run and an end-of-run verdict. The
  pre-clip gradient norm (previously discarded) is now captured and surfaced on the progress bar
  (`|g|`), in the JSONL, and in W&B.
- Reference-attribution gap: at checkpoint cadence, two extra no-grad forwards measure
  `loss(wrong reference) − loss(correct reference)` under paired noise (`metrics.paired_difference`
  pins the RNG so the gap reflects only the reference, not sampling variance, and leaves the
  training RNG stream untouched). It is the training-time twin of the inference "remove the
  reference → output stops tracking" check — `> 0` means the reference is load-bearing, `~0` means
  decorative — and feeds the monitor's reference-decorative flag plus the `ref_gap` metric. The
  "wrong" reference is the prior step's clip (batch size 1).
- `audio_reference` training strategy — an **audio-only IC-LoRA** (transfer paradigm):
  an in-context reference *audio* clip steers an attribute of the jointly-generated
  audio+video. The audio stream is `[target (noised) | reference (clean)]` with the
  reference at negative RoPE positions matching the inference convention
  (`ltx_pipelines.lipdub`); loss is on the target audio + the generated video, the
  reference excluded. Example config `configs/ltx2_audio_reference.yaml`.
  **The first learning run was trained on this strategy** against the distilled
  LTX-2.3 22B with a pitch-reference dataset (a voiced reference tone → the generated
  audio adopts that pitch); the F0-tracking eval runs through ComfyUI. Real-22B
  run+fit verified (int8-quanto + block-swap fits a 24 GB card).
- `scripts/precompute_reference_audio.py` + `reference_audio.py`: build the
  `reference_audio_latents` channel by encoding reference WAVs through the audio VAE,
  in the same `[C, T, F]` format as `audio_latents` and paired to each clip by
  relative path so `PrecomputedDataset` joins them.
- Audio-coupling IC-LoRA training config (`configs/ltx2_audio_coupling_ic_lora.yaml`):
  a `video_to_video` condition-mode recipe whose LoRA targets only the audio
  self/cross attention, audio feed-forward, and the audio→video cross-modal
  bridge, to tighten the model's native audio→video coupling without perturbing
  the frozen video branch.
- `strip_block_swap_prefix` in `block_swap.py`, applied automatically in the LoRA
  save path: rewrites the `transformer_blocks.<N>.block.` wrapper segment that
  block-swap injects into swapped-block keys, so checkpoints trained with
  block-swap load directly in inference engines without an external key
  converter. Idempotent — a no-op when block-swap is off.
- Reference-equals-target detection in the precomputed-data validator: warns when
  a reference latent is byte-identical to its target, a degenerate IC-LoRA setup
  where the reference leaks the answer and the conditioning cannot become
  load-bearing.
- `scripts/probe_vae_temporal_aliasing.py`: decodes precomputed target latents and
  compares the recovered periodic-signal rate to the expected rate, to
  characterize the video VAE's temporal-aliasing ceiling on periodic visual
  signals (read-only diagnostic).

### Fixed

- Mono audio now encodes through the audio VAE: the VAE expects a 2-channel (stereo)
  mel, so mono speech/tones previously crashed the audio precompute. `ensure_audio_channels`
  dual-mono-widens (and mean-downmixes) to the encoder's input channel count, applied in
  both the target-audio (`process_videos`) and reference-audio encodes.
- The reference-audio precompute loads the audio VAE in float32 and moves the mel
  processor on-device, matching `process_videos`, so the reference and target audio
  latents come from an identical encoder (no precision/device divergence).
- The end-to-end smoke now generates paired references (reference ≠ target) via
  the paired-reference generator instead of reusing each clip as its own
  reference, so the integration gate exercises the real reference-encoding path
  rather than a degenerate shortcut.

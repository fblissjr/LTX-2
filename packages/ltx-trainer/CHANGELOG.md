# Changelog

All notable changes to the LTX trainer (this fork) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses semantic versioning.

## [Unreleased]

### Added

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

- The end-to-end smoke now generates paired references (reference ≠ target) via
  the paired-reference generator instead of reusing each clip as its own
  reference, so the integration gate exercises the real reference-encoding path
  rather than a degenerate shortcut.

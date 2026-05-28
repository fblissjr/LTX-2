# LTX-2 fork — Claude instructions

Last updated: 2026-05-28

**This is a fork of Lightricks's LTX-2 trainer**, branch `audio-guidance-iclora-vtv`. Used by the parent `ComfyUI-AudioLoopHelper` repo (`../../`) for audio→video IC-LoRA training research. Parent project's CLAUDE.md (`../../CLAUDE.md`) covers the broader ComfyUI work; this file covers what's specific to working IN this fork.

## Layout

- `packages/ltx-core/` — Lightricks's transformer + VAE + tokenizers (upstream-tracking).
- `packages/ltx-trainer/` — the trainer (where our work lives). Audio-IC-LoRA + block-swap port + data validation + e2e smoke all here.
- `packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py` — the `condition` / `generate` / `continuation` audio modes we added.
- `packages/ltx-trainer/src/ltx_trainer/block_swap.py` — the 4090-fit machinery. **READ THE MODULE DOCSTRING FIRST** — it documents the optimum.quanto Parameter-setter trap that wastes hours if you re-derive it.
- `packages/ltx-trainer/tests/` — unit tests for the strategy, codec load, block-swap.

## Critical findings (read before touching the trainer)

### Block-swap + optimum.quanto Parameter trap (2026-05-27)

**`Parameter.data = w.data.to('cpu')` for `WeightQBytesTensor` SILENTLY KEEPS storage on the original device.** The Parameter setter has device-preservation logic that undoes the move; both int8 and fp8 quanto tensors hit this. The fix is FULL PARAMETER REPLACEMENT:

```python
setattr(submodule, name, nn.Parameter(weight.to(device), requires_grad=...))
```

This is what `_move_module_data` in `block_swap.py` does. Trainable params (`requires_grad=True` — LoRA after PEFT wrap) are SKIPPED — replacing them orphans optimizer state. Burned hours on this; the test suite locks both contracts.

Three other paths researched + rejected: fp8-quanto (same trap), `register_parameter` (same trap), mmgp adoption (inference-only by design).

### 4090 feasibility (int8 22B + LoRA)

**E0.2 PASSED 2026-05-27**: int8-quanto 22B LTX-2 distilled + LoRA TRAINING fits a 24 GB 4090 with `block_swap_blocks=36/48` + gradient checkpointing ON. VRAM drops 21.87 → 8.89 GB at attach; peak training 17.59 GB; 3 steps + checkpoint produced in 0.5 min on synthetic 100-clip data. Speed ~11 sec/step (slow due to per-step PCIe transfers, but functional). See `../../internal/audio_iclora_status.md` E0.2 section.

### Trainer codec-load gating

Audio VAE decoder + vocoder are loaded only when **validation will actually generate audio** (gate on `validation.interval AND generate_audio`, NOT strategy's `requires_audio` — training reads pre-encoded audio latents, the decoder is validation-only). Frozen codecs offload to CPU symmetrically across video + audio. ~700 MB saved on a tight 24 GB card. See `_validation_will_run` + `_offload_frozen_codecs` in `trainer.py`.

### Block-swap ordering

`attach_block_swap` MUST run AFTER `accelerator.prepare()` — prepare recursively moves the model to the compute device, undoing any prior offload. The trainer wires it this way; if a future refactor moves the call, the swap silently no-ops.

## Commands

```bash
# Tests (run before committing trainer changes)
uv run --group dev python -m pytest packages/ltx-trainer/tests/ -v

# E2E smoke (generate synthetic data + precompute + validate + train 3 steps)
# Outputs land in ../../data/audio_iclora/<labeled-subfolder>/ per the
# parent project's data-location rule.
uv run --group dev python packages/ltx-trainer/scripts/run_e2e_smoke.py \
  --workdir ../../data/audio_iclora/<labeled-subfolder> \
  --n-clips 20 \
  --model-path <comfyui_models>/Lightricks_LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
  --text-encoder-path <comfyui_models>/gemma-3-12b-it-qat-q4_0-unquantized/weights \
  --load-text-encoder-in-8bit \
  --base-config <your-train-config.yaml>

# Direct training run
uv run --group dev python packages/ltx-trainer/scripts/train.py <config.yaml>
```

For the 4090 training config: set `acceleration.block_swap_blocks: 36`, `acceleration.quantization: int8-quanto`, `optimization.enable_gradient_checkpointing: true`.

## Git conventions for this fork

- Branch: `audio-guidance-iclora-vtv`. **Never push without explicit user approval.**
- Commit signing: this fork uses GPG signing by default. Use `git -c commit.gpgsign=false commit ...` from agentic runs (no signing agent in the harness env).
- **No `Co-Authored-By` lines** in commit messages.
- Commit titles + bodies at the abstract level; no internal-file leaks, no dated empirical numbers in titles (body can paraphrase).
- Always `git status --short` before commit — see `../../CLAUDE.md` "concurrent unstaged edits" warning.

## Sister-repo coordination

This fork tracks musubi-tuner's LTX-2 work as prior art (cloned at `../musubi-tuner/`, branch `ltx-2`). Their `av_ic` strategy + the offload helpers in `coderef/musubi-tuner/src/musubi_tuner/ltx_2/model/ltx2_custom_offloading_utils.py` are direct references. The catalog of which musubi techniques apply to our work lives at `../../internal/audio_iclora_prior_art.md` (private clone only).

### Cross-repo memo channel with audio-loop-lab (established 2026-05-28)

Bilateral async channel between this Claude (LTX-2 fork side) and audio-loop-lab claude (the parent ComfyUI-AudioLoopHelper project that consumes LTX-2 trainer outputs). Mirrors the sage-fork channel pattern.

- **Inbound** (audio-loop-lab → us): `internal/AUDIO_LOOP_CLAUDE_TO_LTX2_CLAUDE_MEMO.md`. SessionStart hook at `.claude/hooks/check_memo_inbox.sh` notifies us when it's newer than `internal/.memo_inbox_seen_at`.
- **Outbound** (us → audio-loop-lab): `coderef/audio-loop-lab/internal/LTX2_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md` (via the reverse symlink at `coderef/audio-loop-lab` → the parent ComfyUI-AudioLoopHelper repo).
- **Skill**: `.claude/skills/cross-repo-handoff/SKILL.md` — trigger phrases like "send memo to audio-loop-lab", "check audio-loop memo", "respond to audio-loop-lab".
- **Send helper**: `internal/scripts/send_memo_to_audio_loop_lab.sh` — bumps outbound mtime after a memo edit.

`.claude/`, `internal/`, and `coderef/` are all gitignored on this fork (they depend on the reverse symlink + local-machine paths other clones don't have).

## Pointers

- Parent project rules: `../../CLAUDE.md`
- Trainer plan + design history: `../../internal/audio_iclora_status.md`, `../../internal/audio_iclora_roadmap.md`, `../../internal/block_swap_port_plan.md` (all private clone)
- Public docs: `packages/ltx-trainer/docs/`

# LTX-2 fork — Claude instructions

Last updated: 2026-06-04

**This is a fork of Lightricks's LTX-2 trainer**, branch `audio-guidance-iclora-vtv`. Used by the parent `ComfyUI-AudioLoopHelper` repo (`../../`) for audio→video IC-LoRA training research. Parent project's CLAUDE.md (`../../CLAUDE.md`) covers the broader ComfyUI work; this file covers what's specific to working IN this fork.

## Layout

- `packages/ltx-core/` — Lightricks's transformer + VAE + tokenizers (upstream-tracking).
- `packages/ltx-trainer/` — the trainer (where our work lives). Audio-IC-LoRA + block-swap port + data validation + e2e smoke all here.
- `packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py` — the `condition` / `generate` / `continuation` audio modes we added.
- `packages/ltx-trainer/src/ltx_trainer/training_strategies/audio_reference.py` — the **audio-only IC-LoRA** strategy: an in-context reference AUDIO steers a generated-audio attribute; AV target; audio stream `[target | reference-clean]`. Its `reference_audio_latents` channel is built by `reference_audio.py` + `scripts/precompute_reference_audio.py`; example config `configs/ltx2_audio_reference.yaml`.
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

### Audio-reference IC-LoRA — the seesaw + train/inference parity (2026-05-30)

The `audio_reference` strategy is the **transfer** paradigm: the in-context reference AUDIO is the *controller*, so the pairing is **different content, shared attribute, reference carries it, caption neutral**. The seesaw (don't re-litigate toward matched content): *matched content ⟺ the caption controls the attribute*; *the audio controls the attribute ⟺ unmatched content* — mutually exclusive. Matched-content is the OTHER (fixed-effect / cowboy-hat) paradigm where text is the controller.

**Train/inference RoPE parity is load-bearing.** The reference is appended `[target | ref]` at NEGATIVE positions matching `ltx_pipelines.lipdub.patchify_lipdub_audio_reference_latent(negative_positions=True)` — shift by the reference's own end-bound + a `0.04` gap so it ends just below the target's 0. A different train-time offset silently degrades generation; locked two ways: literal-pin tests AND a cross-parity test that imports the real upstream `patchify_lipdub_audio_reference_latent` and asserts byte-equal positions (catches upstream drift the literal pin can't). The ComfyUI node side remains the one unguarded copy of the offset.

**Feasibility + fit proven (2026-05-29):** forward+backward on the real 22B (int8 + block-swap + grad-ckpt) gives a finite loss with `∂loss/∂reference ≠ 0` (the reference is load-bearing — the trainer twin of "remove reference → output stops tracking") at **8.70 GB peak**. `block_swap_blocks=36` is overkill here; startup (~17.8 GB, full int8 model pre-swap) is the tight point, so the config uses 24 and can go ~16.

**Eval has two lanes (2026-06-04).** Scripted/automated: `validation_sampler.py` audio lanes (`reference_audio_latents` = `[target | ref]` negative-RoPE swap-eval shape, lipdub cross-parity-locked; `input_audio_latents` = frozen-stream coupling shape) driven by `scripts/render_eval_sweep.py` (model loads once; fixed seeds across refs = the swap contract; reads precompute `.pt`s via `eval_utils`). Manual: ComfyUI nodes (the AUDIO twin of `LTXAddVideoICLoRAGuide` / `…Advanced`, reusing ltx-core `AudioConditionByReferenceLatent` + the lipdub patchify). **When the two lanes disagree on a checkpoint ranking, ComfyUI is the shipped regime and arbitrates.** Train configs for this strategy still keep validation disabled.

**Next-round training levers (implemented 2026-06-04; ready-to-uncomment block at the end of `configs/ltx2_audio_reference.yaml`):** `timestep_sampling_mode: high_sigma_mixture` (soft high-noise bias — the leak fix), `precompute_reference_audio.py --channel-aug-variants K` (anti session-fingerprint; variant 0 = clean), and ambiguity-aware ref-gap verdicts (a non-positive gap does NOT mean the reference is unused — `REF_GAP_AMBIGUITY_NOTE` in `metrics.py`). Still data-side / design-gated: cross-video pairing via `speaker_cluster_id`, target face mask, aux identity loss.

### Environment + data gotchas (each cost real time)

- `ltx_pipelines` imports fine in the workspace venv (co-installed sibling) — usable in tests via `importorskip` for cross-parity locks, but it is NOT a trainer runtime dep: never import it from `src/ltx_trainer/`.
- Load the video VAE from the COMBINED checkpoint only; the stripped standalone VAE file lacks embedded `config` metadata → the configurator defaults `timestep_conditioning=True` → phantom params crash decode on meta.
- `scripts/inference.py`'s on-the-fly prompt encode is latently broken (it never loads an `embeddings_processor`); use `cached_embeddings` (as `render_eval_sweep.py` does).
- Data-layout trap: files in `reference_audio_latents/` are named after the TARGET clip — the name is the dataset PAIRING key, not content provenance. Check the build script / `reference_plan_*.jsonl` (or correlate latents) before claiming where a reference came from.

**First learning run (2026-05-30):** trained `audio_reference` on the **distilled** 22B at commit `05f4e2e`, recipe `configs/ltx2_audio_reference.yaml` (the actual run used a gitignored real-path copy `_run_audio_ref.yaml`), on the 291-pair pitch-reference dataset (a voiced reference tone → the generated audio adopts that pitch). int8-quanto + block-swap 24, 2000 steps, validation off → checkpoint. The F0-tracking eval is audio claude's (ComfyUI, stock `LTXVAudioVAEEncode` + `LTXVSetAudioRefTokens`) — confirm the eval's audio VAE runs **fp32** to match the training encode.

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

**After any `uv sync` / lock change, verify torch before declaring done** (`.venv/bin/python -c "import torch; torch.cuda.is_available()"`): the venv carries out-of-band CUDA pieces a plain sync can prune (it once stripped libcusparseLt/cudnn files and broke torch import; fix = reinstall the `nvidia-*` set at the prior versions).

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
- **Outbound** (us → audio-loop-lab): `coderef/ComfyUI-AudioLoopHelper/internal/LTX2_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md` (via the reverse symlink at `coderef/ComfyUI-AudioLoopHelper` → the parent ComfyUI-AudioLoopHelper repo).
- **Skill**: `.claude/skills/cross-repo-handoff/SKILL.md` — trigger phrases like "send memo to audio-loop-lab", "check audio-loop memo", "respond to audio-loop-lab".
- **Send helper**: `internal/scripts/send_memo_to_audio_loop_claude.sh` — bumps outbound mtime, takes the eager sent-archive copy, and warns if the previous memo is unread.

`.claude/`, `internal/`, and `coderef/` are all gitignored on this fork (they depend on the reverse symlink + local-machine paths other clones don't have).

## Pointers

- Parent project rules: `../../CLAUDE.md`
- Internal wiki (read first): `internal/wiki/index.md` — LLM-consumable knowledge base for the trainer work (strategy mechanics, 4090 fit, eval design, config keys, module/test maps), mined 2026-06-04 from this fork's docs/internal/transcripts/memory + the parent repo's trainer docs. Gitignored, local machine only.
- Trainer plan + design history (living source): `coderef/ComfyUI-AudioLoopHelper/internal/audio_iclora_training/` — `status.md`, `roadmap.md`, `block_swap_port_plan.md`, `prior_art.md`, plus design/eval docs (private clone only; the wiki distills these — defer to them when they disagree).
- Public fork docs: `docs/audio_iclora_trainer_notes.md` (the audio-IC-LoRA experiment write-up; linked from this fork's README + the companion repo's `docs/audio_iclora/index.md`). Per-package docs: each package's own `README.md`.

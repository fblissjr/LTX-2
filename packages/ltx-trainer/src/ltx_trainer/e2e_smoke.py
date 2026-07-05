"""End-to-end integration smoke for the audio→video IC-LoRA pipeline.

Drives the REAL pipeline (not stubs) through every stage and gates between them
on FACTS, so the integration is exercised once cheaply and a failure points at a
specific stage instead of sending you on a goose-hunt during a real run:

  GEN        synthetic clips + handle-only captions          (CPU, automated)
  PRECOMPUTE real process_dataset.py --with-audio            (GPU, invoked)
  VALIDATE   real data_validation on the output    [HARD GATE: must be green]
  TRAIN      real train.py, steps capped to a few   [HARD GATE: a checkpoint]

Guardrails are deliberately FACT-based only (the hard rules we KNOW): required
source dirs exist + non-empty, the validator passes (shapes/counts/conditions are
real pipeline requirements), a checkpoint is produced. Things we merely believe
(caption-leak heuristics, clip-count floors) are NOT hard gates here — they're
the data-generation strategist's advisory call, not facts to block on.

The GPU stages shell out to the genuine scripts; this module owns the sequencing
and the gates. Gate + config-override logic is pure and unit-tested; the GPU
invocations are thin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Source dirs the audio v2v IC-LoRA path requires after precompute (mirrors
# VideoToVideoStrategy.get_data_sources with_audio=True).
REQUIRED_SOURCES = ["latents", "conditions", "reference_latents", "audio_latents"]


class SmokeGateError(RuntimeError):
    """A hard, fact-based gate failed. Message names the stage + the concrete fact."""


def gate_sources_present(precomputed_dir: Path, sources: list[str] = REQUIRED_SOURCES) -> None:
    """FACT: every required source dir must exist and hold at least one .pt.
    (A missing/empty audio_latents is the silent 'with_audio but no audio' gap.)"""
    base = Path(precomputed_dir)
    scan = base / ".precomputed" if (base / ".precomputed").exists() else base
    missing = []
    for s in sources:
        d = scan / s
        if not d.exists() or not any(d.glob("**/*.pt")):
            missing.append(s)
    if missing:
        raise SmokeGateError(
            f"PRECOMPUTE produced no .pt files for: {missing} (under {scan}). "
            "process_dataset.py did not emit these — re-run it (with --with-audio for audio_latents)."
        )


def gate_validation(report) -> None:
    """FACT: the validator must pass before training. Surfaces the specific errors
    (count mismatch / shape / NaN / missing conditions) rather than a vague fail."""
    if not report.ok:
        errs = "\n  - ".join(report.all_errors())
        raise SmokeGateError(f"VALIDATE failed — do not train on this data:\n  - {errs}")


def gate_checkpoint(output_dir: Path) -> Path:
    """FACT (the smoke's success definition): training produced a checkpoint."""
    ckpts = sorted(Path(output_dir).glob("**/*.safetensors")) + sorted(Path(output_dir).glob("**/*.pt"))
    if not ckpts:
        raise SmokeGateError(
            f"TRAIN produced no checkpoint under {output_dir}. The training step ran but emitted "
            "nothing — check the trainer log for the real error (this gate just confirms the smoke "
            "didn't silently no-op)."
        )
    return ckpts[-1]


def make_smoke_train_config(
    base_config: dict,
    preprocessed_root: str,
    output_dir: str,
    steps: int,
    model_path: str | None = None,
    text_encoder_path: str | None = None,
    reference_video: str | None = None,
) -> dict:
    """Override a REAL base config for the smoke: point it at the smoke dataset +
    the model/Gemma, cap steps, set the output dir, and FORCE the audio-guided v2v
    strategy. Pure — we override a known-good config rather than synthesize the
    schema from scratch (the base may be a text_to_video config)."""
    cfg = dict(base_config)
    if model_path or text_encoder_path:
        m = {**cfg.get("model", {})}
        if model_path:
            m["model_path"] = model_path
        if text_encoder_path:
            m["text_encoder_path"] = text_encoder_path
        cfg["model"] = m
    cfg["data"] = {**cfg.get("data", {}), "preprocessed_data_root": preprocessed_root}
    cfg["optimization"] = {**cfg.get("optimization", {}), "steps": int(steps)}  # steps lives under optimization
    cfg["output_dir"] = output_dir
    ts = {**cfg.get("training_strategy", {})}
    ts["name"] = "video_to_video"  # FORCE — base may be text_to_video; we want audio-guided v2v
    ts["with_audio"] = True
    ts.setdefault("audio_mode", "condition")
    cfg["training_strategy"] = ts
    # v2v REQUIRES validation.reference_videos (a cross-field check) even when validation
    # is disabled. Provide a reference + set interval=None so the short smoke just trains
    # and never runs a full validation inference (which would need the upscaler etc).
    val = {**cfg.get("validation", {}), "interval": None}
    if reference_video:
        val["reference_videos"] = [reference_video]
        val["prompts"] = [(val.get("prompts") or ["a shape"])[0]]  # 1 prompt to match 1 reference
    cfg["validation"] = val
    # write a checkpoint within the short smoke (base interval may be large or null)
    cfg["checkpoints"] = {**cfg.get("checkpoints", {}), "interval": int(steps)}
    return cfg


def _run(cmd: list[str], stage: str) -> None:
    print(f"\n=== {stage}: {' '.join(cmd)} ===", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SmokeGateError(f"{stage} command failed (exit {proc.returncode}). See its output above.")


def synthetic_dataset_bucket(width: int, height: int, fps: int, duration_s: float) -> str:
    """process_dataset --resolution-buckets is WxH×FRAMES (a frame count, e.g.
    57/89/121 — all 8k+1), NOT WxH×fps. Derive the frame count the synthetic
    generator actually produces (snapped to 8k+1) so the bucket matches the clips;
    passing fps as the third field would mis-bucket them."""
    from ltx_trainer.synthetic_av import snap_frames_to_8k1

    return f"{width}x{height}x{snap_frames_to_8k1(round(duration_s * fps))}"


def generate_smoke_dataset(
    workdir: str | Path,
    n_clips: int,
    *,
    fps: int,
    width: int,
    height: int,
    duration_s: float,
) -> Path:
    """Generate the synthetic smoke dataset with PAIRED references (ref != target).

    Uses ``synthetic_av.generate_dataset_paired_refs`` so the integration smoke
    exercises the SAME reference-encoding path as real audio→video training (a
    separate reference clip per row, same identity / different BPM) rather than
    the old ref==target shortcut. ref==target both let a broken pipeline pass the
    smoke AND was the data bug that suppressed audio coupling in the first real
    run — keeping the smoke on paired refs closes both. See data plan §1.2.
    """
    from ltx_trainer.synthetic_av import generate_dataset_paired_refs

    return generate_dataset_paired_refs(
        workdir, n_clips, fps=fps, width=width, height=height, duration_s=duration_s
    )


def run_smoke(
    *,
    workdir: Path,
    n_clips: int = 4,
    captions_path: Path | None = None,
    resolution_bucket: str = "256x256x25",  # synthetic CLIP spec: W x H x FPS
    duration_s: float = 3.0,
    dataset_bucket: str | None = None,  # process_dataset WxH×FRAMES; required for real --captions
    model_path: str | None = None,  # single-file LTX-2 checkpoint (VAEs + projectors)
    text_encoder_path: str | None = None,  # Gemma dir
    load_text_encoder_in_8bit: bool = False,  # Gemma-12B bf16 ≈ 24GB; 8bit fits a 4090
    base_config: Path | None = None,
    steps: int = 3,
    python: str | None = None,
) -> None:
    """Run the full smoke. If captions_path is given, use that real dataset;
    otherwise generate synthetic clips. If base_config is given, run the (capped)
    train stage; otherwise stop after VALIDATE and print the train command."""
    import json

    import yaml

    from ltx_trainer.data_validation import validate_dataset

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    py = python or sys.executable
    scripts = Path(__file__).resolve().parents[2] / "scripts"

    w, h, fps = (int(x) for x in resolution_bucket.split("x"))  # clip spec: W x H x FPS

    # GEN
    if captions_path is None:
        print(
            f"=== GEN: {n_clips} synthetic beat→pulse clips ({w}x{h}@{fps}fps, {duration_s}s, "
            "paired refs: ref != target) ===",
            flush=True,
        )
        captions_path = generate_smoke_dataset(workdir, n_clips, fps=fps, width=w, height=h, duration_s=duration_s)
        # process_dataset buckets by FRAME COUNT, not fps — derive it from the clips.
        bucket = dataset_bucket or synthetic_dataset_bucket(w, h, fps, duration_s)
    else:
        if dataset_bucket is None:
            raise SmokeGateError(
                "real --captions requires --dataset-bucket WxHxFRAMES (process_dataset buckets by "
                "frame count, not fps — e.g. 256x256x73 for ~3s @ 25fps)."
            )
        bucket = dataset_bucket

    if not model_path or not text_encoder_path:
        raise SmokeGateError(
            "PRECOMPUTE needs --model-path (single-file LTX-2 checkpoint with VAEs+projectors) "
            "and --text-encoder-path (Gemma dir). On a ComfyUI split-file setup, point --model-path "
            "at the full Lightricks ltx-2.3-22b-distilled-1.1.safetensors."
        )

    precomputed = workdir / "precomputed"

    # PRECOMPUTE (real)
    cmd = [py, str(scripts / "process_dataset.py"), str(captions_path),
           "--output-dir", str(precomputed), "--with-audio",
           "--model-path", str(model_path), "--text-encoder-path", str(text_encoder_path),
           "--resolution-buckets", bucket,
           "--reference-column", "reference", "--caption-column", "caption", "--video-column", "video"]
    if load_text_encoder_in_8bit:
        cmd.append("--load-text-encoder-in-8bit")
    _run(cmd, "PRECOMPUTE")
    gate_sources_present(precomputed)

    # VALIDATE (real, hard gate)
    print("\n=== VALIDATE ===", flush=True)
    report = validate_dataset(precomputed, with_audio=True)
    from ltx_trainer.data_validation import format_report

    print(format_report(report))
    gate_validation(report)
    print("VALIDATE: green — data is shaped + paired + aligned correctly.")

    # TRAIN (real, optional, hard gate)
    if base_config is None:
        print("\nData validated and ready to train. Provide --base-config <your.yaml> to run the "
              "(step-capped) train smoke, or train yourself pointing preprocessed_data_root at:")
        print(f"  {precomputed}")
        return
    out_dir = workdir / "smoke_out"
    # v2v needs a validation reference video — use the first dataset clip (validation
    # runs are disabled in the smoke config, this just satisfies the cross-field check).
    cap_rows = json.loads(Path(captions_path).read_text())
    ref_video = str((Path(captions_path).parent / cap_rows[0]["video"]).resolve()) if cap_rows else None
    smoke_cfg = make_smoke_train_config(
        yaml.safe_load(Path(base_config).read_text()), str(precomputed), str(out_dir), steps,
        model_path=model_path, text_encoder_path=text_encoder_path, reference_video=ref_video)
    cfg_path = workdir / "smoke_config.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_cfg))
    _run([py, str(scripts / "train.py"), str(cfg_path)], "TRAIN")
    ckpt = gate_checkpoint(out_dir)
    print(f"\nSMOKE PASSED — full pipeline ran end to end; checkpoint at {ckpt}")

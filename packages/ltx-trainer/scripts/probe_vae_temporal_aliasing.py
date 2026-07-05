#!/usr/bin/env python3
"""Probe the video VAE's temporal-aliasing ceiling on the beat→pulse coupling.

WHY: the video VAE compresses 8x temporally, so the per-beat brightness pulse in
our synthetic clips is sampled on the latent time axis at only ~fps/8 frames/s.
The Nyquist ceiling there is ~fps * 3.75 BPM (≈94 BPM at 25 fps), and our
[60,160] BPM training range puts most pulses ABOVE it. The open question (see the
audio-loop-lab Nyquist memo) is whether the LEARNED VAE still carries a
supra-Nyquist brightness pulse in its channel code, or aliases it away.

WHAT THIS MEASURES (and its limit): this decodes the EXISTING precomputed target
latents — i.e. exactly what the model is trained to reproduce — back to pixels
and measures the recovered pulse rate vs the manifest's target BPM. It is an
UPPER BOUND on what the model can produce: if the training-target latent already
decodes to an aliased pulse, even a perfect model emits an aliased pulse. A clean
recovery here does NOT prove the DiT can GENERATE that latent (flow-matching is
smoothing-biased and may not land the fragile high-frequency code) — so a PASS
informs the production BPM range, it does not by itself justify widening the
kill-early gate. The gate should stay sub-Nyquist regardless.

This is a read-only probe: no training, no checkpoints written. It needs the
video VAE (a real LTX-2 checkpoint) to run.

    uv run --group dev python packages/ltx-trainer/scripts/probe_vae_temporal_aliasing.py \
        <precomputed_dir> --model-path /path/to/ltx-2.3-...safetensors --limit 24
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from einops import rearrange
from rich.console import Console
from rich.table import Table

from ltx_trainer.eval_audio_coupling import tracking_slope
from ltx_trainer.model_loader import load_video_vae_decoder
from ltx_trainer.synthetic_av import measure_pulse_rate

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

# Latent Nyquist ceiling in BPM: (fps / temporal_factor) / 2 cycles/s * 60.
# temporal_factor = 8 ⇒ fps/16 * 60 = fps * 3.75.
_TEMPORAL_FACTOR = 8


def _nyquist_bpm(fps: float) -> float:
    return fps / _TEMPORAL_FACTOR / 2.0 * 60.0


def _predicted_alias_bpm(true_bpm: float, latent_fps: float) -> float:
    """Where a true_bpm pulse folds to when sampled at latent_fps (Hz→BPM)."""
    f = true_bpm / 60.0
    folded = abs(f - latent_fps * round(f / latent_fps)) if latent_fps > 0 else f
    return folded * 60.0


def _load_target_bpms(precomputed_dir: Path) -> dict[str, float]:
    """Map clip stem (e.g. 'clip_0007') → target_bpm from the dataset manifest."""
    # manifest sits at the dataset root; precomputed/ is a child.
    for candidate in (precomputed_dir.parent / "manifest.jsonl", precomputed_dir / "manifest.jsonl"):
        if candidate.exists():
            out: dict[str, float] = {}
            for line in candidate.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                out[Path(row["video"]).stem] = float(row["target_bpm"])
            return out
    raise typer.BadParameter(f"manifest.jsonl not found near {precomputed_dir}")


@torch.inference_mode()
def _decode_to_frames(vae, latent_file: Path, device: torch.device) -> tuple[np.ndarray, float]:
    """Decode one target latent .pt to a [F, C, H, W] float array in [0,1] + its fps."""
    data = torch.load(latent_file, map_location=device, weights_only=True)
    latents = data["latents"]
    if latents.dim() == 2:  # legacy patchified [seq_len, C]
        latents = rearrange(
            latents, "(f h w) c -> c f h w", f=data["num_frames"], h=data["height"], w=data["width"]
        )
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    video = vae(latents)  # [1, C, F, H, W]
    video = rearrange(video, "1 c f h w -> f c h w")
    video = ((video + 1) / 2).clamp(0, 1)
    return video.float().cpu().numpy(), float(data.get("fps", 25))


@app.command()
def main(
    precomputed_dir: str = typer.Argument(..., help="Dataset precomputed/ dir (has latents/ + ../manifest.jsonl)"),
    model_path: str = typer.Option(..., help="LTX-2 checkpoint (.safetensors) for the video VAE decoder"),
    device: str = typer.Option("cuda", help="Compute device"),
    limit: int = typer.Option(24, help="Max clips to probe (sampled evenly across the BPM range)"),
) -> None:
    """Decode existing target latents and compare recovered pulse rate to target BPM."""
    root = Path(precomputed_dir)
    latents_dir = root / "latents"
    if not latents_dir.exists():
        raise typer.BadParameter(f"no latents/ under {root}")

    bpm_by_stem = _load_target_bpms(root)
    files = sorted(latents_dir.rglob("*.pt"))
    paired = [(f, bpm_by_stem[f.stem]) for f in files if f.stem in bpm_by_stem]
    if not paired:
        raise typer.BadParameter("no latent files matched manifest target_bpm entries")

    # Sample evenly across BPM so we cover both sides of the Nyquist boundary.
    paired.sort(key=lambda p: p[1])
    if limit and len(paired) > limit:
        # even spread across the BPM range; np.unique drops collisions from rounding
        idx = np.unique(np.linspace(0, len(paired) - 1, limit).round().astype(int))
        paired = [paired[i] for i in idx]

    with console.status(f"[bold]Loading video VAE decoder from {model_path}..."):
        vae = load_video_vae_decoder(model_path, device=torch.device(device), dtype=torch.bfloat16)

    # Store only non-derivable values; err/supra are computed at print time.
    rows: list[tuple[float, float, float]] = []  # (expected_bpm, measured_bpm, alias_if_folded)
    for latent_file, exp_bpm in paired:
        frames, fps = _decode_to_frames(vae, latent_file, torch.device(device))
        meas_bpm = measure_pulse_rate(frames, fps)
        rows.append((exp_bpm, meas_bpm, _predicted_alias_bpm(exp_bpm, fps / _TEMPORAL_FACTOR)))

    nyq = _nyquist_bpm(25.0)
    table = Table(title=f"VAE temporal-aliasing probe (latent Nyquist ≈ {nyq:.0f} BPM @ 25fps)")
    for col in ("expected BPM", "measured BPM", "abs err", "alias-if-folded", "supra-Nyquist?"):
        table.add_column(col, justify="right")
    for exp_bpm, meas_bpm, alias in rows:
        err = abs(meas_bpm - exp_bpm)
        marker = "[red]yes[/red]" if exp_bpm > nyq else "[green]no[/green]"
        # Flag rows where the measurement matches the FOLD better than the truth.
        aliased = abs(meas_bpm - alias) + 1.0 < err
        meas_str = f"[red]{meas_bpm:.1f}[/red]" if aliased else f"{meas_bpm:.1f}"
        table.add_row(f"{exp_bpm:.1f}", meas_str, f"{err:.1f}", f"{alias:.1f}", marker)
    console.print(table)

    exp_arr = np.array([r[0] for r in rows])
    meas_arr = np.array([r[1] for r in rows])
    slope, r2 = tracking_slope(exp_arr.tolist(), meas_arr.tolist())
    console.print(f"\noverall tracking slope={slope:.3f} R²={r2:.3f} (slope≈1 = VAE carries the pulse)")
    for label, mask in (("sub-Nyquist ", exp_arr <= nyq), ("supra-Nyquist", exp_arr > nyq)):
        if mask.any():
            s, r = tracking_slope(exp_arr[mask].tolist(), meas_arr[mask].tolist())
            console.print(f"  {label} (n={int(mask.sum())}): slope={s:.3f} R²={r:.3f}")
    console.print(
        "\nREAD: if supra-Nyquist slope collapses toward 0 (or measured tracks the "
        "alias-if-folded column), the VAE does NOT carry the pulse there → cap the gate "
        "≤85 BPM. If supra-Nyquist slope stays ≈1, the learned code carries it → that "
        "informs the PRODUCTION range (the DiT-generation question is still separate)."
    )


if __name__ == "__main__":
    app()

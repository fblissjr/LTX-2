"""Scalable objective eval for audio→video coupling — the output-side counterpart
to data_validation (which checks inputs). This scores whether a trained LoRA made
the video track the audio MORE than the no-LoRA baseline, across many cases, with
a number instead of an eyeball (data plan §4 audio-swap test, made measurable).

The eval is the executable form of the audio-swap test:
  - sweep an input audio quantity (e.g. BPM) across cases, everything else fixed
    (prompt, seed, keyframes — see the manifest);
  - measure the SAME quantity back out of each output video (the coupling metric);
  - a working LoRA's measured-out tracks expected-in (slope≈1); the baseline is
    flat/noisy (slope≈0). The DELTA in tracking slope is the LoRA's contribution
    (index #8 — it must earn its keep).
  - base preservation: neutral-input cases (no coupling cue) should produce ~no
    coupling in the output (the knob is off → base behaves normally).

CPU-side: it consumes the inference outputs you render on GPU (mp4s + a manifest),
the same way data_validation consumes precomputed latents. New couplings plug in
as a metric function in COUPLING_METRICS — that's the scaling seam: one measure-
the-constructed-quantity function per coupling, the tracking/delta/verdict logic is
shared.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ltx_trainer.synthetic_av import measure_pulse_rate

# coupling name -> (frames[F,H,W,3] uint8, fps) -> measured scalar (same unit as the
# swept input). Add a coupling by adding a measurement function here.
COUPLING_METRICS = {
    "beat_pulse": measure_pulse_rate,  # measured BPM out of the video's brightness
}

# A working knob should track its input; below this the use case isn't earning its
# keep (index #8). Tunable per coupling later; one fact-light default for now.
MIN_LORA_SLOPE = 0.5          # LoRA output must follow input at least half-for-half
MIN_DELTA_OVER_BASELINE = 0.3  # and clearly beat the baseline's (loose) coupling


def decode_video_frames(path: str | Path, width: int, height: int) -> np.ndarray:
    """Decode an mp4 to [F,H,W,3] uint8 via ffmpeg (the output side of write_clip)."""
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {proc.stderr.decode()[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def tracking_slope(expected: list[float], measured: list[float]) -> tuple[float, float]:
    """Least-squares slope + R^2 of measured-vs-expected. slope≈1 = tracks the input,
    slope≈0 = ignores it. Pure (no I/O) so it's the unit-tested core."""
    x = np.asarray(expected, dtype=np.float64)
    y = np.asarray(measured, dtype=np.float64)
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return float(slope), float(1.0 - ss_res / ss_tot)


@dataclass
class CouplingEvalReport:
    coupling: str
    n_cases: int
    lora_slope: float = 0.0
    lora_r2: float = 0.0
    baseline_slope: float = 0.0
    baseline_r2: float = 0.0
    base_preservation_ok: bool | None = None  # None = not tested
    notes: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.lora_slope - self.baseline_slope

    @property
    def passed(self) -> bool:
        return self.lora_slope >= MIN_LORA_SLOPE and self.delta >= MIN_DELTA_OVER_BASELINE \
            and (self.base_preservation_ok is not False)


def evaluate_coupling(
    cases: list[dict],
    *,
    coupling: str,
    width: int,
    height: int,
    fps: float,
    neutral_cases: list[dict] | None = None,
    neutral_max_measured: float = 10.0,
) -> CouplingEvalReport:
    """cases: [{"expected": <input value>, "lora_video": path, "baseline_video": path}].
    neutral_cases: [{"lora_video": path}] where the input carries NO coupling cue;
    the LoRA output's measured coupling should stay below neutral_max_measured (the
    knob is off → base preserved). Returns a CouplingEvalReport."""
    if coupling not in COUPLING_METRICS:
        raise ValueError(f"unknown coupling {coupling!r}; have {sorted(COUPLING_METRICS)}")
    metric = COUPLING_METRICS[coupling]
    report = CouplingEvalReport(coupling=coupling, n_cases=len(cases))

    exp, lora_m, base_m = [], [], []
    for c in cases:
        exp.append(float(c["expected"]))
        lora_m.append(metric(decode_video_frames(c["lora_video"], width, height), fps))
        base_m.append(metric(decode_video_frames(c["baseline_video"], width, height), fps))
    report.lora_slope, report.lora_r2 = tracking_slope(exp, lora_m)
    report.baseline_slope, report.baseline_r2 = tracking_slope(exp, base_m)

    if neutral_cases:
        measured = [metric(decode_video_frames(c["lora_video"], width, height), fps) for c in neutral_cases]
        report.base_preservation_ok = all(m <= neutral_max_measured for m in measured)
        if not report.base_preservation_ok:
            report.notes.append(f"neutral inputs produced coupling {max(measured):.1f} > {neutral_max_measured} "
                                 "(LoRA fires when the knob should be off → base not preserved)")
    return report


def evaluate_from_manifest(manifest_path: str | Path, *, coupling: str, width: int, height: int, fps: float) -> CouplingEvalReport:
    """Manifest = json with {"cases": [...], "neutral_cases": [...]} (paths relative
    to the manifest dir). Scales: add cases/couplings to the manifest, not the code."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    base = manifest_path.parent

    def _abs(c: dict) -> dict:
        return {**c, **{k: str(base / c[k]) for k in ("lora_video", "baseline_video") if k in c}}

    return evaluate_coupling(
        [_abs(c) for c in data["cases"]], coupling=coupling, width=width, height=height, fps=fps,
        neutral_cases=[_abs(c) for c in data.get("neutral_cases", [])] or None,
    )


def format_report(r: CouplingEvalReport) -> str:
    lines = [f"=== audio→video coupling eval: {r.coupling} ({r.n_cases} cases) ==="]
    lines.append(f"LoRA     tracking slope={r.lora_slope:+.2f}  R^2={r.lora_r2:.2f}  (≈1 = follows the audio)")
    lines.append(f"baseline tracking slope={r.baseline_slope:+.2f}  R^2={r.baseline_r2:.2f}  (≈0 = ignores it)")
    lines.append(f"DELTA (LoRA - baseline) = {r.delta:+.2f}  (the LoRA's contribution; needs ≥ {MIN_DELTA_OVER_BASELINE})")
    if r.base_preservation_ok is not None:
        lines.append(f"base preservation: {'OK' if r.base_preservation_ok else 'FAILED'}")
    lines.extend(f"  ! {n}" for n in r.notes)
    lines.append("VERDICT: " + ("EARNS ITS KEEP — audio drives video, beyond baseline"
                                 if r.passed else "NOT DEMONSTRATED — delta too small / doesn't track / base not preserved"))
    return "\n".join(lines)

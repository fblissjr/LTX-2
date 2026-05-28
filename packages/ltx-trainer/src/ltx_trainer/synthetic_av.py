"""Synthetic audio↔video clips with a KNOWN, measurable coupling.

Purpose (data plan §8): prove the training MECHANISM before collecting real
footage, on data where we CONSTRUCT the coupling and can predict + measure the
exact result. Procedural, CPU-only (numpy + ffmpeg), no model.

Current coupling: **beat→pulse** — a shape whose size+brightness pulses on the
audio's beats. The audio is a click track at a chosen BPM; the video pulses on
exactly those beats, so `measure_pulse_rate(frames)` recovers the BPM. That
closing-the-loop property is what makes it a falsifiable test substrate: a trained
LoRA's output can be measured the same way and compared to the input beat rate.

HARD model rules are enforced here as FACTS (not heuristics): frame count must be
`8k+1` and resolution divisible by 32 — the model/pipeline require these
(DummyDataset raises on violation), so the generator snaps/validates to them and
can never emit pipeline-invalid clips.
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_SAMPLE_RATE = 16_000
FRAME_TEMPORAL_BASE = 8  # frame rule: (n - 1) % 8 == 0  → 8k+1
RES_MULTIPLE = 32  # width/height must be divisible by 32

# Handle-only caption for the beat-pulse coupling: names the concept the audio
# binds to (the handle) but NOT the rate/timing the audio determines (the
# execution) — see data plan §1. Deliberately rate-free.
BEAT_PULSE_CAPTION = "a glowing shape pulsing on a dark background"


def snap_frames_to_8k1(n: int) -> int:
    """Nearest valid frame count `8k+1` at or below n, floored at 9."""
    n = max(9, int(n))
    return ((n - 1) // FRAME_TEMPORAL_BASE) * FRAME_TEMPORAL_BASE + 1


def assert_resolution(width: int, height: int) -> None:
    """Hard rule: width/height divisible by 32 (model requirement)."""
    if width % RES_MULTIPLE or height % RES_MULTIPLE:
        raise ValueError(f"resolution {width}x{height} must be divisible by {RES_MULTIPLE}")


def _beat_times(bpm: float, duration_s: float) -> np.ndarray:
    interval = 60.0 / bpm
    return np.arange(0.0, duration_s, interval)


def _pulse_envelope(frame_times: np.ndarray, beat_times: np.ndarray, tau: float = 0.12) -> np.ndarray:
    """Per-frame pulse strength in [0,1]: a sharp onset at each beat that decays
    with time constant tau. Periodic in the beats → its fundamental is the beat
    frequency, which is what measure_pulse_rate recovers."""
    env = np.zeros_like(frame_times, dtype=np.float64)
    for b in beat_times:
        after = frame_times >= b
        env[after] = np.maximum(env[after], np.exp(-(frame_times[after] - b) / tau))
    return env


@dataclass
class ClipSpec:
    bpm: float
    duration_s: float
    fps: int
    width: int
    height: int
    shape: str = "circle"          # nuisance variable
    color: tuple = (255, 255, 255)  # nuisance variable
    center: tuple | None = None     # nuisance variable (fractional x,y); None=center


def generate_beat_pulse_clip(spec: ClipSpec) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Render one beat→pulse clip. Returns (frames[F,H,W,3] uint8, audio[N] float32,
    sample_rate, beat_times). Frame count snapped to 8k+1; resolution validated."""
    assert_resolution(spec.width, spec.height)
    n_frames = snap_frames_to_8k1(round(spec.duration_s * spec.fps))
    duration_s = n_frames / spec.fps  # actual duration after the snap
    frame_times = np.arange(n_frames) / spec.fps
    beats = _beat_times(spec.bpm, duration_s)
    env = _pulse_envelope(frame_times, beats)  # [F]

    # Video: a shape whose radius+brightness scale with the pulse envelope.
    h, w = spec.height, spec.width
    cx = (spec.center[0] if spec.center else 0.5) * w
    cy = (spec.center[1] if spec.center else 0.5) * h
    yy, xx = np.mgrid[0:h, 0:w]
    base_r = 0.12 * min(h, w)
    amp_r = 0.18 * min(h, w)
    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    color = np.array(spec.color, dtype=np.float64)
    for i in range(n_frames):
        r = base_r + amp_r * env[i]
        brightness = 0.35 + 0.65 * env[i]
        if spec.shape == "square":
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        else:  # circle (default)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        frames[i][mask] = np.clip(color * brightness, 0, 255).astype(np.uint8)

    # Audio: a short decaying sine "click" at each beat.
    n_samples = int(duration_s * AUDIO_SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float32)
    click_len = int(0.05 * AUDIO_SAMPLE_RATE)
    t_click = np.arange(click_len) / AUDIO_SAMPLE_RATE
    click = (np.sin(2 * np.pi * 880 * t_click) * np.exp(-t_click / 0.02)).astype(np.float32)
    for b in beats:
        s = int(b * AUDIO_SAMPLE_RATE)
        end = min(s + click_len, n_samples)
        audio[s:end] += click[: end - s]
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (0.9 * audio / peak).astype(np.float32)
    return frames, audio, AUDIO_SAMPLE_RATE, beats


def measure_pulse_rate(frames: np.ndarray, fps: float) -> float:
    """Recover the dominant pulse rate (BPM) from a video's per-frame brightness
    via a zero-padded FFT. Used to (a) self-test the generator and (b) score a
    trained LoRA's output the same way (the objective eval, data plan §4/§8)."""
    sig = frames.reshape(frames.shape[0], -1).mean(axis=1).astype(np.float64)
    sig = sig - sig.mean()
    if not np.any(sig):
        return 0.0
    n_pad = max(1024, 8 * len(sig))  # zero-pad for finer frequency resolution
    mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=1.0 / fps)
    mag[0] = 0.0  # drop DC
    return float(freqs[int(np.argmax(mag))] * 60.0)


def write_clip(frames: np.ndarray, audio: np.ndarray, sample_rate: int, fps: int, out_path: Path) -> None:
    """Mux frames + audio into an mp4 via ffmpeg (rawvideo pipe + a temp wav)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wf:
        # LTX-2's audio VAE expects STEREO (2-channel) input — write the mono click
        # track duplicated to L+R so the precompute mel is [B, 2, T, mel].
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        stereo = np.stack([audio, audio], axis=-1)  # [N, 2] interleaved
        wf.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path),
    ]
    proc = subprocess.run(cmd, input=frames.astype(np.uint8).tobytes(), capture_output=True)
    wav_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-500:]}")


def generate_dataset(
    out_dir: str | Path,
    n: int,
    *,
    bpm_range: tuple[float, float] = (60.0, 160.0),
    duration_s: float = 3.0,
    fps: int = 25,
    width: int = 256,
    height: int = 256,
    seed: int = 0,
) -> Path:
    """Write n beat→pulse clips + a captions.json (handle-only) + manifest.jsonl
    (ground-truth bpm/beats per clip for the objective eval). Nuisance variables
    (shape/color/position) are randomized so a LoRA can't shortcut on them; the
    pulse TIMING is the only thing correlated with the audio. Returns the
    captions.json path (the input to process_dataset.py)."""
    out_dir = Path(out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shapes = ["circle", "square"]
    captions, manifest = [], []
    for i in range(n):
        spec = ClipSpec(
            bpm=float(rng.uniform(*bpm_range)),
            duration_s=duration_s, fps=fps, width=width, height=height,
            shape=shapes[int(rng.integers(len(shapes)))],
            color=tuple(int(c) for c in rng.integers(120, 256, size=3)),
            center=(float(rng.uniform(0.35, 0.65)), float(rng.uniform(0.35, 0.65))),
        )
        frames, audio, sr, beats = generate_beat_pulse_clip(spec)
        rel = f"clips/clip_{i:04d}.mp4"
        write_clip(frames, audio, sr, fps, out_dir / rel)
        captions.append({"video": rel, "caption": BEAT_PULSE_CAPTION})
        manifest.append({"video": rel, "bpm": spec.bpm, "n_beats": int(len(beats)),
                         "shape": spec.shape, "duration_s": frames.shape[0] / fps})
    captions_path = out_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))
    (out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    return captions_path

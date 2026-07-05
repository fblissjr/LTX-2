"""Tests for the synthetic AV generator (ltx_trainer.synthetic_av).

The generator's whole value is that it produces a KNOWN, measurable coupling, so
the key test closes the loop: a rendered beat→pulse clip's measured pulse rate
must recover the requested BPM. Also locks the hard model rules (8k+1 frames,
div-32 resolution) the generator self-enforces, and the dataset writer's output
shape (paired clips + handle-only captions + ground-truth manifest).

Pure-numpy tests always run; the ffmpeg mux/dataset tests skip if ffmpeg is absent.
"""

from __future__ import annotations

import json
import shutil

import pytest

from ltx_trainer.synthetic_av import (
    BEAT_PULSE_CAPTION,
    ClipSpec,
    assert_resolution,
    generate_beat_pulse_clip,
    generate_dataset,
    measure_pulse_rate,
    snap_frames_to_8k1,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")


@pytest.mark.parametrize("bpm", [80, 120, 150])
def test_measured_pulse_rate_recovers_bpm(bpm):
    """The loop-closure: the rendered video's brightness oscillates at the beat
    rate, so measure_pulse_rate recovers the BPM we asked for. This is what makes
    the data a falsifiable test substrate."""
    spec = ClipSpec(bpm=bpm, duration_s=6.0, fps=25, width=128, height=128)
    frames, _audio, _sr, beats = generate_beat_pulse_clip(spec)
    measured = measure_pulse_rate(frames, spec.fps)
    assert abs(measured - bpm) < 8.0, f"measured {measured:.1f} vs requested {bpm}"
    assert len(beats) >= 2


def test_frame_count_is_8k1():
    """Hard rule: frame count is 8k+1 (model requirement), snapped automatically."""
    spec = ClipSpec(bpm=120, duration_s=3.7, fps=25, width=64, height=64)
    frames, *_ = generate_beat_pulse_clip(spec)
    assert (frames.shape[0] - 1) % 8 == 0


def test_snap_frames_to_8k1():
    assert snap_frames_to_8k1(75) == 73
    assert snap_frames_to_8k1(100) == 97
    assert snap_frames_to_8k1(1) == 9  # floored at the minimum valid count


def test_resolution_must_be_div32():
    assert_resolution(256, 128)  # ok, no raise
    with pytest.raises(ValueError, match="divisible by 32"):
        assert_resolution(250, 128)
    with pytest.raises(ValueError, match="divisible by 32"):
        ClipSpec(bpm=120, duration_s=2.0, fps=25, width=100, height=64)  # noqa: F841 — raises in generate
        generate_beat_pulse_clip(ClipSpec(bpm=120, duration_s=2.0, fps=25, width=100, height=64))


def test_audio_length_matches_video_duration():
    spec = ClipSpec(bpm=120, duration_s=4.0, fps=25, width=64, height=64)
    frames, audio, sr, _ = generate_beat_pulse_clip(spec)
    video_dur = frames.shape[0] / spec.fps
    audio_dur = len(audio) / sr
    assert abs(video_dur - audio_dur) < 0.1  # audio spans the (snapped) clip


def test_caption_is_handle_only_no_leak():
    """The shipped caption names the handle ('pulsing') but leaks NO rate/timing —
    the audio must carry that (data plan §1)."""
    c = BEAT_PULSE_CAPTION.lower()
    assert "puls" in c  # the handle is present
    for leak in ("bpm", "beat", "tempo", "120", "fast", "slow", "per second"):
        assert leak not in c, f"caption leaks execution detail: {leak!r}"


@_needs_ffmpeg
def test_generate_dataset_writes_paired_outputs(tmp_path):
    captions_path = generate_dataset(tmp_path, n=3, duration_s=2.0, fps=25, width=64, height=64, seed=1)
    assert captions_path.exists()
    captions = json.loads(captions_path.read_text())
    assert len(captions) == 3
    # every caption row points at a real clip + carries the handle-only caption
    for row in captions:
        assert (tmp_path / row["video"]).exists()
        assert row["caption"] == BEAT_PULSE_CAPTION
    # manifest carries ground-truth bpm per clip (for the objective eval)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 3
    assert all("bpm" in m and m["bpm"] > 0 for m in manifest)


@_needs_ffmpeg
def test_written_clip_round_trips_pulse_rate(tmp_path):
    """Belt-and-suspenders: after the ffmpeg encode round-trip, the clip still
    decodes to frames whose pulse rate recovers the BPM (no encode corruption)."""
    import numpy as np

    from ltx_trainer.synthetic_av import AUDIO_SAMPLE_RATE, write_clip  # noqa: F401

    spec = ClipSpec(bpm=120, duration_s=6.0, fps=25, width=64, height=64)
    frames, audio, sr, _ = generate_beat_pulse_clip(spec)
    out = tmp_path / "clip.mp4"
    write_clip(frames, audio, sr, spec.fps, out)
    assert out.exists() and out.stat().st_size > 0
    # decode back via ffmpeg → rawvideo → frames, re-measure
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
    )
    decoded = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 64, 64, 3)
    assert abs(measure_pulse_rate(decoded, spec.fps) - 120) < 10.0

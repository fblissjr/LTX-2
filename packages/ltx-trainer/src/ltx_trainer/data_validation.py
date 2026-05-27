"""Validate a precomputed training dataset BEFORE spending GPU hours on it.

Catches the "data smells wrong but the code says nothing" failures that
otherwise surface as a LoRA that won't learn — and get blamed on the model code:

  * source count mismatch (the silent-intersection footgun): PrecomputedDataset
    pairs samples by INTERSECTION across sources, so 100 video latents + 3 audio
    latents trains on 3 samples with only an INFO log. We surface it as an ERROR.
  * wrong latent shapes (video must be [128, F, H, W]; audio [8, T, 16]).
  * NaN / inf / all-zero latents (a broken encode pass).
  * missing conditions (esp. audio_prompt_embeds when training with audio — the
    strategy needs it; None → a deep model failure).
  * audio<->video temporal MISALIGNMENT: the audio latent length should track the
    video duration by a roughly constant rate. We learn that rate empirically from
    the dataset (median) and flag per-sample outliers — this catches an audio clip
    that doesn't correspond to its video (the worst silent data bug for an
    audio->video guidance LoRA).

Pure / CPU-only: loads the precomputed .pt tensors, no model, no GPU. Run it on
your real precomputed dir; unit-tested against synthetic .pt fixtures.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Expected latent layouts (LTX-2). Video latent: [C=128, F, H, W]. Audio latent
# (non-patchified, as written by process_videos): [C=8, T, mel=16].
VIDEO_LATENT_CHANNELS = 128
AUDIO_LATENT_CHANNELS = 8
AUDIO_MEL_BINS = 16
VIDEO_TEMPORAL_SCALE = 8  # latent frame -> pixel frame: (F-1)*8 + 1

# Default ratio tolerance for the empirical audio/video-rate alignment check.
DEFAULT_ALIGNMENT_TOLERANCE = 0.10  # 10% off the dataset median = suspicious


@dataclass
class SampleReport:
    index: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    video_shape: tuple[int, ...] | None = None
    audio_shape: tuple[int, ...] | None = None
    ref_shape: tuple[int, ...] | None = None
    video_duration_s: float | None = None
    audio_frames: int | None = None
    audio_per_second: float | None = None  # audio_frames / video_duration_s


@dataclass
class DatasetReport:
    source_counts: dict[str, int] = field(default_factory=dict)  # raw .pt count per source dir
    paired_count: int = 0  # samples after intersection (what training actually sees)
    samples: list[SampleReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio_rate_median: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors and all(not s.errors for s in self.samples)

    def all_errors(self) -> list[str]:
        out = list(self.errors)
        for s in self.samples:
            out.extend(f"sample {s.index}: {e}" for e in s.errors)
        return out

    def all_warnings(self) -> list[str]:
        out = list(self.warnings)
        for s in self.samples:
            out.extend(f"sample {s.index}: {w}" for w in s.warnings)
        return out


def count_source_files(data_root: Path, source_dirs: list[str]) -> dict[str, int]:
    """Raw `**/*.pt` count per source dir (NOT the paired/intersection count)."""
    counts: dict[str, int] = {}
    for d in source_dirs:
        p = data_root / d
        counts[d] = len(list(p.glob("**/*.pt"))) if p.exists() else -1  # -1 = dir missing
    return counts


def _video_duration_seconds(sample: dict) -> float | None:
    """Clip duration in seconds from a video-latent sample dict."""
    latents = sample.get("latents")
    if not isinstance(latents, torch.Tensor) or latents.dim() != 4:
        return None
    num_frames = int(sample.get("num_frames", latents.shape[1]))
    fps = float(sample.get("fps", 0) or 0)
    if fps <= 0:
        return None
    pixel_frames = (num_frames - 1) * VIDEO_TEMPORAL_SCALE + 1
    return pixel_frames / fps


def _check_latent(t: object, name: str, expected_channels: int, sr: SampleReport) -> None:
    """Shape + finiteness checks for a [C, ...] latent tensor."""
    if not isinstance(t, torch.Tensor):
        sr.errors.append(f"{name} latent missing or not a tensor")
        return
    if t.dim() != 4:
        sr.errors.append(f"{name} latent has {t.dim()} dims, expected 4 ([C, ...])")
    elif t.shape[0] != expected_channels:
        sr.errors.append(f"{name} latent channels={t.shape[0]}, expected {expected_channels}")
    if torch.isnan(t).any() or torch.isinf(t).any():
        sr.errors.append(f"{name} latent contains NaN/Inf (broken encode pass)")
    elif float(t.abs().max()) == 0.0:
        sr.warnings.append(f"{name} latent is all zeros (suspicious)")


def validate_sample(
    sample: dict,
    *,
    with_audio: bool,
    with_reference: bool,
    video_key: str = "latents",
    audio_key: str = "audio_latents",
    ref_key: str = "ref_latents",
    conditions_key: str = "conditions",
) -> SampleReport:
    """Validate one already-loaded paired sample (a dict of the source outputs)."""
    sr = SampleReport(index=int(sample.get("idx", -1)))

    video = sample.get(video_key, {})
    video_t = video.get("latents") if isinstance(video, dict) else None
    _check_latent(video_t, "video", VIDEO_LATENT_CHANNELS, sr)
    if isinstance(video_t, torch.Tensor) and video_t.dim() == 4:
        sr.video_shape = tuple(video_t.shape)
        sr.video_duration_s = _video_duration_seconds(video)

    if with_reference:
        ref = sample.get(ref_key, {})
        ref_t = ref.get("latents") if isinstance(ref, dict) else None
        _check_latent(ref_t, "reference", VIDEO_LATENT_CHANNELS, sr)
        if isinstance(ref_t, torch.Tensor) and ref_t.dim() == 4:
            sr.ref_shape = tuple(ref_t.shape)

    if with_audio:
        audio = sample.get(audio_key, {})
        audio_t = audio.get("latents") if isinstance(audio, dict) else None
        if not isinstance(audio_t, torch.Tensor):
            sr.errors.append("audio latent missing (with_audio=True but no audio_latents)")
        else:
            if audio_t.dim() != 3:
                sr.errors.append(f"audio latent has {audio_t.dim()} dims, expected 3 ([C, T, mel])")
            else:
                if audio_t.shape[0] != AUDIO_LATENT_CHANNELS:
                    sr.errors.append(f"audio channels={audio_t.shape[0]}, expected {AUDIO_LATENT_CHANNELS}")
                if audio_t.shape[2] != AUDIO_MEL_BINS:
                    sr.errors.append(f"audio mel bins={audio_t.shape[2]}, expected {AUDIO_MEL_BINS}")
                sr.audio_frames = int(audio_t.shape[1])
            if torch.isnan(audio_t).any() or torch.isinf(audio_t).any():
                sr.errors.append("audio latent contains NaN/Inf")
            sr.audio_shape = tuple(audio_t.shape)
        if sr.audio_frames and sr.video_duration_s and sr.video_duration_s > 0:
            sr.audio_per_second = sr.audio_frames / sr.video_duration_s

    conds = sample.get(conditions_key, {})
    if not isinstance(conds, dict):
        sr.errors.append("conditions missing")
    else:
        if not isinstance(conds.get("video_prompt_embeds"), torch.Tensor):
            sr.errors.append("conditions.video_prompt_embeds missing")
        if conds.get("prompt_attention_mask") is None:
            sr.errors.append("conditions.prompt_attention_mask missing")
        if with_audio and not isinstance(conds.get("audio_prompt_embeds"), torch.Tensor):
            sr.errors.append(
                "conditions.audio_prompt_embeds missing/None (with_audio needs the audio "
                "text-encoder connector — the strategy will refuse this)"
            )
    return sr


def validate_dataset(
    data_root: str | Path,
    *,
    with_audio: bool,
    with_reference: bool = True,
    sample_limit: int | None = 16,
    alignment_tolerance: float = DEFAULT_ALIGNMENT_TOLERANCE,
) -> DatasetReport:
    """Validate a precomputed dataset dir end to end. CPU only, no model.

    Loads via PrecomputedDataset (so it sees exactly what training sees, incl. the
    intersection pairing), cross-checks raw per-source counts to expose silent
    intersection drops, then per-sample shape/finiteness/alignment checks.
    """
    from ltx_trainer.datasets import PRECOMPUTED_DIR_NAME, PrecomputedDataset

    report = DatasetReport()
    root = Path(data_root).expanduser().resolve()
    scan_root = root / PRECOMPUTED_DIR_NAME if (root / PRECOMPUTED_DIR_NAME).exists() else root

    source_dirs = ["latents", "conditions"]
    if with_reference:
        source_dirs.append("reference_latents")
    if with_audio:
        source_dirs.append("audio_latents")

    # Raw per-source counts (the intersection footgun lives in the GAP between these).
    report.source_counts = count_source_files(scan_root, source_dirs)
    for d, c in report.source_counts.items():
        if c < 0:
            report.errors.append(f"source dir '{d}' does not exist under {scan_root}")
    present = {d: c for d, c in report.source_counts.items() if c >= 0}
    if present and len(set(present.values())) > 1:
        report.errors.append(
            f"source file counts differ {present} — PrecomputedDataset pairs by "
            "intersection, so training will SILENTLY use only the overlap. Re-run "
            "precompute so every source has one .pt per sample."
        )

    # Mirror VideoToVideoStrategy.get_data_sources exactly (reference_latents dir
    # -> "ref_latents" output key) so the validator sees what the strategy sees.
    data_sources = {"latents": "latents", "conditions": "conditions"}
    if with_reference:
        data_sources["reference_latents"] = "ref_latents"
    if with_audio:
        data_sources["audio_latents"] = "audio_latents"
    try:
        ds = PrecomputedDataset(str(root), data_sources)
    except Exception as e:  # noqa: BLE001 - surface any setup failure as a report error
        report.errors.append(f"PrecomputedDataset failed to load: {e}")
        return report

    report.paired_count = len(ds)
    if report.paired_count == 0:
        report.errors.append("0 paired samples after intersection")
        return report

    n = report.paired_count if sample_limit is None else min(sample_limit, report.paired_count)
    for i in range(n):
        sample = ds[i]  # PrecomputedDataset.__getitem__ sets sample["idx"] = i
        report.samples.append(
            validate_sample(sample, with_audio=with_audio, with_reference=with_reference)
        )

    # Empirical audio/video-rate alignment: learn the rate from the data, flag outliers.
    if with_audio:
        rates = [s.audio_per_second for s in report.samples if s.audio_per_second]
        if rates:
            report.audio_rate_median = statistics.median(rates)
            med = report.audio_rate_median
            for s in report.samples:
                if s.audio_per_second and med > 0:
                    if abs(s.audio_per_second - med) / med > alignment_tolerance:
                        s.warnings.append(
                            f"audio/video rate {s.audio_per_second:.2f}/s deviates >"
                            f"{alignment_tolerance:.0%} from dataset median {med:.2f}/s "
                            "(audio may not correspond to this video)"
                        )
    return report


def format_report(report: DatasetReport) -> str:
    """Human-readable smell report for the console."""
    lines = ["=== precomputed training-data validation ==="]
    lines.append(f"source counts (raw .pt): {report.source_counts}")
    lines.append(f"paired samples (training will use): {report.paired_count}")
    if report.audio_rate_median is not None:
        lines.append(f"audio rate (median): {report.audio_rate_median:.2f} latent-frames/s")
    shown = report.samples[:8]
    for s in shown:
        bits = [f"  sample {s.index}: video={s.video_shape}"]
        if s.audio_shape is not None:
            bits.append(f"audio={s.audio_shape}")
        if s.video_duration_s is not None:
            bits.append(f"dur={s.video_duration_s:.2f}s")
        if s.audio_per_second is not None:
            bits.append(f"a/v={s.audio_per_second:.1f}/s")
        lines.append(" ".join(bits))
    errs = report.all_errors()
    warns = report.all_warnings()
    if errs:
        lines.append(f"ERRORS ({len(errs)}):")
        lines.extend(f"  ✗ {e}" for e in errs[:20])
    if warns:
        lines.append(f"WARNINGS ({len(warns)}):")
        lines.extend(f"  ! {w}" for w in warns[:20])
    lines.append("VERDICT: " + ("OK — data looks right" if report.ok else "PROBLEMS FOUND — fix before training"))
    return "\n".join(lines)

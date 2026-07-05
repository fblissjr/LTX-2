#!/usr/bin/env python
"""Verify a precomputed training dataset looks right BEFORE training.

Run this on your precomputed dir (the output of process_dataset.py) to catch
count mismatches, wrong latent shapes, NaN/Inf encodes, missing conditions, and
audio<->video misalignment — so a LoRA that won't learn gets blamed on the data,
not the model code. CPU only, no GPU, no model load.

    uv run python packages/ltx-trainer/scripts/verify_training_data.py <data_root> --with-audio

Exit code 0 = OK, 1 = problems found (see the report).
"""

from __future__ import annotations

import argparse
import sys

from ltx_trainer.data_validation import DEFAULT_ALIGNMENT_TOLERANCE, format_report, validate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", help="Precomputed dataset root (the dir process_dataset.py wrote)")
    ap.add_argument("--with-audio", action="store_true", help="Dataset includes audio_latents/ (audio-guided IC-LoRA)")
    ap.add_argument("--no-reference", action="store_true", help="Dataset has no reference_latents/ (non-IC-LoRA)")
    ap.add_argument("--sample-limit", type=int, default=16, help="Max samples to deep-check (None=all)")
    ap.add_argument("--alignment-tolerance", type=float, default=DEFAULT_ALIGNMENT_TOLERANCE,
                    help="Max audio/video-rate deviation from the dataset median before warning")
    args = ap.parse_args()

    report = validate_dataset(
        args.data_root,
        with_audio=args.with_audio,
        with_reference=not args.no_reference,
        sample_limit=args.sample_limit,
        alignment_tolerance=args.alignment_tolerance,
    )
    print(format_report(report))
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()

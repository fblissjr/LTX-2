#!/usr/bin/env python
"""Score an audio→video LoRA objectively: does it make the video track the audio
MORE than the no-LoRA baseline, across a swept input? (data plan §4 audio-swap
test, made measurable + scalable.)

You render the A/B inference outputs on GPU (LoRA arm + baseline arm, same prompt/
seed/keyframes, only the input audio swept), list them in a manifest, and this
scores them on CPU. Exits nonzero if the LoRA doesn't earn its keep.

Manifest (json):
  {
    "cases": [
      {"expected": 80,  "lora_video": "lora/c0.mp4",  "baseline_video": "base/c0.mp4"},
      {"expected": 120, "lora_video": "lora/c1.mp4",  "baseline_video": "base/c1.mp4"},
      ...
    ],
    "neutral_cases": [{"lora_video": "lora/neutral0.mp4"}]   # optional base-preservation
  }

    uv run python packages/ltx-trainer/scripts/run_audio_coupling_eval.py \
        manifest.json --coupling beat_pulse --res 256x256x25
"""

from __future__ import annotations

import argparse
import sys

from ltx_trainer.eval_audio_coupling import COUPLING_METRICS, evaluate_from_manifest, format_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="json manifest of A/B cases (see header)")
    ap.add_argument("--coupling", default="beat_pulse", choices=sorted(COUPLING_METRICS))
    ap.add_argument("--res", default="256x256x25", help="WxHxFPS of the output videos")
    args = ap.parse_args()

    w, h, fps = (int(x) for x in args.res.split("x"))
    report = evaluate_from_manifest(args.manifest, coupling=args.coupling, width=w, height=h, fps=fps)
    print(format_report(report))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

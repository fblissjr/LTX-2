#!/usr/bin/env python
"""Generate a synthetic audio↔video dataset with a KNOWN, measurable coupling.

For proving the training MECHANISM before real footage (data plan §8). Currently
beat→pulse: a shape pulses on the audio's beats; measure_pulse_rate recovers the
BPM, so a trained LoRA's output can be scored the same way (objective eval).
Writes clips/ + captions.json (handle-only) + manifest.jsonl (ground-truth bpm).
Procedural, CPU-only (numpy + ffmpeg), no model.

    uv run python packages/ltx-trainer/scripts/generate_synthetic_av_data.py \
        --out /tmp/synth --n 30 --bpm-min 60 --bpm-max 160 --duration 3 --res 256x256x25
"""

from __future__ import annotations

import argparse

from ltx_trainer.synthetic_av import (
    generate_dataset,
    generate_dataset_paired_refs,
    generate_dataset_static_ref,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--bpm-min", type=float, default=60.0)
    ap.add_argument("--bpm-max", type=float, default=160.0)
    ap.add_argument("--duration", type=float, default=3.0, help="seconds (snapped to the 8k+1 frame rule)")
    ap.add_argument("--res", default="256x256x25", help="WxHxFPS")
    ap.add_argument("--seed", type=int, default=0)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--paired-refs", action="store_true",
                      help="v2: each row gets a SEPARATE reference clip, same visual identity "
                           "but a different BPM (audio load-bearing). Has a corr(target,ref) "
                           "leak in narrow ranges — prefer --static-refs for the gate.")
    mode.add_argument("--static-refs", action="store_true",
                      help="v3 (gate): reference is a FROZEN identity frame (no pulse, no rate). "
                           "Audio is the only temporal signal; no corr leak by construction.")
    ap.add_argument("--min-bpm-gap", type=float, default=20.0,
                    help="(paired-refs only) minimum |target_bpm - reference_bpm| per row.")
    args = ap.parse_args()

    w, h, fps = (int(x) for x in args.res.split("x"))
    common = dict(bpm_range=(args.bpm_min, args.bpm_max), duration_s=args.duration,
                  fps=fps, width=w, height=h, seed=args.seed)

    if args.static_refs:
        captions = generate_dataset_static_ref(args.out, args.n, **common)
        kind = f"{args.n} target clips + {args.n} static reference clips"
    elif args.paired_refs:
        captions = generate_dataset_paired_refs(args.out, args.n, min_bpm_gap=args.min_bpm_gap, **common)
        kind = f"{args.n} target clips + {args.n} reference clips"
    else:
        captions = generate_dataset(args.out, args.n, **common)
        kind = f"{args.n} clips"
    print(f"Wrote {kind} + {captions} + manifest.jsonl under {args.out}")


if __name__ == "__main__":
    main()

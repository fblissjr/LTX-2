#!/usr/bin/env python
"""End-to-end integration smoke for the audio→video IC-LoRA pipeline.

Drives the REAL pipeline — synthetic clips → process_dataset.py --with-audio →
data validation (hard gate) → train.py (step-capped, hard gate: a checkpoint) —
so the integration is exercised once and failures land at a specific stage.

    # data pipeline only (no GPU train; stops after validation):
    uv run python packages/ltx-trainer/scripts/run_e2e_smoke.py --workdir /tmp/smoke

    # full smoke including a few train steps (needs a real base config):
    uv run python packages/ltx-trainer/scripts/run_e2e_smoke.py --workdir /tmp/smoke \
        --base-config packages/ltx-trainer/configs/ltx2_av_lora_low_vram.yaml --steps 3

    # use your own real clips instead of synthetic (a captions json with video/caption[/reference]):
    uv run python packages/ltx-trainer/scripts/run_e2e_smoke.py --workdir /tmp/smoke --captions my_captions.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ltx_trainer.e2e_smoke import SmokeGateError, run_smoke


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="Scratch dir for generated data + precompute + smoke output")
    ap.add_argument("--n-clips", type=int, default=4, help="Synthetic clips to generate (ignored if --captions)")
    ap.add_argument("--captions", type=Path, default=None, help="Use a real captions json instead of synthetic")
    ap.add_argument("--resolution-bucket", default="256x256x25", help="Synthetic clip spec: WxHxFPS (NOT frames)")
    ap.add_argument("--duration", type=float, default=3.0, help="Synthetic clip length in seconds")
    ap.add_argument("--dataset-bucket", default=None,
                    help="process_dataset WxHxFRAMES bucket (REQUIRED with --captions; derived for synthetic)")
    ap.add_argument("--model-path", default=None,
                    help="Single-file LTX-2 checkpoint with VAEs+projectors (REQUIRED for precompute)")
    ap.add_argument("--text-encoder-path", default=None, help="Gemma model dir (REQUIRED for precompute)")
    ap.add_argument("--base-config", type=Path, default=None, help="Real train YAML to step-cap for the train smoke")
    ap.add_argument("--steps", type=int, default=3, help="Training steps for the smoke")
    args = ap.parse_args()

    try:
        run_smoke(
            workdir=args.workdir, n_clips=args.n_clips, captions_path=args.captions,
            resolution_bucket=args.resolution_bucket, duration_s=args.duration,
            dataset_bucket=args.dataset_bucket, model_path=args.model_path,
            text_encoder_path=args.text_encoder_path, base_config=args.base_config, steps=args.steps,
        )
    except SmokeGateError as e:
        raise SystemExit(f"\nSMOKE FAILED at a gate:\n{e}") from e


if __name__ == "__main__":
    main()

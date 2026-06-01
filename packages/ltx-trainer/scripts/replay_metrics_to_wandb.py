#!/usr/bin/env python
"""Replay an offline metrics.jsonl into a fresh W&B run.

The always-on metrics.jsonl is the durable record of a run; W&B SDK 0.24.0 has a known bug that can
SILENTLY fail to upload. After upgrading wandb, re-log any affected run by replaying its JSONL here.
Train rows (loss / ema_loss / grad_norm / lr / step_time / sigma buckets / ref_gap) and held-out val
rows (val_loss / val_ref_gap) are logged at their step under train/* and val/* so the panels match a
live run.

Usage:
  replay_metrics_to_wandb.py --metrics <run>/metrics.jsonl --run-name <name> \
    [--project ltx2-audio-ic-lora] [--config <run>/training_config.yaml] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# metrics.jsonl key -> wandb panel key, so a replay matches the live-run schema.
_TRAIN_MAP = {
    "loss": "train/loss",
    "ema_loss": "train/ema_loss",
    "lr": "train/learning_rate",
    "grad_norm": "train/grad_norm",
    "step_time": "train/step_time",
    "ref_gap": "train/ref_gap",
}


def _to_wandb(row: dict) -> dict:
    """Map one JSONL row to a wandb log dict (None values dropped; sigma buckets pass through)."""
    if "val_loss" in row:  # a held-out eval row
        out = {"val/loss": row["val_loss"]}
        if row.get("val_ref_gap") is not None:
            out["val/ref_gap"] = row["val_ref_gap"]
        return out
    out = {}
    for k, v in row.items():
        if k == "step" or v is None:
            continue
        out[_TRAIN_MAP.get(k, k)] = v  # train/loss_sigma_* already namespaced -> pass through
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay metrics.jsonl into a fresh W&B run.")
    ap.add_argument("--metrics", required=True, help="path to a run's metrics.jsonl")
    ap.add_argument("--run-name", required=True, help="name for the replayed wandb run")
    ap.add_argument("--project", default="ltx2-audio-ic-lora")
    ap.add_argument("--config", default=None, help="optional training_config.yaml to attach as wandb config")
    ap.add_argument("--dry-run", action="store_true", help="parse + summarize only; do NOT touch wandb")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {args.metrics}")
    train_rows = [r for r in rows if "val_loss" not in r]
    val_rows = [r for r in rows if "val_loss" in r]
    print(
        f"{len(rows)} rows: {len(train_rows)} train, {len(val_rows)} val; "
        f"steps {rows[0].get('step')}..{rows[-1].get('step')}"
    )

    if args.dry_run:
        for r in rows[:1] + rows[-2:]:
            print("  step", r.get("step"), "->", _to_wandb(r))
        return

    import wandb  # imported lazily so --dry-run needs no wandb install
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else None
    run = wandb.init(project=args.project, name=args.run_name, config=cfg, tags=["replay", "from-metrics-jsonl"])
    logged = 0
    for r in rows:
        data = _to_wandb(r)
        if data:
            wandb.log(data, step=r.get("step"))
            logged += 1
    run.finish()
    print(f"replayed {logged} rows -> wandb run '{args.run_name}' (project {args.project})")


if __name__ == "__main__":
    main()

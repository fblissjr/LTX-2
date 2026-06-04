#!/usr/bin/env python
"""Sigma-resolved reference-attribution gap: WHERE on the noise schedule is the reference used?

The flow-matching objective lets the noised TARGET leak the attribute (e.g. identity, which is
literally on screen) for much of the schedule, so the in-context reference is only *forced* to
matter at high sigma where the target is destroyed. A single averaged ref_gap hides this. This
sweeps sigma and reports, at each level:

    gap(sigma) = loss(wrong reference) - loss(correct reference)      (paired noise)

The two forwards share timestep + noise (metrics.paired_difference), so the difference is purely the
reference. Reading the curve — measured experience (2026-06 identity runs), not the naive prior:
do NOT expect a clean "0 at low sigma, rising at high sigma" shape. A NEGATIVE gap (correct
reference reconstructs WORSE) is possible and ambiguous: it can mean the reference is load-bearing
but pulls toward a generic rendition of the shared attribute (a reconstruction penalty), or it can
be noise — which is why every row now carries a 95% CI and its raw per-pair gaps. Reconstruction
gaps on leaked-target (e.g. same-clip-reference) data cannot localize generation-time reference
control; treat the curve as a canary, and treat generation-from-noise swaps as the arbiter.

Reuses the trainer's own setup + ref-gap computation verbatim (no re-derivation): builds LtxvTrainer
with the trained LoRA loaded, overrides the timestep sampler to a fixed sigma per sweep point, and
runs the held-out val set through the exact _reference_attribution_gap path. Forward-only, no_grad.
Run it AFTER the training run frees the GPU.

Usage:
  reference_gap_by_sigma.py --config <train.yaml> --checkpoint <lora_ckpt> \
    [--sigmas 0.05,0.1,0.3,0.5,0.7,0.9,0.95] [--max-batches 24] [--out gap_by_sigma.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from accelerate.utils import send_to_device

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.metrics import summarize_gaps
from ltx_trainer.trainer import LtxvTrainer


class _FixedSigma:
    """Timestep sampler that pins every sample to one sigma (matches TimestepSampler.sample_for)."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample_for(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full((latents.shape[0],), self.sigma, dtype=latents.dtype, device=latents.device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sigma-resolved reference-attribution gap.")
    ap.add_argument("--config", required=True, help="the training YAML (e.g. the final run config)")
    ap.add_argument("--checkpoint", required=True, help="trained LoRA checkpoint (dir / file / step) to load")
    # End-points included by default: the schedule's extremes are where the leak story lives
    # (the 2026-06 runs' 0.1..0.9 grid left both ends unmeasured exactly where it mattered).
    ap.add_argument(
        "--sigmas", default="0.05,0.1,0.3,0.5,0.7,0.9,0.95", help="comma-separated sigma sweep points"
    )
    ap.add_argument("--max-batches", type=int, default=24, help="held-out batches per sigma point")
    ap.add_argument("--out", default=None, help="optional JSON output path for the curve")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]

    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg_dict.setdefault("model", {})["load_checkpoint"] = args.checkpoint
    config = LtxTrainerConfig(**cfg_dict)

    # Full model setup happens in __init__ (load -> int8 quantize -> LoRA-from-checkpoint ->
    # block-swap -> accelerator.prepare). train() only adds the dataloaders + sampler, so we
    # build just the val loader and drive the reference-gap path ourselves.
    trainer = LtxvTrainer(config)
    trainer._init_dataloader()
    if trainer._val_dataloader is None:
        raise SystemExit("No held-out val set: set validation.holdout_data_root in the config.")

    device = trainer._accelerator.device
    trainer._transformer.eval()

    rows: list[dict] = []
    print(f"{'sigma':>6} | {'mean ref_gap':>14} | {'95% CI':>20} | {'mean loss@sigma':>15} | {'n_pairs':>7}")
    print("-" * 75)
    with torch.no_grad():
        for sigma in sigmas:
            trainer._timestep_sampler = _FixedSigma(sigma)
            gaps: list[float] = []
            losses: list[float] = []
            prev_ref: dict | None = None
            torch.manual_seed(config.validation.seed)  # same noise across sweep points -> comparable
            for i, batch in enumerate(trainer._val_dataloader):
                if i >= args.max_batches:
                    break
                batch = send_to_device(batch, device)
                trainer._prepare_conditions(batch["conditions"])
                losses.append(trainer._forward_loss_no_grad(batch))
                g = trainer._reference_attribution_gap(batch, prev_reference=prev_ref)
                if g is not None:
                    gaps.append(g)
                prev_ref = batch.get("reference_audio_latents")
            s = summarize_gaps(gaps)
            ml = sum(losses) / len(losses) if losses else float("nan")
            # Keep the original keys (downstream notes already parse them); add the spread
            # fields + the raw per-pair gaps so the curve carries its own noise yardstick
            # and stays re-analyzable without a GPU.
            rows.append(
                {
                    "sigma": sigma,
                    "mean_ref_gap": s["mean"],
                    "std_ref_gap": s["std"],
                    "ci95_lo": s["ci95_lo"],
                    "ci95_hi": s["ci95_hi"],
                    "mean_loss_at_sigma": ml,
                    "n_pairs": s["n"],
                    "gaps": gaps,
                }
            )
            print(
                f"{sigma:>6.2f} | {s['mean']:>+14.5f} | [{s['ci95_lo']:>+8.5f},{s['ci95_hi']:>+8.5f}]"
                f" | {ml:>15.5f} | {s['n']:>7d}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

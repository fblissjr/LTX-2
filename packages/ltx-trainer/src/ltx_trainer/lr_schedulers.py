"""Learning-rate scheduler construction, factored out of the trainer so it is pure and unit-testable.

The cosine branch supports an optional ``warmup_steps``: a linear ramp-in chained before the cosine
decay via ``SequentialLR``, so a high peak LR doesn't slam freshly-initialised adapters cold. The
``eta_min`` floor keeps the cosine tail from decaying to ~0. All other branches are ported verbatim
from the original trainer method.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
    StepLR,
)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str | None,
    steps: int,
    params: dict[str, Any] | None = None,
) -> LRScheduler | None:
    """Build an LR scheduler. Returns None for ``constant`` (and ``None``) types.

    ``params`` mirrors ``OptimizationConfig.scheduler_params``; keys are consumed per branch.
    For ``cosine``, ``warmup_steps`` (linear ramp-in) and ``eta_min`` (decay floor) are honoured.
    """
    params = dict(params or {})

    if scheduler_type is None or scheduler_type == "constant":
        return None

    if scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=params.pop("start_factor", 1.0),
            end_factor=params.pop("end_factor", 0.1),
            total_iters=steps,
            **params,
        )

    if scheduler_type == "cosine":
        warmup = int(params.pop("warmup_steps", 0))
        eta_min = params.pop("eta_min", 0)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, steps - warmup), eta_min=eta_min, **params)
        if warmup > 0:
            warm = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup)
            return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])
        return cosine

    if scheduler_type == "cosine_with_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.pop("T_0", steps // 4),
            T_mult=params.pop("T_mult", 1),
            eta_min=params.pop("eta_min", 5e-5),
            **params,
        )

    if scheduler_type == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=steps,
            power=params.pop("power", 1.0),
            **params,
        )

    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=params.pop("step_size", steps // 2),
            gamma=params.pop("gamma", 0.1),
            **params,
        )

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")

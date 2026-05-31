"""LR scheduler construction — pure, no model/GPU. The cosine branch grew warmup support
(a linear ramp-in chained before the cosine decay via SequentialLR) so a high peak LR like 2e-4
doesn't hit the fresh adapters cold; the eta_min floor keeps the tail from decaying to ~0. These
tests step a one-parameter optimizer through the schedule and read the LR off it."""

from __future__ import annotations

import torch

from ltx_trainer.lr_schedulers import build_lr_scheduler


def _opt(lr: float) -> torch.optim.Optimizer:
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=lr)


def _lr(opt: torch.optim.Optimizer) -> float:
    return opt.param_groups[0]["lr"]


def test_cosine_with_warmup_ramps_then_decays_to_floor():
    lr, warmup, steps, floor = 2e-4, 10, 100, 2e-5
    opt = _opt(lr)
    sched = build_lr_scheduler(
        opt, scheduler_type="cosine", steps=steps, params={"warmup_steps": warmup, "eta_min": floor}
    )
    start = _lr(opt)
    assert start < lr * 0.5  # warmup starts well below peak (linear ramp-in)
    lrs = [start]
    for _ in range(steps):
        opt.step()
        sched.step()
        lrs.append(_lr(opt))
    assert abs(lrs[warmup] - lr) < lr * 0.05  # at the end of warmup we are at (near) the peak LR
    assert lrs[warmup] > lrs[-1]  # then it decays
    assert lrs[-1] >= floor - 1e-9  # never below the eta_min floor
    assert abs(lrs[-1] - floor) < lr * 0.1  # and it lands near the floor by the end


def test_cosine_without_warmup_starts_at_peak():
    lr = 1e-4
    opt = _opt(lr)
    sched = build_lr_scheduler(opt, scheduler_type="cosine", steps=50, params={"eta_min": 0.0})
    assert abs(_lr(opt) - lr) < 1e-12  # no warmup -> first LR is the peak
    for _ in range(50):
        opt.step()
        sched.step()
    assert _lr(opt) < lr * 0.1  # decayed toward 0


def test_linear_branch_unchanged():
    lr = 1e-4
    opt = _opt(lr)
    sched = build_lr_scheduler(opt, scheduler_type="linear", steps=10, params={})
    assert abs(_lr(opt) - lr) < 1e-12  # start_factor default 1.0
    for _ in range(10):
        opt.step()
        sched.step()
    assert abs(_lr(opt) - lr * 0.1) < lr * 0.02  # end_factor default 0.1


def test_constant_returns_none():
    assert build_lr_scheduler(_opt(1e-4), scheduler_type="constant", steps=10, params={}) is None


def test_unknown_type_raises():
    try:
        build_lr_scheduler(_opt(1e-4), scheduler_type="banana", steps=10, params={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown scheduler type")

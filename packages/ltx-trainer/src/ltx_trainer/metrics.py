"""Training telemetry: EMA smoothing, a convergence/overfit/underfit monitor, and a
JSONL metrics writer.

Why this exists: the live Rich progress bar shows loss/lr but writes nothing to a
redirected/unattended log, and W&B is opt-in — so a backgrounded run left no curve and
no signal for "is it learning / converged / overfitting?". This module gives an
always-on, offline, greppable metrics stream plus a pure monitor that turns the loss
curves into actionable flags. Everything here is pure (no torch, no model) so it unit-tests
without a GPU; the trainer feeds it scalars each eval/log step.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EMA:
    """Exponential moving average. First value seeds it; `beta` is the decay (higher =
    smoother). Used to denoise the very noisy per-step flow-matching loss before it's
    handed to the monitor / shown on the bar."""

    def __init__(self, beta: float = 0.98) -> None:
        self.beta = beta
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.beta * self.value + (1.0 - self.beta) * x
        return self.value


@dataclass
class ConvergenceStatus:
    """Snapshot of what the monitor currently believes about the run."""

    not_learning: bool = False
    plateaued: bool = False
    overfitting: bool = False
    converged: bool = False
    ref_decorative: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def should_stop(self) -> bool:
        """Either we're past the optimum (overfitting) or there's nothing more to gain."""
        return self.overfitting or self.converged


class ConvergenceMonitor:
    """Watches the (ideally EMA-smoothed) train loss, plus optional held-out loss and a
    reference-attribution gap, and flags the situation each eval:

      - not_learning : after warmup, the best train loss hasn't dropped >= `min_rel_drop`
                       from the initial loss (underfit / mis-wired).
      - plateaued    : train loss is flat (relative range over the last `plateau_window`
                       evals < `plateau_tol`).
      - overfitting  : held-out loss has risen for `overfit_patience` consecutive evals.
      - converged    : plateaued and not overfitting (no more to gain → safe to stop).
      - ref_decorative: reference-attribution gap (loss with a shuffled reference minus
                       loss with the correct reference) <= `ref_gap_tol`, i.e. the model
                       isn't using the in-context reference (the task-specific "not
                       learning the coupling" signal).

    Pure: feed scalars, never touches the model.
    """

    def __init__(
        self,
        *,
        warmup_evals: int = 3,
        plateau_tol: float = 0.005,
        plateau_window: int = 5,
        min_rel_drop: float = 0.05,
        overfit_patience: int = 3,
        ref_gap_tol: float = 0.0,
    ) -> None:
        self.warmup_evals = warmup_evals
        self.plateau_tol = plateau_tol
        self.plateau_window = plateau_window
        self.min_rel_drop = min_rel_drop
        self.overfit_patience = overfit_patience
        self.ref_gap_tol = ref_gap_tol

        self._train: deque[float] = deque(maxlen=max(plateau_window, 2))
        self._n_evals = 0
        self._init_loss: float | None = None
        self._min_loss = float("inf")
        self._prev_heldout: float | None = None
        self._heldout_rising = 0
        self._last_ref_gap: float | None = None

    def update(
        self,
        *,
        step: int,
        train_loss: float,
        heldout_loss: float | None = None,
        ref_gap: float | None = None,
    ) -> ConvergenceStatus:
        self._n_evals += 1
        if self._init_loss is None:
            self._init_loss = train_loss
        self._min_loss = min(self._min_loss, train_loss)
        self._train.append(train_loss)

        if heldout_loss is not None:
            if self._prev_heldout is not None and heldout_loss > self._prev_heldout:
                self._heldout_rising += 1
            else:
                self._heldout_rising = 0
            self._prev_heldout = heldout_loss

        if ref_gap is not None:
            self._last_ref_gap = ref_gap

        return self.status()

    def status(self) -> ConvergenceStatus:
        s = ConvergenceStatus()

        past_warmup = self._n_evals > self.warmup_evals
        if past_warmup and self._init_loss and self._min_loss >= self._init_loss * (1.0 - self.min_rel_drop):
            s.not_learning = True
            s.messages.append(
                f"train loss has not dropped {self.min_rel_drop:.0%} from start "
                f"({self._init_loss:.4f}) after {self._n_evals} evals — underfit / not learning"
            )

        if len(self._train) >= self.plateau_window:
            window = list(self._train)[-self.plateau_window :]
            mean = sum(window) / len(window)
            rel_range = (max(window) - min(window)) / mean if mean else 0.0
            if rel_range < self.plateau_tol:
                s.plateaued = True
                s.messages.append(
                    f"train loss plateaued (range <{self.plateau_tol:.1%} over {self.plateau_window} evals)"
                )

        if self._heldout_rising >= self.overfit_patience:
            s.overfitting = True
            s.messages.append(
                f"held-out loss rising for {self._heldout_rising} evals while train falls — "
                "overfitting; stop or use an earlier checkpoint"
            )

        if s.plateaued and not s.overfitting:
            s.converged = True
            s.messages.append("train loss converged (plateaued, held-out not rising) — safe to stop")

        if self._last_ref_gap is not None and self._last_ref_gap <= self.ref_gap_tol:
            s.ref_decorative = True
            s.messages.append(
                f"reference-attribution gap {self._last_ref_gap:+.4f} <= {self.ref_gap_tol} — the "
                "reference is decorative (model not using it / not learning the coupling)"
            )

        return s


class MetricsWriter:
    """Append one JSON object per logged step to a JSONL file. Always-on, offline,
    greppable — the source of truth a redirected/unattended run can't get from the Rich
    progress bar. Line-buffered so a `tail -f` sees rows live."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, default=float) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def build_metrics_row(
    *,
    step: int,
    loss: float,
    ema_loss: float,
    lr: float,
    step_time: float,
    grad_norm: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one JSONL metrics row from per-step scalars.

    ``grad_norm`` is kept as an explicit ``null`` when gradient clipping is off, so the
    column is always present and the schema stays stable for downstream tooling / grep.
    ``extra`` (e.g. the per-sigma-bucket losses) is merged in alongside the core fields.
    """
    row: dict[str, Any] = {
        "step": step,
        "loss": loss,
        "ema_loss": ema_loss,
        "grad_norm": grad_norm,
        "lr": lr,
        "step_time": step_time,
    }
    if extra:
        row.update(extra)
    return row

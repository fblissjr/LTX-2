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
import math
from collections import deque
from collections.abc import Callable, Sequence
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
                       loss with the correct reference) <= `ref_gap_tol`, i.e. the reference
                       is not helping RECONSTRUCTION. NOTE this is ambiguous on leaked-target
                       tasks (the noised target can carry the controlled attribute itself):
                       it flags "run the swap eval", it does not prove the reference unused.

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
            # Deliberately hedged: a non-positive gap means the reference is not helping
            # RECONSTRUCTION. On leaked-target tasks (e.g. identity, where the noised target
            # carries the answer) that is AMBIGUOUS — the 2026-06 identity runs measured
            # negative gaps on a model that visibly responds to references at generation time
            # (a load-bearing reference can pay a reconstruction penalty by pulling toward a
            # generic rendition of the shared attribute). Only a generation-from-noise swap
            # eval can tell "unused" from "load-bearing but reconstruction-penalized".
            s.messages.append(
                f"reference-attribution gap {self._last_ref_gap:+.4f} <= {self.ref_gap_tol} — the "
                "reference is not helping reconstruction. AMBIGUOUS on leaked-target tasks "
                "(could be unused OR load-bearing-but-penalized); the generation-from-noise "
                "swap eval is the arbiter"
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


def paired_difference(
    measure: "Callable[[Any], float]",
    a: Any,
    b: Any,
    *,
    save_state: "Callable[[], Any]",
    restore_state: "Callable[[Any], None]",
) -> float:
    """``measure(b) - measure(a)`` evaluated under IDENTICAL RNG, so a *stochastic* measure's
    randomness cancels and the difference reflects only the ``a -> b`` change.

    This is the load-bearing trick behind the reference-attribution gap (loss with a wrong
    reference minus loss with the correct one): each forward samples a fresh timestep + noise,
    so without pinning the RNG the gap would be dominated by sampling variance, not the
    reference. We snapshot the RNG before ``a``, run it, restore the snapshot, then run ``b`` —
    both see the same timestep and noise, so a non-zero result is the reference's doing.
    """
    state = save_state()
    ma = measure(a)
    restore_state(state)
    mb = measure(b)
    return mb - ma


def summarize_gaps(gaps: "Sequence[float]") -> dict[str, float | int]:
    """Mean + spread for a set of per-pair reference-attribution gaps.

    The 2026-06 identity runs emitted bare per-sigma means (n≈23, magnitudes ~1% of the
    loss) that later could not be distinguished from noise — the curve sparked a real
    "is the reference load-bearing?" debate that the data couldn't settle. This makes
    every future curve carry its own yardstick: a normal-approximation 95% CI
    (``mean ± 1.96·sd/√n``). n is small in practice, so treat the interval as a noise
    yardstick ("is this gap distinguishable from 0?"), not exact inference.

    Degenerate inputs return NaN rather than something fake: n=0 → all NaN; n=1 → mean
    is real but std/CI are NaN (a single batch must not masquerade as a tight measurement).
    """
    n = len(gaps)
    nan = float("nan")
    if n == 0:
        return {"mean": nan, "std": nan, "n": 0, "ci95_lo": nan, "ci95_hi": nan}
    mean = sum(gaps) / n
    if n == 1:
        return {"mean": mean, "std": nan, "n": 1, "ci95_lo": nan, "ci95_hi": nan}
    std = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1))  # sample std
    half = 1.96 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "n": n, "ci95_lo": mean - half, "ci95_hi": mean + half}


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


def emit_if_fresh(metrics: dict[str, Any], key: str, value: float | None) -> dict[str, Any]:
    """Merge a periodically-sampled scalar into a metrics dict ONLY when it was freshly measured
    this step (``value is not None``), returning a new dict.

    Forward-filling the last measured value onto every step would paint a fake continuous curve in
    the JSONL / W&B for a metric that is actually sampled at a coarser cadence (e.g. the
    reference-attribution gap, measured only at the checkpoint interval) — the exact "looks like a
    signal but isn't" trap. Omitting it on the in-between steps keeps the curve sparse and honest.
    Returns ``metrics`` unchanged when ``value`` is None.
    """
    if value is not None:
        return {**metrics, key: value}
    return metrics

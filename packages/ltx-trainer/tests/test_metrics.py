"""Unit tests for training telemetry: EMA smoothing, the convergence/overfit/
underfit monitor, and the JSONL metrics writer. All pure — no model, no GPU.

The monitor is the "knowing when to stop / underfit / overfit / not converging"
core: it ingests the smoothed train loss (+ optional held-out loss + the
reference-attribution gap) per eval and flags the situation. These tests feed it
synthetic curves and assert the flags fire correctly.
"""

from __future__ import annotations

import json

from ltx_trainer.metrics import (
    EMA,
    ConvergenceMonitor,
    MetricsWriter,
    build_metrics_row,
    emit_if_fresh,
    paired_difference,
)


# --- EMA ----------------------------------------------------------------------


def test_ema_tracks_and_smooths():
    ema = EMA(beta=0.5)
    assert ema.update(1.0) == 1.0  # first value seeds it
    assert ema.update(0.0) == 0.5  # 0.5*1 + 0.5*0
    assert ema.update(0.0) == 0.25
    assert ema.value == 0.25


# --- ConvergenceMonitor -------------------------------------------------------


def _feed(monitor, train_curve, heldout_curve=None, ref_curve=None):
    status = None
    for i, tl in enumerate(train_curve):
        hl = heldout_curve[i] if heldout_curve is not None else None
        rg = ref_curve[i] if ref_curve is not None else None
        status = monitor.update(step=(i + 1) * 100, train_loss=tl, heldout_loss=hl, ref_gap=rg)
    return status


def test_healthy_decrease_no_flags():
    m = ConvergenceMonitor(warmup_evals=2, plateau_window=4)
    s = _feed(m, [1.0, 0.8, 0.65, 0.55, 0.48, 0.43])
    assert not s.not_learning
    assert not s.plateaued
    assert not s.converged


def test_not_learning_when_train_loss_flat_after_warmup():
    m = ConvergenceMonitor(warmup_evals=2, min_rel_drop=0.05)
    s = _feed(m, [1.0, 1.0, 0.99, 1.0, 0.995, 0.99])  # never drops ~5%
    assert s.not_learning
    assert any("not" in msg.lower() or "underfit" in msg.lower() for msg in s.messages)


def test_plateau_detected_when_train_loss_stops_moving():
    m = ConvergenceMonitor(warmup_evals=2, plateau_tol=0.01, plateau_window=4)
    s = _feed(m, [1.0, 0.6, 0.4, 0.40, 0.399, 0.400, 0.3995])  # dropped then flat
    assert s.plateaued


def test_overfitting_when_heldout_rises_while_train_falls():
    m = ConvergenceMonitor(warmup_evals=1, overfit_patience=3)
    s = _feed(
        m,
        train_curve=[1.0, 0.7, 0.5, 0.4, 0.33, 0.28],
        heldout_curve=[1.1, 0.9, 0.85, 0.9, 0.97, 1.05],  # bottoms out then rises
    )
    assert s.overfitting
    assert s.should_stop
    assert any("overfit" in msg.lower() for msg in s.messages)


def test_converged_when_both_plateau():
    m = ConvergenceMonitor(warmup_evals=1, plateau_tol=0.01, plateau_window=3, overfit_patience=3)
    s = _feed(
        m,
        train_curve=[1.0, 0.5, 0.41, 0.405, 0.402, 0.401],
        heldout_curve=[1.0, 0.6, 0.52, 0.519, 0.520, 0.519],
    )
    assert s.converged
    assert s.should_stop
    assert not s.overfitting


def test_reference_decorative_when_gap_not_positive():
    m = ConvergenceMonitor(warmup_evals=1, ref_gap_tol=0.0)
    # ref_gap = loss(shuffled ref) - loss(correct ref); <=0 means the model ignores the ref
    s = _feed(m, train_curve=[1.0, 0.7, 0.55], ref_curve=[0.0, -0.01, 0.0])
    assert s.ref_decorative
    assert any("reference" in msg.lower() for msg in s.messages)


def test_reference_load_bearing_when_gap_positive():
    m = ConvergenceMonitor(warmup_evals=1, ref_gap_tol=0.0)
    s = _feed(m, train_curve=[1.0, 0.7, 0.55], ref_curve=[0.05, 0.08, 0.10])
    assert not s.ref_decorative


# --- MetricsWriter (JSONL) ----------------------------------------------------


def test_metrics_writer_appends_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    w = MetricsWriter(path)
    w.write({"step": 1, "loss": 0.5, "grad_norm": 1.2})
    w.write({"step": 2, "loss": 0.4, "grad_norm": 0.9})
    w.close()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["step"] == 1 and rows[1]["loss"] == 0.4
    assert rows[0]["grad_norm"] == 1.2


def test_metrics_writer_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.jsonl"
    w = MetricsWriter(path)
    w.write({"step": 1})
    w.close()
    assert path.is_file()


# --- build_metrics_row --------------------------------------------------------


def test_build_metrics_row_core_fields():
    row = build_metrics_row(step=10, loss=0.5, ema_loss=0.6, lr=2e-4, step_time=1.2, grad_norm=3.4)
    assert row == {
        "step": 10,
        "loss": 0.5,
        "ema_loss": 0.6,
        "grad_norm": 3.4,
        "lr": 2e-4,
        "step_time": 1.2,
    }


def test_build_metrics_row_grad_norm_null_when_absent():
    # clipping off -> grad_norm column still present, as null (stable, greppable schema)
    row = build_metrics_row(step=1, loss=0.5, ema_loss=0.5, lr=1e-4, step_time=1.0)
    assert "grad_norm" in row and row["grad_norm"] is None


def test_build_metrics_row_merges_extra():
    row = build_metrics_row(
        step=1, loss=0.5, ema_loss=0.5, lr=1e-4, step_time=1.0, extra={"train/sigma_bucket_0": 0.3}
    )
    assert row["train/sigma_bucket_0"] == 0.3
    assert row["step"] == 1  # core fields still present alongside the merged extras


# --- emit_if_fresh (honest sparse emission of periodically-sampled metrics) ----


def test_emit_if_fresh_skips_stale():
    # ref_gap is measured only at the checkpoint cadence; on the steps in between the value is
    # None and must be OMITTED, not forward-filled (a forward-filled flat line is a fake signal).
    assert emit_if_fresh({"loss": 0.5}, "ref_gap", None) == {"loss": 0.5}


def test_emit_if_fresh_includes_measured():
    assert emit_if_fresh({"loss": 0.5}, "ref_gap", 0.03) == {"loss": 0.5, "ref_gap": 0.03}


def test_emit_if_fresh_does_not_mutate_input():
    base = {"loss": 0.5}
    emit_if_fresh(base, "ref_gap", 0.03)
    assert "ref_gap" not in base  # returns a new dict, leaves the caller's dict untouched


# --- paired_difference (the reference-attribution-gap noise-pairing trick) -----


def test_paired_difference_cancels_shared_randomness():
    # A stochastic measure: depends on the arg AND an advancing "RNG" counter. Pairing the
    # state across the two calls must cancel the stochastic term, leaving only the arg diff.
    state = {"v": 0}

    def measure(x):
        state["v"] += 1  # stochastic component
        return x * 10 + state["v"]

    gap = paired_difference(
        measure, 2.0, 5.0, save_state=lambda: state["v"], restore_state=lambda s: state.update(v=s)
    )
    assert gap == (5.0 * 10) - (2.0 * 10)  # the +state term cancels exactly


def test_paired_difference_with_torch_rng():
    # Same trick with the real torch RNG state the trainer uses for the ref-gap forwards.
    import torch

    def measure(x):
        return x + torch.randn(()).item()

    gap = paired_difference(
        measure, 1.0, 3.0, save_state=torch.get_rng_state, restore_state=torch.set_rng_state
    )
    assert abs(gap - 2.0) < 1e-6  # identical noise cancels; gap = pure input difference

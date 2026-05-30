"""Unit tests for the training progress info formatter — the pure string assembly
behind the live progress bar's "Loss | LR | s/step | |g|" readout. Pure (no Rich,
no model), so it tests without a display."""

from __future__ import annotations

from ltx_trainer.progress import _format_training_info


def test_format_info_basic_fields():
    s = _format_training_info(loss=0.5, lr=2e-4, step_time=1.2)
    assert "Loss: 0.5000" in s
    assert "LR: 2.00e-04" in s
    assert "1.20s/step" in s


def test_format_info_includes_grad_norm_when_present():
    s = _format_training_info(loss=0.5, lr=2e-4, step_time=1.2, grad_norm=3.4)
    assert "3.40" in s
    assert "|g|" in s


def test_format_info_omits_grad_norm_when_none():
    s = _format_training_info(loss=0.5, lr=2e-4, step_time=1.2, grad_norm=None)
    assert "|g|" not in s

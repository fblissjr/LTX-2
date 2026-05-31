"""Held-out validation-loss config: the fields that turn on the overfitting detector
(a periodic forward+loss on a disjoint precomputed set, logged alongside train metrics)."""
import pytest
from pydantic import ValidationError

from ltx_trainer.config import ValidationConfig


def test_holdout_fields_default_off():
    c = ValidationConfig()
    assert c.holdout_data_root is None
    assert c.holdout_interval == 0
    assert c.holdout_max_batches == 0


def test_holdout_fields_settable():
    c = ValidationConfig(holdout_data_root="/data/val", holdout_interval=50, holdout_max_batches=8)
    assert c.holdout_data_root == "/data/val"
    assert c.holdout_interval == 50
    assert c.holdout_max_batches == 8


def test_holdout_interval_non_negative():
    with pytest.raises(ValidationError):
        ValidationConfig(holdout_interval=-1)

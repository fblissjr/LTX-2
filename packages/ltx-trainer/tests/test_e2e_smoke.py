"""Tests for the e2e smoke's FACT-based gates + config override (the runner's
pure, testable core). The GPU stages (process_dataset/train) are real subprocess
invocations exercised by running the runner; here we lock the gate logic that
decides pass/fail and the config-override that step-caps a real base config.
"""

from __future__ import annotations

import pytest

from ltx_trainer.data_validation import DatasetReport
from ltx_trainer.e2e_smoke import (
    SmokeGateError,
    gate_checkpoint,
    gate_sources_present,
    gate_validation,
    make_smoke_train_config,
)


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_gate_sources_present_ok(tmp_path):
    for s in ["latents", "conditions", "reference_latents", "audio_latents"]:
        _touch(tmp_path / s / "f.pt")
    gate_sources_present(tmp_path)  # no raise


def test_gate_sources_present_missing_audio_raises(tmp_path):
    for s in ["latents", "conditions", "reference_latents"]:
        _touch(tmp_path / s / "f.pt")
    (tmp_path / "audio_latents").mkdir()  # exists but empty
    with pytest.raises(SmokeGateError, match="audio_latents"):
        gate_sources_present(tmp_path)


def test_gate_validation_blocks_on_errors():
    bad = DatasetReport(errors=["source file counts differ"])
    with pytest.raises(SmokeGateError, match="do not train"):
        gate_validation(bad)
    gate_validation(DatasetReport())  # ok (no errors) → no raise


def test_gate_checkpoint(tmp_path):
    with pytest.raises(SmokeGateError, match="no checkpoint"):
        gate_checkpoint(tmp_path)
    _touch(tmp_path / "run" / "adapter.safetensors")
    assert gate_checkpoint(tmp_path).name == "adapter.safetensors"


def test_make_smoke_train_config_overrides():
    base = {
        "model": {"model_path": "x"},
        "data": {"preprocessed_data_root": "OLD"},
        "training": {"steps": 5000, "batch_size": 1},
        "training_strategy": {"name": "video_to_video"},
        "output_dir": "OLD",
    }
    cfg = make_smoke_train_config(base, preprocessed_root="NEW", output_dir="OUT", steps=3)
    assert cfg["data"]["preprocessed_data_root"] == "NEW"
    assert cfg["training"]["steps"] == 3
    assert cfg["training"]["batch_size"] == 1  # preserved
    assert cfg["output_dir"] == "OUT"
    assert cfg["training_strategy"]["with_audio"] is True
    assert cfg["training_strategy"]["audio_mode"] == "condition"
    assert cfg["model"]["model_path"] == "x"  # untouched

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
    synthetic_dataset_bucket,
)


def test_synthetic_dataset_bucket_is_frames_not_fps():
    """process_dataset buckets by FRAME COUNT; the bucket must be WxHx<frames>,
    derived from fps*duration (snapped to 8k+1), NOT the fps value."""
    # 25fps * 3s = 75 -> snap to 73 (8k+1). Must NOT be the fps (25).
    assert synthetic_dataset_bucket(256, 256, 25, 3.0) == "256x256x73"
    # frame count is 8k+1
    w, h, n = synthetic_dataset_bucket(320, 192, 24, 4.0).split("x")
    assert (int(n) - 1) % 8 == 0 and (w, h) == ("320", "192")


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


def test_generate_smoke_dataset_uses_paired_refs(tmp_path):
    """The smoke must generate ref != target data (paired references), so the
    integration gate exercises the real reference-encoding path and never
    green-lights the ref==target shortcut that broke the first coupling run."""
    import json

    from ltx_trainer.e2e_smoke import generate_smoke_dataset

    captions_path = generate_smoke_dataset(tmp_path, 4, fps=25, width=64, height=64, duration_s=1.0)
    rows = json.loads(captions_path.read_text())
    assert len(rows) == 4
    assert all("reference" in r for r in rows), rows
    assert all(r["reference"] != r["video"] for r in rows), (
        f"smoke generated ref==target (the failure mode): {rows}"
    )


def test_make_smoke_train_config_overrides():
    base = {
        "model": {"model_path": "x", "training_mode": "lora"},
        "data": {"preprocessed_data_root": "OLD"},
        "optimization": {"steps": 5000, "batch_size": 1},  # steps lives under optimization, not training
        "training_strategy": {"name": "text_to_video", "first_frame_conditioning_p": 0.1},  # base may be t2v
        "output_dir": "OLD",
        "checkpoints": {"interval": 250},
    }
    cfg = make_smoke_train_config(base, preprocessed_root="NEW", output_dir="OUT", steps=3,
                                  model_path="CKPT", text_encoder_path="GEMMA")
    assert cfg["data"]["preprocessed_data_root"] == "NEW"
    assert cfg["optimization"]["steps"] == 3
    assert cfg["optimization"]["batch_size"] == 1  # preserved
    assert cfg["output_dir"] == "OUT"
    assert cfg["training_strategy"]["name"] == "video_to_video"  # FORCED (was text_to_video)
    assert cfg["training_strategy"]["with_audio"] is True
    assert cfg["training_strategy"]["audio_mode"] == "condition"
    assert cfg["model"]["model_path"] == "CKPT"  # injected
    assert cfg["model"]["text_encoder_path"] == "GEMMA"
    assert cfg["model"]["training_mode"] == "lora"  # preserved
    assert cfg["checkpoints"]["interval"] == 3  # capped so the smoke writes a checkpoint

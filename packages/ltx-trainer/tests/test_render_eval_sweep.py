"""Unit tests for the GPU-free half of the audio-reference eval sweep.

The render itself needs the 22B on a GPU, but the sweep-config construction and reference collection
are pure and must be correct: every arm has to carry the lipdub-negative reference geometry (or the
eval silently mismatches the training convention), and the (ref, seed) grid has to be exact.
"""

import importlib.util
from pathlib import Path

import pytest

from ltx_trainer.config import ReferenceConditionConfig

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_eval_sweep.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("render_eval_sweep", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_config_is_refs_x_seeds_all_lipdub_negative():
    mod = _load_script()
    refs = [Path("a.wav"), Path("b.wav"), Path("c.wav")]
    seeds = [1, 2]
    config = mod.build_sweep_config(
        prompt="neutral caption",
        negative_prompt="bad",
        references=refs,
        seeds=seeds,
        width=512,
        height=512,
        num_frames=81,
        frame_rate=25.0,
        inference_steps=30,
        guidance_scale=4.0,
        stg_scale=1.0,
        stg_blocks=[29],
        stg_mode="stg_av",
    )
    # Full grid, audio-only.
    assert len(config.samples) == len(refs) * len(seeds)
    assert config.generate_audio is True
    assert config.generate_video is False

    # EVERY arm must pin lipdub-negative (a target_frame arm would mismatch a lipdub-trained LoRA).
    for sample in config.samples:
        assert len(sample.conditions) == 1
        cond = sample.conditions[0]
        assert isinstance(cond, ReferenceConditionConfig)
        assert cond.audio is not None
        assert cond.audio_positions_mode == "lipdub_negative"

    # Grid order is refs-outer, seeds-inner (the manifest arm map relies on this).
    got = [(c.audio, s.seed) for s in config.samples for c in s.conditions]
    expected = [(str(ref), seed) for ref in refs for seed in seeds]
    assert got == expected


def test_collect_references_expands_dirs_and_dedupes(tmp_path):
    mod = _load_script()
    (tmp_path / "one.wav").touch()
    (tmp_path / "two.mp3").touch()
    (tmp_path / "notes.txt").touch()  # ignored — not an audio extension
    explicit = tmp_path / "one.wav"  # also passed explicitly -> must dedupe

    refs = mod._collect_references([str(tmp_path), str(explicit)])
    names = sorted(r.name for r in refs)
    assert names == ["one.wav", "two.mp3"]  # txt excluded, no duplicate of one.wav


def test_collect_references_rejects_missing_path():
    mod = _load_script()
    with pytest.raises(FileNotFoundError):
        mod._collect_references(["/no/such/path.wav"])

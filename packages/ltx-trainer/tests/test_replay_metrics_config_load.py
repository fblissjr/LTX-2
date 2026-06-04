"""The replay script must load the trainer's own saved training_config.yaml, which pyyaml
dumps with python-object tags (e.g. ``video_dims: !!python/tuple``) that safe_load rejects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from replay_metrics_to_wandb import load_config_yaml  # noqa: E402


def test_loads_trainer_dumped_yaml_with_python_tuple(tmp_path):
    p = tmp_path / "training_config.yaml"
    p.write_text("validation:\n  video_dims: !!python/tuple\n  - 512\n  - 512\n  - 121\nseed: 42\n")
    cfg = load_config_yaml(p)
    assert cfg["seed"] == 42
    assert list(cfg["validation"]["video_dims"]) == [512, 512, 121]

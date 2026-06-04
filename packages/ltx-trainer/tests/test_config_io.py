"""The training_config.yaml round-trip: the trainer dumps it, tools (the W&B replay script)
reload it. Three contracts: (1) the dump is plain safe_load-able YAML — no python-object tags,
tuples become lists; (2) the loader tolerates the ``!!python/tuple`` tag that pre-fix dumps
already on disk carry; (3) the loader still rejects arbitrary-object tags (the security
boundary of the SafeLoader subclass)."""

import pytest
import yaml

from ltx_trainer.config_io import dump_config_yaml, load_config_yaml


def test_dump_is_safe_loadable_and_tuple_free(tmp_path):
    p = tmp_path / "training_config.yaml"
    with open(p, "w") as f:
        dump_config_yaml({"validation": {"video_dims": (512, 512, 121)}, "seed": 42}, f)
    text = p.read_text()
    assert "!!python" not in text
    cfg = yaml.safe_load(text)
    assert cfg["seed"] == 42
    assert cfg["validation"]["video_dims"] == [512, 512, 121]


def test_load_tolerates_python_tuple_tag_from_historical_dumps(tmp_path):
    p = tmp_path / "training_config.yaml"
    p.write_text("validation:\n  video_dims: !!python/tuple\n  - 512\n  - 512\n  - 121\nseed: 42\n")
    cfg = load_config_yaml(p)
    assert cfg["seed"] == 42
    assert list(cfg["validation"]["video_dims"]) == [512, 512, 121]


def test_load_rejects_arbitrary_python_object_tags(tmp_path):
    p = tmp_path / "training_config.yaml"
    p.write_text("x: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config_yaml(p)

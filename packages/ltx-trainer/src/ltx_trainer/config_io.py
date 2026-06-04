"""The training_config.yaml round-trip: the trainer dumps it (`Trainer._save_config`), tools
reload it (the W&B replay script's ``--config``).

Why this module exists: the dump used to go through plain ``yaml.dump``, whose default Dumper
serializes Python tuples (e.g. the config's ``video_dims``) as ``!!python/tuple`` — a tag
``yaml.safe_load`` rejects, so the trainer's own primary artifact could not be re-read safely.
``dump_config_yaml`` uses the safe dumper (tuples become plain lists; Pydantic coerces them back
to tuples on validation), and ``load_config_yaml`` stays tolerant of the ``!!python/tuple`` tag
so dumps written before the fix remain loadable.

Security note: the loader is a SafeLoader subclass whose ONLY extension is ``python/tuple``,
constructed from an already-safe sequence — ``!!python/object`` / ``!!python/name`` etc. remain
rejected, so no arbitrary object construction or code execution is possible.
"""

from pathlib import Path
from typing import IO, Any

import yaml


class _TupleTolerantSafeLoader(yaml.SafeLoader):
    """SafeLoader plus the single ``python/tuple`` extension (see module docstring)."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))  # type: ignore[arg-type]  # tag only appears on sequence nodes


_TupleTolerantSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)


def dump_config_yaml(config: dict[str, Any], stream: IO[str]) -> None:
    """Write a config dict as plain, ``safe_load``-able YAML (no python-object tags)."""
    yaml.safe_dump(config, stream, default_flow_style=False, indent=2)


def load_config_yaml(path: str | Path) -> dict:
    """Load a training_config.yaml as dumped by the trainer (current or pre-fix)."""
    return yaml.load(Path(path).read_text(), Loader=_TupleTolerantSafeLoader)

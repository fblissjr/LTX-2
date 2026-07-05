"""Import-smoke for the shipped CLI scripts.

Scripts are not unit-tested, so a broken import (a renamed symbol, a module that moved in an
upstream sync, a reference to retired code) would ship silently and only surface when a user runs
the tool. This locks that every script under scripts/ imports cleanly — cheap insurance that also
guards against a future sync breaking one of the ported audio-reference tools.

Scripts must guard execution behind ``if __name__ == "__main__"`` so importing does not run them.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

# The audio-reference / eval tooling ported onto the unified-strategy base. Scoped to these
# (not a glob of scripts/) so the guard covers our ported tools without asserting on upstream
# scripts that use intra-scripts sibling imports (only import-clean when run from scripts/).
_SCRIPTS = [
    "check_reference_encode_parity.py",
    "generate_synthetic_av_data.py",
    "precompute_reference_audio.py",
    "probe_vae_temporal_aliasing.py",
    "render_eval_sweep.py",
    "replay_metrics_to_wandb.py",
    "run_audio_coupling_eval.py",
    "run_e2e_smoke.py",
    "verify_codecs.py",
    "verify_training_data.py",
]


@pytest.mark.parametrize("script_name", _SCRIPTS)
def test_script_imports_clean(script_name: str):
    path = _SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # raises on any import error / module-level failure

"""Helpers for the scripted render/eval lane (scripts/render_eval_sweep.py).

Small, file-IO-adjacent utilities that the validation sampler itself should not own
(the sampler stays pure tensor plumbing) but that eval scripts share.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def load_audio_latents_file(path: str | Path, variant: int = 0) -> Tensor:
    """Load audio-VAE latents ``[C, T, F]`` from a precompute ``.pt`` file.

    Accepts the formats the precompute pipeline writes, so eval sweeps can point at the
    exact files training consumed:

    - ``{latents: [C, T, F], ...}`` — a plain target/reference latent (``process_videos`` /
      ``precompute_reference_audio`` output).
    - ``{latents: [K, C, T, F], variants: K}`` — a K-variant reference stack
      (``variant`` selects which; training picks one at random per load, an eval must be
      deterministic about it).
    - A bare ``[C, T, F]`` tensor.
    """
    # weights_only=True matches PrecomputedDataset's loading of the same files (tensors +
    # primitives only; refuses pickled code).
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    latents = obj["latents"] if isinstance(obj, dict) else obj
    if latents.dim() == 4:  # [K, C, T, F] variant stack
        if not 0 <= variant < latents.shape[0]:
            raise ValueError(f"variant {variant} out of range for a {latents.shape[0]}-variant stack in {path}")
        latents = latents[variant]
    if latents.dim() != 3:
        raise ValueError(f"expected audio latents of shape [C, T, F] (or a [K, C, T, F] stack), got {tuple(latents.shape)} in {path}")
    return latents

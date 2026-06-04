"""Unit tests for timestep samplers — currently the high-sigma mixture sampler.

The mixture sampler is the training-side leak fix for reference-conditioned tasks: bias
steps toward high sigma (where the noised target is destroyed and the reference must do
the work) WITHOUT abandoning low sigma entirely — the LoRA still applies at every sigma
at inference, so a hard floor (achievable via ``uniform`` + ``min_value``) would train a
band the model never saw. Soft mixture: with probability ``band_prob`` draw from
``uniform[band_min, 1]``, else ``uniform[0, 1]``.
"""

from __future__ import annotations

import pytest
import torch

from ltx_trainer.timestep_samplers import SAMPLERS, HighSigmaMixtureTimestepSampler


def _samples(sampler, n: int = 4000, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return sampler.sample(n)


def test_band_prob_one_confines_to_the_band():
    s = _samples(HighSigmaMixtureTimestepSampler(band_min=0.6, band_prob=1.0))
    assert (s >= 0.6).all() and (s <= 1.0).all()


def test_band_prob_zero_is_plain_uniform():
    s = _samples(HighSigmaMixtureTimestepSampler(band_min=0.6, band_prob=0.0))
    assert (s >= 0.0).all() and (s <= 1.0).all()
    # roughly uniform: the high band holds ~ its measure, not more
    in_band = (s >= 0.6).float().mean().item()
    assert in_band == pytest.approx(0.4, abs=0.05)


def test_mixture_actually_biases_toward_high_sigma():
    # band_prob=0.5, band [0.6, 1]: expected in-band mass = 0.5 + 0.5*0.4 = 0.7
    s = _samples(HighSigmaMixtureTimestepSampler(band_min=0.6, band_prob=0.5))
    in_band = (s >= 0.6).float().mean().item()
    assert in_band == pytest.approx(0.7, abs=0.05)
    # and the low band keeps real coverage (the soft part of "soft bias")
    assert (s < 0.6).float().mean().item() > 0.2


def test_sample_for_matches_batch_shape_and_device():
    sampler = HighSigmaMixtureTimestepSampler()
    batch = torch.zeros(3, 10, 128)
    torch.manual_seed(0)
    out = sampler.sample_for(batch)
    assert out.shape == (3,)
    assert out.device == batch.device
    with pytest.raises(ValueError, match="3 dimensions"):
        sampler.sample_for(torch.zeros(3, 10))


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        HighSigmaMixtureTimestepSampler(band_min=1.5)
    with pytest.raises(ValueError):
        HighSigmaMixtureTimestepSampler(band_prob=-0.1)


def test_registered_in_samplers_and_config_mode():
    """The sampler must be reachable from config: SAMPLERS registry + the
    FlowMatchingConfig mode literal (the trainer does SAMPLERS[mode](**params))."""
    assert SAMPLERS["high_sigma_mixture"] is HighSigmaMixtureTimestepSampler

    from ltx_trainer.config import FlowMatchingConfig

    cfg = FlowMatchingConfig(
        timestep_sampling_mode="high_sigma_mixture",
        timestep_sampling_params={"band_min": 0.7, "band_prob": 0.6},
    )
    sampler = SAMPLERS[cfg.timestep_sampling_mode](**cfg.timestep_sampling_params)
    assert sampler.band_min == 0.7

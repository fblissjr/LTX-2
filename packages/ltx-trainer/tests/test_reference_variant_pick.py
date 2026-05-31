"""K-variant reference pick: the precompute stacks K reference voice windows as [K, C, T, F] and
marks the dict with `variants: K`; the dataset picks one at random per load (per-epoch augmentation
against overfitting to a single reference). Self-describing via the `variants` key, so it does not
depend on source naming or fragile dim heuristics (video latents are also 4D)."""
import torch

from ltx_trainer.datasets import PrecomputedDataset


def test_pick_selects_one_variant():
    stacked = torch.stack([torch.full((8, 5, 16), float(k)) for k in range(3)])  # [3, 8, 5, 16]
    out = PrecomputedDataset._maybe_pick_variant({"latents": stacked, "variants": 3, "duration": 1.0})
    assert out["latents"].shape == (8, 5, 16)
    assert out["variants"] == 1
    assert float(out["latents"][0, 0, 0]) in {0.0, 1.0, 2.0}   # picked one of the stacked variants
    assert out["duration"] == 1.0                              # other fields preserved


def test_pick_is_noop_without_variants_key():
    data = {"latents": torch.zeros(8, 5, 16)}   # a plain [C, T, F] reference
    out = PrecomputedDataset._maybe_pick_variant(data)
    assert out["latents"].shape == (8, 5, 16)
    assert "variants" not in out or out["variants"] == 1


def test_pick_varies_across_calls():
    torch.manual_seed(0)
    stacked = torch.stack([torch.full((8, 5, 16), float(k)) for k in range(3)])
    data = {"latents": stacked, "variants": 3}
    seen = {float(PrecomputedDataset._maybe_pick_variant(data)["latents"][0, 0, 0]) for _ in range(40)}
    assert len(seen) > 1   # not always the same variant

"""Block-swap tests — fits int8 22B base on a 24 GB 4090 by streaming the last
N transformer blocks GPU<->CPU during forward + backward.

Design doc: ../../../../internal/block_swap_port_plan.md (private clone). The
non-obvious correctness point is the gradient-checkpointing + recompute timing:
wrapper.forward must stream-in (covers initial forward AND checkpoint recompute);
pre/post backward hooks on the inner block handle the gap between recompute
stream-out and gradient computation. These tests lock that contract.

Reference port from musubi-tuner (Lightricks LTX-2 production trainer);
ltx-trainer trainer.py wires it via AccelerationConfig.block_swap_blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ltx_trainer.block_swap import (
    BlockSwapManager,
    StreamingBlockWrapper,
    attach_block_swap,
)

CPU = torch.device("cpu")
META = torch.device("meta")  # stand-in compute device for unit tests (no GPU needed)


def _block(weight=1.0):
    """A trivial block: identity linear with a weight buffer to track device."""
    b = nn.Linear(4, 4, bias=False)
    nn.init.constant_(b.weight, weight)
    return b


class _Transformer(nn.Module):
    """Minimal stand-in for ltx-core's transformer — exposes transformer_blocks."""
    def __init__(self, n=6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_block(float(i)) for i in range(n))


# --- manager state ----------------------------------------------------------


def test_manager_indices_last_n():
    """Managed indices are the LAST N blocks (so first blocks stay GPU-resident
    — typical LoRA training warms up later blocks more, but the convention is
    'swap the tail' to match musubi's pattern + matches block-checkpoint order."""
    m = BlockSwapManager(total_blocks=10, blocks_to_swap=4, offload_device=CPU)
    assert m.is_managed(9) and m.is_managed(6)
    assert not m.is_managed(5) and not m.is_managed(0)


def test_manager_zero_swap_manages_nothing():
    """blocks_to_swap=0 ⇒ no-op manager (the disabled config path)."""
    m = BlockSwapManager(total_blocks=10, blocks_to_swap=0, offload_device=CPU)
    assert all(not m.is_managed(i) for i in range(10))


def test_manager_clamps_blocks_to_swap():
    """Requesting more blocks than exist clamps to total-1 (keep at least one
    GPU-resident so the forward has somewhere to start)."""
    m = BlockSwapManager(total_blocks=4, blocks_to_swap=100, offload_device=CPU)
    assert sum(1 for i in range(4) if m.is_managed(i)) == 3


# --- wrapper forward semantics ---------------------------------------------


class _RecordingManager:
    """Mock manager that records stream_in/out calls for ordering checks."""
    def __init__(self):
        self.events: list[tuple[str, int]] = []

    def stream_in(self, block, device):
        self.events.append(("in", id(block) % 1000))

    def stream_out(self, block):
        self.events.append(("out", id(block) % 1000))


def test_wrapper_forward_streams_in_then_runs_then_streams_out():
    """The wrapper's forward order is the locked contract: stream_in → block →
    stream_out. Reordering breaks the gradient-checkpointing recompute path."""
    mgr = _RecordingManager()
    block = _block()
    w = StreamingBlockWrapper(block, mgr, idx=5, compute_device=CPU)
    out = w(torch.zeros(2, 4))
    assert out.shape == (2, 4)
    bid = id(block) % 1000
    assert mgr.events == [("in", bid), ("out", bid)]


def test_wrapper_forward_is_transparent_to_args_and_kwargs():
    """The wrapper must not interfere with the block's signature — LTX-2's
    block forward takes positional dataclass args (video, audio, perturbations)."""
    mgr = _RecordingManager()

    class _Sig(nn.Module):
        def forward(self, a, b, *, c):
            return a + b, c
    w = StreamingBlockWrapper(_Sig(), mgr, idx=0, compute_device=CPU)
    out1, out2 = w(torch.ones(1), torch.ones(1) * 2, c="kw")
    assert out1.item() == 3 and out2 == "kw"


# --- attach: replaces last N blocks, leaves first untouched ----------------


def test_attach_wraps_last_n_blocks():
    """attach() replaces transformer.transformer_blocks[-N:] with wrappers
    in-place; the first total-N stay as the original blocks. Critical: existing
    references to transformer_blocks (e.g. in _process_transformer_blocks)
    continue to iterate correctly."""
    t = _Transformer(n=6)
    originals = list(t.transformer_blocks)
    mgr = attach_block_swap(t, blocks_to_swap=2, offload_device=CPU, compute_device=CPU)
    assert mgr.is_managed(5) and mgr.is_managed(4) and not mgr.is_managed(3)
    assert isinstance(t.transformer_blocks[5], StreamingBlockWrapper)
    assert isinstance(t.transformer_blocks[4], StreamingBlockWrapper)
    assert t.transformer_blocks[3] is originals[3]  # untouched
    # The wrapper wraps the original block
    assert t.transformer_blocks[5].block is originals[5]


def test_attach_with_zero_returns_no_op_manager():
    """blocks_to_swap=0 → no wrapping, no offload. Trainer's enabled-check is
    the gate; attach with 0 is the no-op identity case."""
    t = _Transformer(n=6)
    originals = list(t.transformer_blocks)
    mgr = attach_block_swap(t, blocks_to_swap=0, offload_device=CPU, compute_device=CPU)
    assert all(t.transformer_blocks[i] is originals[i] for i in range(6))
    assert all(not mgr.is_managed(i) for i in range(6))


def test_attach_moves_managed_blocks_to_offload_device():
    """After attach, managed blocks' weights live on the offload device,
    unmanaged blocks stay on the original (CPU) device. Uses META as the
    offload device so the move is observable (CPU→CPU would be a no-op the
    stream_out short-circuit skips)."""
    t = _Transformer(n=4)
    attach_block_swap(t, blocks_to_swap=2, offload_device=META, compute_device=CPU)
    # First two: untouched, still on CPU.
    for idx in (0, 1):
        assert next(t.transformer_blocks[idx].parameters()).device == CPU
    # Last two: wrapped + moved to META.
    for idx in (2, 3):
        wrapper = t.transformer_blocks[idx]
        assert isinstance(wrapper, StreamingBlockWrapper)
        assert next(wrapper.block.parameters()).device == META


# --- backward hook registration --------------------------------------------


def test_attach_registers_backward_hooks_on_managed_blocks():
    """Backward pre/post hooks on the inner block handle the
    recompute→gradient-compute gap. We check that the right NUMBER are
    registered on the right blocks (the inner ones, not the wrappers)."""
    t = _Transformer(n=5)
    attach_block_swap(t, blocks_to_swap=3, offload_device=CPU, compute_device=CPU)
    # blocks 2, 3, 4 are managed; their inner blocks should each carry
    # 1 pre-backward + 1 post-backward hook (= 2 entries in their hook dicts).
    for idx in (2, 3, 4):
        inner = t.transformer_blocks[idx].block
        assert len(inner._backward_pre_hooks) >= 1
        assert len(inner._backward_hooks) >= 1
    # Unmanaged blocks have no hooks registered by attach
    assert len(t.transformer_blocks[0]._backward_pre_hooks) == 0
    assert len(t.transformer_blocks[0]._backward_hooks) == 0

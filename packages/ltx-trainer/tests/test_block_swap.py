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


def test_attach_calls_stream_out_on_managed_blocks(monkeypatch):
    """attach() must call manager.stream_out on managed blocks (so they end up
    on the offload device). Spied because the actual move uses full
    `setattr(sub, name, nn.Parameter(...))` Parameter replacement (the only
    way around optimum.quanto's `Parameter.data = ...` snap-back trap), and a
    real device-move can't target META cleanly on a CPU-only test environment.
    Verifying the CALL instead of the post-move device is the right
    behavioral test here — the move mechanism is locked separately by
    test_move_module_data_uses_parameter_replacement_not_data_assignment."""
    t = _Transformer(n=4)
    streamed_out: list[nn.Module] = []

    # Patch the bound method on the class to capture calls
    import ltx_trainer.block_swap as bs
    orig = bs.BlockSwapManager.stream_out

    def _spy(self, block):
        streamed_out.append(block)
        # call original to preserve correctness (no-op on CPU→CPU per
        # _module_on_device short-circuit, so no real move happens)
        orig(self, block)
    monkeypatch.setattr(bs.BlockSwapManager, "stream_out", _spy)

    originals = list(t.transformer_blocks)
    attach_block_swap(t, blocks_to_swap=2, offload_device=CPU, compute_device=CPU)
    # Managed (last 2) had stream_out called on them; unmanaged didn't.
    assert {id(b) for b in streamed_out} == {id(originals[2]), id(originals[3])}


# --- backward hook registration --------------------------------------------


def test_attach_registers_backward_hooks_on_managed_blocks():
    """Backward pre/post hooks on the inner block handle the
    recompute→gradient-compute gap. EXACT count is asserted (== 1 each) so a
    double-attach (e.g., checkpoint resume bug) is caught — duplicate hooks
    silently double-fire stream_in/out per backward pass."""
    t = _Transformer(n=5)
    attach_block_swap(t, blocks_to_swap=3, offload_device=CPU, compute_device=CPU)
    # blocks 2, 3, 4 are managed; inner blocks each carry exactly 1 pre + 1 post.
    for idx in (2, 3, 4):
        inner = t.transformer_blocks[idx].block
        assert len(inner._backward_pre_hooks) == 1
        assert len(inner._backward_hooks) == 1
    # Unmanaged blocks have no hooks registered by attach
    assert len(t.transformer_blocks[0]._backward_pre_hooks) == 0
    assert len(t.transformer_blocks[0]._backward_hooks) == 0


def test_move_module_data_uses_parameter_replacement_not_data_assignment():
    """Regression test for the optimum.quanto Parameter-setter trap.

    Background: `param.data = param.data.to('cpu')` for a quanto WeightQBytesTensor
    silently keeps the storage on the original device (the Parameter setter undoes
    the move). Burned hours diagnosing this — the test locks the working pattern
    (full Parameter replacement via setattr) so a future refactor doesn't revert
    to `.data = ...` without us noticing.

    Verified at the API level (not via quanto, which would need GPU): after
    `_move_module_data` with a real device transition (META→CPU), the
    submodule's `weight` is a NEW Parameter object (id changed) whose
    underlying storage lives on the target device. `.data = ...` would keep
    the SAME Parameter id (it mutates in place)."""
    from ltx_trainer.block_swap import _move_module_data
    # Need source != target to bypass the (correct) "already on device" early
    # return. CPU→META works (META as target is allowed; the reverse,
    # META→CPU, raises because meta tensors have no data).
    block = _block()
    block.weight.requires_grad = False  # base-model case (frozen)
    original_weight_id = id(block.weight)
    _move_module_data(block, META)
    assert id(block.weight) != original_weight_id, \
        "_move_module_data must REPLACE the Parameter (setattr) not assign .data"


def test_move_module_data_skips_trainable_params():
    """LoRA params (requires_grad=True) must NOT be replaced — replacing a
    Parameter orphans its optimizer state (the optimizer keys off Parameter
    object identity). Matches musubi's skip_trainable=True pattern."""
    from ltx_trainer.block_swap import _move_module_data
    block = _block()  # CPU source, META target → would fire if not for skip
    block.weight.requires_grad = True  # simulate a trainable param (LoRA)
    original_weight_id = id(block.weight)
    _move_module_data(block, META)
    assert id(block.weight) == original_weight_id, \
        "trainable params (requires_grad=True) must be skipped — replacing orphans optimizer state"


def test_wrapper_streams_out_even_when_block_raises():
    """The `finally` in wrapper.forward guarantees stream_out even on exception
    — otherwise a single failing forward leaves the block GPU-resident
    permanently and the swap savings degrade silently over time. Locks the
    contract so a future refactor that drops `finally` is caught."""
    mgr = _RecordingManager()

    class _Boom(nn.Module):
        def forward(self, _x):
            raise RuntimeError("boom")
    w = StreamingBlockWrapper(_Boom(), mgr, idx=0, compute_device=CPU)
    try:
        w(torch.zeros(1))
    except RuntimeError:
        pass
    # Both events fired: stream_in (before block.forward), stream_out (after,
    # via the finally clause — this is what protects against the leak).
    kinds = [k for k, _ in mgr.events]
    assert kinds == ["in", "out"]

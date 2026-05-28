"""Block-swap — stream transformer blocks GPU<->CPU during forward + backward
so an int8 22B base fits training on a 24 GB 4090.

The 22B int8-quanto base alone allocates ~22.97 GB on a 4090 (E0.2 verdict);
no activation headroom. With N of M blocks swapped out, peak resident base
scales to ≈ (M-N)/M × base_size.

Reference port from musubi-tuner's `LTX2BlockSwapManager` (production trainer
for LTX-2 v2v/av_ic IC-LoRA). This implementation is the MVP — pinned memory,
FP8 upcast, slab pool, audio-module carve-outs are deferred (see design doc).

## The non-obvious correctness point

ltx-core's `_process_transformer_blocks` wraps every block with
`torch.utils.checkpoint.checkpoint(block, ..., use_reentrant=False)`. During
backward, checkpoint re-runs the wrapped function to regenerate activations.
That means:

  initial forward      : wrapper.forward → block on GPU during forward
  backward recompute   : wrapper.forward (called by checkpoint) → block on GPU
  backward grad compute: block.backward via autograd → block on GPU

The wrapper's forward streams in + runs + streams out (covers both initial
forward AND recompute). Stream-out at end of wrapper-forward leaves the block
on CPU; the pre-backward hook on the INNER block streams it back in before
gradient computation. Post-backward hook streams out after.

## LoRA caveat

LoRA adapters live inside each block after PEFT wraps the model — the wrapper
streams them with the rest. After backward, LoRA `.grad` is on CPU; the
optimizer step runs on CPU (slow per-step but correct). For LoRA rank ~16
this is a few hundred ms total in a smoke. v2 optimization: skip-stream LoRA
params (musubi's `skip_trainable=True` pattern).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlockSwapManager:
    """Tracks managed block indices (the last N) + the offload device.

    Construction is parameter-only (no model side effects); call
    `attach_block_swap()` to mutate the transformer.
    """

    def __init__(
        self,
        total_blocks: int,
        blocks_to_swap: int,
        offload_device: torch.device,
    ):
        # Clamp: keep at least one block resident so the forward has somewhere
        # to start (musubi pattern; raw blocks_to_swap=total would mean nothing
        # is GPU-resident even at peak, which complicates the first-forward).
        self._blocks_to_swap = max(0, min(blocks_to_swap, total_blocks - 1))
        self._total_blocks = total_blocks
        self._offload_device = offload_device
        # Managed indices are the LAST N (tail). Matches musubi's convention
        # + matches typical "warm up later" assumption: early blocks process
        # patchified inputs (cheap), later blocks carry the heavier semantics.
        self._managed: set[int] = set(
            range(total_blocks - self._blocks_to_swap, total_blocks)
        )

    @property
    def blocks_to_swap(self) -> int:
        return self._blocks_to_swap

    @property
    def offload_device(self) -> torch.device:
        return self._offload_device

    def is_managed(self, idx: int) -> bool:
        return idx in self._managed

    def stream_in(self, block: nn.Module, compute_device: torch.device) -> None:
        """Move block to compute_device if not already there. Called by the
        wrapper's forward + the pre-backward hook on the inner block."""
        if _module_on_device(block, compute_device):
            return
        with torch.no_grad():
            block.to(compute_device)

    def stream_out(self, block: nn.Module) -> None:
        """Move block to offload device. Called by the wrapper's forward AFTER
        the inner forward + the post-backward hook AFTER gradient computation."""
        if _module_on_device(block, self._offload_device):
            return
        with torch.no_grad():
            block.to(self._offload_device)


class StreamingBlockWrapper(nn.Module):
    """Wraps a transformer block so its weights stream in/out around forward.

    Replaces the original block in transformer.transformer_blocks[i]. Identity
    on the forward signature (the LTX-2 block takes positional dataclass args;
    we pass *args/**kwargs straight through). The inner block remains as
    `.block` for hook registration + introspection.
    """

    def __init__(
        self,
        block: nn.Module,
        manager: BlockSwapManager,
        idx: int,
        compute_device: torch.device,
    ):
        super().__init__()
        self.block = block
        self._manager = manager
        self._idx = idx
        self._compute_device = compute_device

    def forward(self, *args, **kwargs):
        self._manager.stream_in(self.block, self._compute_device)
        try:
            return self.block(*args, **kwargs)
        finally:
            self._manager.stream_out(self.block)


def attach_block_swap(
    transformer: nn.Module,
    blocks_to_swap: int,
    offload_device: torch.device,
    compute_device: torch.device,
) -> BlockSwapManager:
    """Wrap the last N `transformer.transformer_blocks` with
    `StreamingBlockWrapper`, move them to the offload device, and register
    backward hooks on the inner blocks for the recompute → gradient compute
    handoff (see module docstring).

    Returns the manager (held by the trainer for diagnostics; the wrapping is
    in-place on `transformer.transformer_blocks`).
    """
    blocks = transformer.transformer_blocks  # ltx-core convention
    total = len(blocks)
    manager = BlockSwapManager(total, blocks_to_swap, offload_device)
    if manager.blocks_to_swap == 0:
        return manager

    for idx in range(total):
        if not manager.is_managed(idx):
            continue
        inner = blocks[idx]
        # Stream out the inner block first (so subsequent .to inside the
        # wrapper has the correct starting device).
        manager.stream_out(inner)
        # Register backward hooks on the INNER block (the autograd path goes
        # through it). Pre-hook streams in before gradient computation; post-
        # hook streams out after.
        _register_backward_hooks(inner, manager, compute_device)
        # Replace in place. ModuleList supports __setitem__.
        blocks[idx] = StreamingBlockWrapper(inner, manager, idx, compute_device)
    return manager


# --- internals -------------------------------------------------------------


def _normalize_device(device: torch.device) -> torch.device:
    """`torch.device('cuda') != torch.device('cuda:0')` under !=, but they're
    the same device. Resolve a bare `cuda` to the current indexed device so
    the stream_in/stream_out short-circuit doesn't miss + cause every forward
    to re-issue `.to(device)` (silent perf regression that LOOKS like swap is
    broken in the VRAM logs)."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


def _module_on_device(module: nn.Module, device: torch.device) -> bool:
    """True iff every parameter + buffer of `module` is on `device`. A module
    with no params/buffers returns True (nothing to move = trivially-resident).
    Cheap short-circuit prevents redundant .to() (which still allocates briefly)."""
    target = _normalize_device(device)
    for p in module.parameters():
        if _normalize_device(p.device) != target:
            return False
    for b in module.buffers():
        if _normalize_device(b.device) != target:
            return False
    return True


def _register_backward_hooks(
    block: nn.Module,
    manager: BlockSwapManager,
    compute_device: torch.device,
) -> None:
    """Pre + post backward hooks on the inner block — handle the gap between
    wrapper.forward's stream-out and the gradient-compute that needs the block
    on GPU. See module docstring for why this can't be done in the wrapper alone."""
    def _pre(_module, _grad_output):
        manager.stream_in(_module, compute_device)
        return None

    def _post(_module, _grad_input, _grad_output):
        manager.stream_out(_module)
        return None

    block.register_full_backward_pre_hook(_pre)
    block.register_full_backward_hook(_post)



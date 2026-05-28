"""Block-swap — stream transformer blocks GPU<->CPU during forward + backward
so a large base fits training on a 24 GB 4090.

## STATUS: INCOMPLETE for int8-quanto (the immediate target)

Despite engaging without error, this MVP moves only ~0.3 GB of a 22 GB int8
base. Root cause empirically isolated 2026-05-27 late: in optimum.quanto,
assignment to `Parameter.data` of a `WeightQBytesTensor` SILENTLY KEEPS the
internal `_data` / `_scale` on the original device. Reproducer:

    w = quantized_linear.weight    # WeightQBytesTensor on cuda
    moved = w.data.to('cpu')       # moved._data, moved._scale on CPU (good)
    w.data = moved                 # w.data.device snaps BACK to cuda:0 (bad)
    w.data._data = w.data._data.to('cpu')  # also snaps back

So `_move_module_data` below correctly produces a CPU version + assigns it,
but the Parameter setter undoes the assignment for int8-quant weights. The
`Block-swap attached: ... (VRAM A -> B)` log shows B nearly equal to A
because the bulk-storage move is being silently reverted.

Three known forward paths (none implemented here yet — needs daylight):

  1. Switch the trainer's quant to **fp8-quanto** (musubi-tuner's actual
     target — `.data = ...` reportedly works for fp8 because its tensor
     wrapping is different). Cheap config-level change once verified.
  2. **register_parameter("weight", nn.Parameter(moved_data))** to fully
     unregister + replace the Parameter. Non-trivial: QLinear has fast-paths
     that may break on re-registration; needs careful testing.
  3. Use **mmgp** (Wan2GP's library) directly — quanto-aware by construction;
     skips the Parameter-setter trap. Bigger integration cost.

## What works (the parts we keep)

The MANAGER + WRAPPER + HOOK MACHINERY is correct and survives the int8
issue. When `_move_module_data` is replaced with one of the three options
above, the rest of the port should fit together unchanged. The wrapper's
streaming contract + the backward-hook timing is exercised by 10 unit tests.

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
        _move_module_data(block, compute_device)

    def stream_out(self, block: nn.Module) -> None:
        """Move block to offload device. Called by the wrapper's forward AFTER
        the inner forward + the post-backward hook AFTER gradient computation."""
        if _module_on_device(block, self._offload_device):
            return
        _move_module_data(block, self._offload_device)


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
    # CRITICAL: empty the CUDA caching allocator. Param `.data = .to('cpu')`
    # releases the GPU storage reference, but PyTorch's allocator caches the
    # freed memory in its reserved-but-unallocated pool instead of returning
    # it to the OS. Without empty_cache, the freed VRAM isn't observable by
    # other processes/allocations and the block-swap delta looks like a no-op
    # in nvidia-smi / memory_allocated. Cheap (one call at attach time).
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()
    return manager


# --- internals -------------------------------------------------------------


_LINEAR_QUANT_ATTRS = ("weight", "bias", "scale_weight")  # quanto: weight may be a non-Parameter


def _move_module_data(module: nn.Module, device: torch.device) -> None:
    """Move every weight tensor of `module` to `device` via direct `.data`
    assignment — bypasses `nn.Module.to()` (which calls `_apply` → `swap`,
    which fails on optimum.quanto's QLinear).

    The non-obvious correctness point: after `quanto.quantize(model)`, a
    QLinear's `weight` is REPLACED with a non-`nn.Parameter` custom tensor
    type. `module.parameters()` skips it entirely — iterating params alone
    misses the bulk of an int8 model's storage (the 0.32 GB symptom on a
    22 GB base = parameters() found only the LoRA + norm + embedding params).
    Combine standard iteration with the explicit quanto attrs musubi uses.

    Without this, block-swap looks attached but `memory_allocated()` doesn't
    drop. With it, the int8 weight tensors actually move."""
    non_blocking = device.type != "cpu"
    target = torch.device(device)
    with torch.no_grad():
        # Standard path: regular Parameters + Buffers (norms, embeddings, biases).
        for p in module.parameters():
            p.data = p.data.to(device, non_blocking=non_blocking)
        for b in module.buffers():
            b.data = b.data.to(device, non_blocking=non_blocking)
        # Quanto path: explicit Linear-like attrs, catches the QLinear weight
        # that's NOT a registered Parameter post-quantization.
        for sub in module.modules():
            if not sub.__class__.__name__.endswith("Linear"):
                continue
            for attr in _LINEAR_QUANT_ATTRS:
                t = getattr(sub, attr, None)
                if t is None or not hasattr(t, "data") or not hasattr(t.data, "to"):
                    continue
                if t.data.device == target:
                    continue
                t.data = t.data.to(device, non_blocking=non_blocking)


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



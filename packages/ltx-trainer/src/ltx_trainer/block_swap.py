"""Block-swap — stream transformer blocks GPU<->CPU during forward + backward
so a large base fits training on a 24 GB 4090.

## The key correctness point (the trap that ate hours of debugging)

In optimum.quanto, `Parameter.data = w.data.to('cpu')` SILENTLY KEEPS the
underlying `_data` / `_scale` on the original device — the Parameter setter
has device-preservation logic that undoes the move. Both int8 and fp8 quanto
tensors hit this. Reproducer in repo history (`internal/audio_iclora_status.md`
2026-05-27 entry).

`_move_module_data` below uses **full Parameter replacement via setattr** to
bypass the trap. Empirically verified across three quant modes (int8, fp8,
unquantized) by direct repro: the weight ACTUALLY moves, forward succeeds.

## What this MVP intentionally defers

Pinned memory, FP8 upcast on offload, slab pool, audio-module carve-outs.
See `internal/block_swap_port_plan.md`. The wrapper + manager + hook
machinery doesn't need them; they're throughput optimizations.

Reference shape from musubi-tuner's `LTX2BlockSwapManager` (production
trainer for LTX-2 v2v/av_ic IC-LoRA); our `_move_module_data` is the
correctness fix for the quanto Parameter-setter case they don't trigger
(musubi targets fp8 with a different load path).

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

import re

import torch
import torch.nn as nn

# `attach_block_swap` replaces transformer_blocks[i] with a StreamingBlockWrapper
# holding the real block at `.block`, so swapped blocks serialize their params as
# `transformer_blocks.<N>.block.<rest>` while kept blocks stay
# `transformer_blocks.<N>.<rest>`. This regex matches that wrapper segment (with
# or without a leading `diffusion_model.` prefix). Mirrors the ComfyUI-side
# converter's pattern so the trainer can emit inference-ready keys directly.
_BLOCK_SWAP_KEY_RE = re.compile(r"(transformer_blocks\.\d+)\.block\.")


def strip_block_swap_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite ``transformer_blocks.<N>.block.<rest>`` keys to ``transformer_blocks.<N>.<rest>``.

    ComfyUI's transformer has no StreamingBlockWrapper, so the spurious ``.block.``
    segment makes those LoRA keys silently no-op at inference (the swapped blocks —
    often the majority of the adapter — would contribute nothing). Stripping it at
    save time makes the checkpoint load correctly without the external converter.
    Idempotent: keys without the segment pass through unchanged.
    """
    return {_BLOCK_SWAP_KEY_RE.sub(r"\1.", k): v for k, v in state_dict.items()}


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
        # Double-attach guard (/simplify finding #3): if the block is already
        # wrapped (re-entry through a future validation-prep or hot-reload
        # path), wrapping again would silently duplicate hooks → double-fire
        # stream_in/out per backward. Cheap structural check.
        if isinstance(inner, StreamingBlockWrapper):
            continue
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


def _move_module_data(module: nn.Module, device: torch.device) -> None:
    """Move FROZEN weight tensors of `module` to `device` via full Parameter
    REPLACEMENT (not `.data = ...`).

    The trap this avoids: for optimum.quanto's `WeightQBytesTensor` (int8 OR
    fp8), `p.data = p.data.to('cpu')` SILENTLY KEEPS the underlying
    `_data` / `_scale` on the original device. The Parameter setter has
    device-preservation logic that undoes the move. Empirically verified
    2026-05-27 by direct repro on a quantized Linear.

    What works: `setattr(submodule, 'weight', nn.Parameter(moved, ...))`
    routes through `nn.Module.__setattr__` which DOES honor a full Parameter
    replacement. Verified end-to-end: weight moves to CPU, back to GPU,
    forward through the QLinear succeeds with correct device output.

    LoRA caveat: trainable params (`requires_grad=True` — typically the
    LoRA adapters after PEFT wraps the model) are SKIPPED. Replacing a
    Parameter orphans its optimizer state (the optimizer keys off Parameter
    object identity); keeping LoRA on GPU also lets the optimizer step
    happen on GPU rather than CPU. Matches musubi's `skip_trainable` pattern.

    Buffers (non-Parameter tensors like RMSNorm's running stats) use the
    plain `.data = ...` path — they don't have the quanto wrapper trap."""
    # _normalize_device + indexed-param comparison: `torch.device('cuda')` and
    # `torch.device('cuda:0')` are NOT equal under ==, so the early-return
    # short-circuit silently misses when the compute device is bare 'cuda'.
    # /simplify finding #1 — without this, every wrapper.forward re-walks the
    # full parameter list and re-issues .to() even when the block is resident.
    target = _normalize_device(torch.device(device))
    non_blocking = device.type != "cpu"
    with torch.no_grad():
        for sub in module.modules():
            # Direct (non-recursive) parameters of this submodule. Iterating
            # at the submodule level + named_parameters(recurse=False) is
            # important so `setattr(sub, name, ...)` targets the right module.
            for name, p in list(sub.named_parameters(recurse=False)):
                if p.requires_grad:
                    continue  # LoRA / trainable — keep GPU-resident, preserve optimizer state
                if _normalize_device(p.device) == target:
                    continue
                moved = p.to(device, non_blocking=non_blocking)
                setattr(sub, name, nn.Parameter(moved, requires_grad=False))
            # Buffers move via the standard .data path.
            for name, b in list(sub.named_buffers(recurse=False)):
                if _normalize_device(b.device) == target:
                    continue
                b.data = b.data.to(device, non_blocking=non_blocking)


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

    Cost: O(params) walk on every call — acceptable for MVP since the typical
    pattern is "skip when already there." For v2 we'd cache the per-block
    device on the manager; defer until profiling shows it matters."""
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



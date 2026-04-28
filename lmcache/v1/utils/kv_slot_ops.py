# SPDX-License-Identifier: Apache-2.0
"""Shared slot-based KV tensor utilities.

This module is the single source of truth for the two ndim-aware
operations used by both the vLLM-side internal API and the
multiprocess HTTP cache API:

* :func:`extract_kv_at_slots` - gather KV values for a set of
  ``slot_indices`` out of a per-layer KV tensor, flattening the
  ``(num_blocks, block_size)`` pair into a single slot dimension.
* :func:`slice_by_slot_dim`   - slice the *result* of
  :func:`extract_kv_at_slots` along its slot dimension.

Centralising them here lets callers stay completely
shape-agnostic: future KV layouts only need to be taught to this
module once, and every consumer benefits automatically.

Supported per-layer KV tensor layouts:

* MHA 5D: ``[2, num_blocks, block_size, num_heads, head_size]``
* MLA 3D: ``[num_blocks, block_size, head_size]``
* 4D:     ``[num_blocks, block_size, num_heads, head_size]``
  (used by a few alternative paths / tests)
"""

# Third Party
import torch


def extract_kv_at_slots(
    kv_tensor: torch.Tensor, slot_tensor: torch.Tensor
) -> torch.Tensor:
    """Extract KV data at specified slot positions from kv_tensor.

    The slot_mapping contract is::

        slot_idx = block_id * block_size + block_offset

    so we can reshape ``(num_blocks, block_size)`` into a single
    slot dimension and index by ``slot_tensor`` directly.

    Args:
        kv_tensor: The KV cache tensor for a single layer.
        slot_tensor: 1D tensor of slot indices to extract.

    Returns:
        Tensor with KV data at the specified slots:

        * MHA (5D input): ``[2, num_slots, num_heads, head_size]``
        * 4D  (4D input): ``[num_slots, num_heads, head_size]``
        * MLA (3D input): ``[num_slots, head_size]``
    """
    ndim = kv_tensor.ndim

    if ndim == 5:
        # MHA-style: [KV, num_blocks, block_size, num_heads, head_size].
        # The leading "KV" axis size is *not* fixed to 2: custom
        # KV cache shape specs may use KV=1 (e.g. MLA embedded in
        # a 5D tensor) or other values. Preserve the input's
        # shape[0] to stay layout-agnostic.
        kv_size = kv_tensor.shape[0]
        num_heads = kv_tensor.shape[3]
        head_size = kv_tensor.shape[4]
        kv_reshaped = kv_tensor.reshape(kv_size, -1, num_heads, head_size)
        return kv_reshaped[:, slot_tensor, :, :]
    elif ndim == 3:
        # MLA: [num_blocks, block_size, head_size]
        head_size = kv_tensor.shape[2]
        kv_reshaped = kv_tensor.reshape(-1, head_size)
        return kv_reshaped[slot_tensor, :]
    elif ndim == 4:
        # Alternative: [num_blocks, block_size, num_heads, head_size]
        num_heads = kv_tensor.shape[2]
        head_size = kv_tensor.shape[3]
        kv_reshaped = kv_tensor.reshape(-1, num_heads, head_size)
        return kv_reshaped[slot_tensor, :, :]
    else:
        raise ValueError(
            "Unsupported kv_tensor ndim=%d, shape=%s; expected 3/4/5."
            % (ndim, tuple(kv_tensor.shape))
        )


def slice_by_slot_dim(
    kv_at_slots: torch.Tensor, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Slice an ``extract_kv_at_slots`` result along its slot dim.

    The slot dimension's position depends on the output layout of
    :func:`extract_kv_at_slots`:

    * 4D output (from MHA 5D input): slot dim = 1
    * 3D output (from 4D input):     slot dim = 0
    * 2D output (from MLA 3D input): slot dim = 0

    Args:
        kv_at_slots: Output of :func:`extract_kv_at_slots`.
        start_idx: Inclusive start slot index.
        end_idx: Exclusive end slot index.
    """
    if kv_at_slots.ndim == 4:
        # MHA: [2, num_slots, num_heads, head_size]
        return kv_at_slots[:, start_idx:end_idx, :, :]
    elif kv_at_slots.ndim == 3:
        # 4D case: [num_slots, num_heads, head_size]
        return kv_at_slots[start_idx:end_idx, :, :]
    else:
        # MLA: [num_slots, head_size]
        return kv_at_slots[start_idx:end_idx, :]

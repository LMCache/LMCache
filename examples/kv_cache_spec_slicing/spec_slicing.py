# SPDX-License-Identifier: Apache-2.0
"""CPU reference experiment for vLLM RFC #42082, not a vLLM integration.

The two example specs implement the RFC's proposed slicing signature. They
return matching [layers, blocks, heads, states, head_dim] component views.
The transfer/store/load code only consumes those views; it never detects a
backend layout. Only evenly sharded GQA without head replication is modeled.
"""

# Standard
from dataclasses import dataclass
from typing import Protocol, cast

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


class SliceSpec(Protocol):
    """Experimental subset of the RFC; this is not an upstream public API."""

    @property
    def num_kv_heads(self) -> int:
        """Number of real local KV heads, excluding K/V storage slots."""
        ...

    def slice_for_tp_transfer(
        self,
        my_tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_spec: "SliceSpec",
        other_tp: int,
        other_rank: int,
    ) -> list[torch.Tensor]:
        """Return storage-sharing component views in matching semantic order."""
        ...


def _head_overlap(
    spec: SliceSpec,
    my_tp: int,
    my_rank: int,
    other_spec: SliceSpec,
    other_tp: int,
    other_rank: int,
) -> slice:
    if my_tp <= 0 or other_tp <= 0:
        raise ValueError("TP sizes must be positive")
    if not 0 <= my_rank < my_tp or not 0 <= other_rank < other_tp:
        raise ValueError("Rank must be within its TP group")
    if spec.num_kv_heads * my_tp != other_spec.num_kv_heads * other_tp:
        raise ValueError("This experiment requires evenly sharded matching heads")
    begin = max(my_rank * spec.num_kv_heads, other_rank * other_spec.num_kv_heads)
    end = min(
        (my_rank + 1) * spec.num_kv_heads,
        (other_rank + 1) * other_spec.num_kv_heads,
    )
    return slice(
        max(0, begin - my_rank * spec.num_kv_heads),
        max(0, end - my_rank * spec.num_kv_heads),
    )


@dataclass(frozen=True)
class PackedHeadSpec:
    """Example producer spec for [L, B, 2H, N, D], ordered [K heads | V heads].

    num_kv_heads counts real local KV heads, not the 2H storage slots. Packing
    into heads is an experimental interpretation that needs upstream agreement.
    """

    num_kv_heads: int

    def slice_for_tp_transfer(
        self,
        my_tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_spec: SliceSpec,
        other_tp: int,
        other_rank: int,
    ) -> list[torch.Tensor]:
        """Return separate K/V head-overlap views; never concatenate or copy."""
        heads = _head_overlap(self, my_tp, my_rank, other_spec, other_tp, other_rank)
        return [
            my_tensor[:, :, heads, :, :],
            my_tensor[
                :,
                :,
                self.num_kv_heads + heads.start : self.num_kv_heads + heads.stop,
                :,
                :,
            ],
        ]


@dataclass(frozen=True)
class PackedContentSpec:
    """Example producer spec for [L, B, H, N, 2D] with contiguous [K | V]."""

    num_kv_heads: int

    def slice_for_tp_transfer(
        self,
        my_tensor: torch.Tensor,
        my_tp: int,
        my_rank: int,
        other_spec: SliceSpec,
        other_tp: int,
        other_rank: int,
    ) -> list[torch.Tensor]:
        """Return K/V views in the same order as PackedHeadSpec."""
        heads = _head_overlap(self, my_tp, my_rank, other_spec, other_tp, other_rank)
        width = my_tensor.shape[-1] // 2
        return [my_tensor[:, :, heads, :, :width], my_tensor[:, :, heads, :, width:]]


def store_views(
    views: list[torch.Tensor], block_table: list[int], start: int, end: int
) -> TensorMemoryObj:
    """Save a logical token range into a real LMCache KV_2LTD memory object.

    Args:
        views: Two matching component views [L, B, H, N, D], from a spec.
        block_table: Logical-to-physical block mapping.
        start: Inclusive logical token offset.
        end: Exclusive offset, greater than start and within block_table.

    Returns:
        CPU TensorMemoryObj containing [2, L, end-start, H*D] data.

    Raises:
        ValueError: Component count or token range violates the experiment contract.

    Notes:
        This deliberately slow CPU reference does not replace a GPU connector.
        Storage allocation/copy is expected; only spec slicing is zero-copy.
    """
    if len(views) != 2 or start < 0 or end <= start:
        raise ValueError("Expected two components and a nonempty token range")
    layers, _, heads, states, width = views[0].shape
    if end > len(block_table) * states:
        raise ValueError("Token range exceeds the block table")
    shape = torch.Size((2, layers, end - start, heads * width))
    raw = torch.empty(shape.numel() * views[0].element_size(), dtype=torch.uint8)
    memory = TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=shape,
            dtype=views[0].dtype,
            address=raw.data_ptr(),
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
        ),
        parent_allocator=None,
    )
    packed = cast(torch.Tensor, memory.tensor)
    for component, view in enumerate(views):
        for offset, token in enumerate(range(start, end)):
            block = block_table[token // states]
            packed[component, :, offset].view(layers, heads, width).copy_(
                view[:, block, :, token % states, :]
            )
    return memory


def load_views(
    memory: TensorMemoryObj,
    views: list[torch.Tensor],
    block_table: list[int],
    start: int,
    end: int,
) -> None:
    """Restore the saved range through spec views into producer-owned storage.

    Args:
        memory: Object produced by store_views for the same head partition.
        views: Two destination components with matching L/H/D dimensions.
        block_table: Destination logical-to-physical block mapping.
        start: Inclusive destination logical token offset.
        end: Exclusive destination offset; end-start matches the stored length.

    Raises:
        ValueError: Stored and requested token counts differ.
    """
    packed = cast(torch.Tensor, memory.tensor)
    if memory.get_num_tokens() != end - start:
        raise ValueError("Stored and destination token counts must match")
    layers, _, heads, states, width = views[0].shape
    for component, view in enumerate(views):
        for offset, token in enumerate(range(start, end)):
            block = block_table[token // states]
            view[:, block, :, token % states, :].copy_(
                packed[component, :, offset].view(layers, heads, width)
            )

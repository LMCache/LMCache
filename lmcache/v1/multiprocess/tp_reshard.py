# SPDX-License-Identifier: Apache-2.0
"""Head-sharded KV resharding between tensor-parallel sizes.

A head-sharded object stores, per kernel group, a dense
``[kv_size, num_layers, num_tokens, local_heads * head_elems]`` tensor. Heads
are contiguous and innermost, so the heads a rank needs are a slice of the
last dimension. This module maps the heads of a destination rank at one TP
size onto the source ranks at another TP size and copies those slices.

Rank-to-head assignment follows vLLM's ``QKVParallelLinear``: with
``replicas = max(1, tp // num_kv_heads)``, rank ``r`` holds KV-head shard
``r // replicas`` of size ``max(1, num_kv_heads // tp)``.
"""

# Standard
from dataclasses import dataclass

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc


def _check_tp(num_kv_heads: int, tp: int) -> None:
    if not is_valid_tp(num_kv_heads, tp):
        raise ValueError(
            f"tp={tp} must divide num_kv_heads={num_kv_heads} or be a multiple of it"
        )


def _replicas(num_kv_heads: int, tp: int) -> int:
    return max(1, tp // num_kv_heads)


def is_valid_tp(num_kv_heads: int, tp: int) -> bool:
    """Return whether vLLM can shard ``num_kv_heads`` KV heads over ``tp`` ranks.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.

    Returns:
        True if both are positive and ``tp`` divides ``num_kv_heads`` or is a
        multiple of it.
    """
    if num_kv_heads < 1 or tp < 1:
        return False
    return num_kv_heads % tp == 0 or tp % num_kv_heads == 0


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    """Return the number of KV heads one rank holds at a TP size.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.

    Returns:
        ``max(1, num_kv_heads // tp)``.

    Raises:
        ValueError: If ``tp`` neither divides nor is a multiple of
            ``num_kv_heads``.
    """
    _check_tp(num_kv_heads, tp)
    return max(1, num_kv_heads // tp)


def kv_head_range(num_kv_heads: int, tp: int, rank: int) -> tuple[int, int]:
    """Return the KV heads a rank holds as ``(first_head, num_heads)``.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.
        rank: Rank in ``[0, tp)``.

    Returns:
        The first global head index and the number of heads.

    Raises:
        ValueError: If the TP size is invalid for ``num_kv_heads`` or
            ``rank`` is out of range.
    """
    if not 0 <= rank < tp:
        raise ValueError(f"rank={rank} out of range for tp={tp}")
    n = local_kv_heads(num_kv_heads, tp)
    return (rank // _replicas(num_kv_heads, tp)) * n, n


@dataclass(frozen=True)
class HeadCopy:
    """Copy ``num_heads`` heads from a source rank into a destination rank.

    Attributes:
        src_rank: Source rank holding the heads.
        src_head: First head to copy, local to the source rank.
        dst_head: First head to write, local to the destination rank.
        num_heads: Number of contiguous heads.
    """

    src_rank: int
    src_head: int
    dst_head: int
    num_heads: int


def plan_head_reshard(
    num_kv_heads: int, src_tp: int, dst_tp: int, dst_rank: int
) -> list[HeadCopy]:
    """Plan the copies that build one destination rank's heads.

    Every destination head is covered exactly once. Among replicated source
    ranks the lowest rank is used.

    Args:
        num_kv_heads: Total KV heads of the model.
        src_tp: TP size the objects were stored with.
        dst_tp: TP size of the reader.
        dst_rank: Destination rank in ``[0, dst_tp)``.

    Returns:
        Copies ordered by destination head.

    Raises:
        ValueError: If either TP size is invalid for ``num_kv_heads`` or
            ``dst_rank`` is out of range.
    """
    dst_first, dst_n = kv_head_range(num_kv_heads, dst_tp, dst_rank)
    src_n = local_kv_heads(num_kv_heads, src_tp)
    src_replicas = _replicas(num_kv_heads, src_tp)
    copies = []
    head = dst_first
    while head < dst_first + dst_n:
        shard = head // src_n
        src_first = shard * src_n
        n = min(src_first + src_n, dst_first + dst_n) - head
        copies.append(
            HeadCopy(
                src_rank=shard * src_replicas,
                src_head=head - src_first,
                dst_head=head - dst_first,
                num_heads=n,
            )
        )
        head += n
    return copies


def reshard_layout_desc(
    layout_desc: MemoryLayoutDesc, num_kv_heads: int, from_tp: int, to_tp: int
) -> MemoryLayoutDesc:
    """Derive the layout one rank's object has at another TP size.

    Only the last (``local_heads * head_elems``) dimension of each kernel
    group changes.

    Args:
        layout_desc: Layout at ``from_tp``.
        num_kv_heads: Total KV heads of the model.
        from_tp: TP size ``layout_desc`` describes.
        to_tp: TP size to derive the layout for.

    Returns:
        The layout at ``to_tp``.

    Raises:
        ValueError: If a TP size is invalid for ``num_kv_heads`` or a shape's
            last dimension is not a multiple of the local head count.
    """
    from_n = local_kv_heads(num_kv_heads, from_tp)
    to_n = local_kv_heads(num_kv_heads, to_tp)
    shapes = []
    for shape in layout_desc.shapes:
        if shape[-1] % from_n != 0:
            raise ValueError(
                f"last dim {shape[-1]} of {tuple(shape)} is not a multiple of "
                f"{from_n} local heads"
            )
        shapes.append(torch.Size((*shape[:-1], shape[-1] // from_n * to_n)))
    return MemoryLayoutDesc(shapes=shapes, dtypes=list(layout_desc.dtypes))


def copy_heads(
    dst: list[torch.Tensor],
    src_by_rank: dict[int, list[torch.Tensor]],
    copies: list[HeadCopy],
    num_dst_heads: int,
) -> None:
    """Execute a head reshard plan for every kernel group of one object.

    Args:
        dst: Destination tensors, one per kernel group, last dimension
            ``num_dst_heads * head_elems``.
        src_by_rank: Source tensors per source rank, one per kernel group.
        copies: Plan from :func:`plan_head_reshard`.
        num_dst_heads: Local heads of the destination rank.

    Raises:
        KeyError: If a source rank in the plan is missing from
            ``src_by_rank``.
        ValueError: If a source rank and the destination have different
            kernel group counts.
    """
    for copy in copies:
        for src_t, dst_t in zip(src_by_rank[copy.src_rank], dst, strict=True):
            head_elems = dst_t.shape[-1] // num_dst_heads
            s = copy.src_head * head_elems
            d = copy.dst_head * head_elems
            n = copy.num_heads * head_elems
            dst_t[..., d : d + n].copy_(src_t[..., s : s + n])

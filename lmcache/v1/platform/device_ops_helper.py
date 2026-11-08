# SPDX-License-Identifier: Apache-2.0
"""Opt-in lowering helpers for backends whose block kernel takes addresses.

The CUDA, XPU and Ascend ``multi_layer_block_kv_transfer`` kernels take raw
addresses rather than tensors, so their :class:`DeviceOps` overrides call
:func:`transfer_with_address_kernel` to lower tensor operands before launch.
This module is a library below the ops boundary, not part of the dispatch
path: backends import it, the logical layer
(``transfer_context/base.py``) does not, and a backend bypasses it by not
calling it from its own override. Launch limits stay per backend: each
passes its own ``max_chunks_per_launch``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import synchronize_device
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)


def _paged_pointer_table(
    paged_kv: DiscoverableKVCache,
    engine_kv_format: lmcache_native.EngineKVFormat,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the kernel-ordered per-layer base addresses of ``paged_kv``.

    Args:
        paged_kv: Normalized paged KV tensors.
        engine_kv_format: Layout of ``paged_kv``.
        num_layers: Number of layers to address.
        device: Device the kernel runs on.

    Returns:
        A 1-D int64 tensor on ``device`` with the addresses in the order
        ``KVFormatSpec.data_ptrs`` defines for the format.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only once the backends that
    # import this module have been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec

    addresses = get_spec(paged_kv, engine_kv_format).data_ptrs(list(range(num_layers)))
    # Addresses are unsigned; the kernel reads the same bits as int64.
    as_int64 = np.array(addresses, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(as_int64).to(device=device)


def allocate_pinned_chunk(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a host chunk an address-taking kernel can read and write.

    The kernel dereferences host chunk addresses directly, which only works
    for host memory the device can reach (page-locked memory on CUDA and
    Ascend, USM host memory on XPU).

    Args:
        shape: Canonical chunk shape.
        dtype: KV element dtype.

    Returns:
        An uninitialized pinned CPU tensor of ``shape`` and ``dtype``.
    """
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _stage_unpinned_host_chunks(
    chunks: list[torch.Tensor], is_d2h: bool
) -> tuple[list[torch.Tensor], bool]:
    """Give the kernel chunks it can reach, and say whether a sync is owed.

    The kernel dereferences chunk addresses directly, which only works for
    memory it can reach (device memory, page-locked memory on CUDA and
    Ascend, USM host memory on XPU). Device-resident chunks pass through
    untouched. For unpinned host chunks, ``D2H`` cannot pin the caller's
    tensors in place (that would replace them and break the in-place fill),
    so it writes pinned staging buffers that are copied back after the
    sync. ``H2D`` pins each unpinned host chunk into a temporary whose
    lifetime the sync protects.

    Args:
        chunks: Caller-supplied chunk tensors, host or device resident.
        is_d2h: Whether the transfer gathers paged KV into the chunks.

    Returns:
        ``(kernel_chunks, needs_sync)``: the chunks whose addresses the
        kernel may use, and whether unpinned input forced staging or
        pinning that must be completed by a device sync.
    """
    unstaged = [c.device.type == "cpu" and not c.is_pinned() for c in chunks]
    if not any(unstaged):
        return chunks, False
    if is_d2h:
        logger.warning(
            "Unpinned host chunks in a native D2H block transfer (likely "
            "shared memory); staging through pinned buffers adds a host "
            "copy and a device sync."
        )
        return [
            torch.empty_like(c, pin_memory=True) if s else c
            for c, s in zip(chunks, unstaged, strict=True)
        ], True
    logger.warning(
        "Unpinned host chunks in a native H2D block transfer; pinning "
        "temporaries adds a device sync."
    )
    return [
        c.pin_memory() if s else c for c, s in zip(chunks, unstaged, strict=True)
    ], True


def _launch_block_transfer_batches(
    kernel: Callable[..., None],
    paged_addresses: torch.Tensor,
    chunk_addresses: list[int],
    block_id_tensor: torch.Tensor,
    device: torch.device,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
    max_chunks_per_launch: int,
) -> None:
    """Launch ``kernel`` in batches of at most ``max_chunks_per_launch``.

    ``skip_prefix_n_blocks`` counts leading blocks across the whole
    transfer, so a skip longer than one launch carries into the following
    launches.
    """
    blocks_per_launch = max_chunks_per_launch * lmcache_chunk_size // int(shape_desc.bs)
    for launch, first in enumerate(
        range(0, len(chunk_addresses), max_chunks_per_launch)
    ):
        start = launch * blocks_per_launch
        launch_block_ids = block_id_tensor[start : start + blocks_per_launch]
        kernel(
            paged_addresses,
            chunk_addresses[first : first + max_chunks_per_launch],
            launch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            # A skip longer than one launch carries into the next launches.
            max(0, min(skip_prefix_n_blocks - start, len(launch_block_ids))),
        )


def transfer_with_address_kernel(
    kernel: Callable[..., None],
    paged_kv: DiscoverableKVCache,
    chunks: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
    max_chunks_per_launch: int,
) -> None:
    """Serve :meth:`DeviceOps.multi_layer_block_kv_transfer` on an address kernel.

    Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`. Tensor
    operands are lowered to the kernel's address form, in launches of at
    most ``max_chunks_per_launch`` chunks. Unpinned host chunks are staged
    through pinned buffers for ``D2H`` and pinned into temporaries for
    ``H2D``; both cases synchronize ``device`` before returning, because the
    kernel reaches them through raw addresses that torch cannot track. Pinned
    and device-resident chunks keep the asynchronous path.

    Args:
        kernel: The compiled ``multi_layer_block_kv_transfer``.
        paged_kv: Normalized paged KV tensors on ``device``.
        chunks: Chunk tensors, host or device resident.
        block_ids: Paged-block IDs in chunk order.
        device: Device that owns the paged KV tensors.
        direction: ``D2H`` or ``H2D``.
        shape_desc: Paged-buffer shape descriptor.
        lmcache_chunk_size: Tokens per chunk.
        engine_kv_format: Engine KV layout.
        skip_prefix_n_blocks: Leading blocks to skip across the whole
            transfer; each launch skips the part that falls inside it.
        max_chunks_per_launch: Chunks one kernel launch can address; each
            backend passes its own limit.
    """
    device = torch.device(device)
    is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
    kernel_chunks, needs_sync = _stage_unpinned_host_chunks(chunks, is_d2h)

    paged_addresses = _paged_pointer_table(
        paged_kv, engine_kv_format, int(shape_desc.nl), device
    )
    _launch_block_transfer_batches(
        kernel,
        paged_addresses,
        [chunk.data_ptr() for chunk in kernel_chunks],
        torch.as_tensor(block_ids, dtype=torch.int64, device=device),
        device,
        direction,
        shape_desc,
        lmcache_chunk_size,
        engine_kv_format,
        skip_prefix_n_blocks,
        max_chunks_per_launch,
    )

    if needs_sync:
        # The kernel reaches these buffers through raw addresses that torch's
        # stream tracking cannot see: wait before the host reads the staging
        # copies (D2H) or the pinned temporaries are released (H2D).
        synchronize_device(device)
        if is_d2h:
            for dst, src in zip(chunks, kernel_chunks, strict=True):
                if src is not dst:
                    dst.copy_(src)

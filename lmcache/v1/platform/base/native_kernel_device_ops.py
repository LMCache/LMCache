# SPDX-License-Identifier: Apache-2.0
"""DeviceOps for backends whose compiled block-transfer kernel takes addresses.

CUDA/ROCm (``lmcache.cuda_ops``), XPU (``lmcache.xpu_ops``) and Ascend
(``lmcache_ascend.ops``) ship a ``multi_layer_block_kv_transfer`` kernel that
takes a device tensor of per-layer base addresses, chunk addresses as
``list[int]`` and a device block-ID tensor. Its GPU threads read host chunks
directly, so they must be pinned, and one launch addresses at most four chunks.
:class:`NativeKernelDeviceOps` keeps that kernel off the public method and
serves the tensor contract of :meth:`DeviceOps.multi_layer_block_kv_transfer`
on top of it.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import synchronize_device
from lmcache.v1.platform.base.device_ops import DeviceOps
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
    # ``lmcache.device_ops``, which is resolved only once this module (one of
    # the backends behind it) has been imported.
    # First Party
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec

    addresses = get_spec(paged_kv, engine_kv_format).data_ptrs(list(range(num_layers)))
    # Addresses are unsigned; the kernel reads the same bits as int64.
    as_int64 = np.array(addresses, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(as_int64).to(device=device)


class NativeKernelDeviceOps(DeviceOps):
    """DeviceOps whose block transfer runs on an address-taking compiled kernel.

    Subclasses load their extension in :meth:`ensure_native` through
    :meth:`bind_native` as usual. Callers keep passing tensors to
    :meth:`multi_layer_block_kv_transfer`; this class builds the kernel's
    address operands, pins or stages host chunks, and splits launches. Without
    the kernel (extension missing, or built without the op) it behaves exactly
    like :class:`DeviceOps`.
    """

    #: Chunks one kernel launch can address (``TORCH_CHECK(num_objects <= 4)``
    #: in ``csrc/cuda/mp_mem_kernels.cu``).
    MAX_CHUNKS_PER_LAUNCH: ClassVar[int] = 4

    def __init__(self) -> None:
        super().__init__()
        # The compiled block-transfer kernel, or None while the torch baseline
        # serves this instance.
        self._native_block_transfer: Callable[..., None] | None = None

    def bind_native(self, module: object) -> None:
        """Bind ``module`` like :class:`DeviceOps`, keeping its block kernel private.

        Args:
            module: Compiled extension module.
        """
        super().bind_native(module)
        # The base installs every callable as an instance attribute, which
        # would hide the tensor method below; take the block kernel back.
        self._native_block_transfer = vars(self).pop(
            "multi_layer_block_kv_transfer", None
        )

    def allocate_block_transfer_output(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate pinned host chunks when the kernel will read them.

        Args:
            shape: Canonical chunk shape.
            dtype: KV element dtype.

        Returns:
            A pinned CPU tensor while the kernel is bound, otherwise the
            :class:`DeviceOps` allocation.
        """
        if self._native_block_transfer is None:
            return super().allocate_block_transfer_output(shape, dtype)
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: torch.Tensor | list,
        lmcache_objects_ptrs: list[int] | list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Run the block transfer on the compiled kernel when one is bound.

        Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`.
        Unpinned host chunks are staged through pinned buffers for ``D2H`` and
        pinned into temporaries for ``H2D``; both cases synchronize ``device``
        before returning, because the kernel reaches them through raw
        addresses that torch cannot track. Pinned chunks keep the asynchronous
        path.

        Args:
            paged_buffer_ptrs_tensor: Normalized paged KV tensors on ``device``.
            lmcache_objects_ptrs: Host chunk tensors.
            block_ids: Paged-block IDs in chunk order.
            device: Device that owns the paged KV tensors.
            direction: ``D2H`` or ``H2D``.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per chunk.
            engine_kv_format: Engine KV layout.
            skip_prefix_n_blocks: Leading blocks to skip, applied to the first
                kernel launch.
        """
        native = self._native_block_transfer
        args = (
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        if native is None:
            super().multi_layer_block_kv_transfer(*args)
            return
        if not all(isinstance(chunk, torch.Tensor) for chunk in lmcache_objects_ptrs):
            # LMCache-driven callers still pass addresses they batched
            # themselves; forward them untouched until #4465 step 3.10.
            native(*args)
            return

        chunks = cast("list[torch.Tensor]", lmcache_objects_ptrs)
        device = torch.device(device)
        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        unpinned = not all(chunk.is_pinned() for chunk in chunks)
        kernel_chunks = chunks
        if unpinned and is_d2h:
            # Pinning would replace the caller's tensors and break the
            # in-place fill, so the kernel writes pinned staging buffers that
            # are copied back below.
            logger.warning(
                "Unpinned host chunks in a native D2H block transfer (likely "
                "shared memory); staging through pinned buffers adds a host "
                "copy and a device sync."
            )
            kernel_chunks = [torch.empty_like(c, pin_memory=True) for c in chunks]
        elif unpinned:
            logger.warning(
                "Unpinned host chunks in a native H2D block transfer; pinning "
                "temporaries adds a device sync."
            )
            kernel_chunks = [c if c.is_pinned() else c.pin_memory() for c in chunks]

        paged_addresses = _paged_pointer_table(
            paged_buffer_ptrs_tensor, engine_kv_format, int(shape_desc.nl), device
        )
        chunk_addresses = [chunk.data_ptr() for chunk in kernel_chunks]
        block_id_tensor = torch.as_tensor(block_ids, dtype=torch.int64, device=device)
        blocks_per_launch = (
            self.MAX_CHUNKS_PER_LAUNCH * lmcache_chunk_size // int(shape_desc.bs)
        )
        for launch, first in enumerate(
            range(0, len(chunk_addresses), self.MAX_CHUNKS_PER_LAUNCH)
        ):
            start = launch * blocks_per_launch
            native(
                paged_addresses,
                chunk_addresses[first : first + self.MAX_CHUNKS_PER_LAUNCH],
                block_id_tensor[start : start + blocks_per_launch],
                device,
                direction,
                shape_desc,
                lmcache_chunk_size,
                engine_kv_format,
                skip_prefix_n_blocks if launch == 0 else 0,
            )

        if unpinned:
            # The kernel reaches these buffers through raw addresses that
            # torch's stream tracking cannot see: wait before the host reads
            # the staging copies (D2H) or the pinned temporaries are released
            # (H2D).
            synchronize_device(device)
            if is_d2h:
                for dst, src in zip(chunks, kernel_chunks, strict=True):
                    dst.copy_(src)

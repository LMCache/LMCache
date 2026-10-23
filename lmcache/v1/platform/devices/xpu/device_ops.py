# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built, a warning is logged and the instance stays on the torch
fallback. It overrides :meth:`multi_layer_block_kv_transfer` to lower tensor
operands to the address-taking SYCL kernel.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.native_block_transfer import (
    allocate_pinned_chunk,
    transfer_with_address_kernel,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def __init__(self) -> None:
        super().__init__()
        # The compiled address-taking block kernel. It stays None only before
        # ensure_native() runs or when the extension is not installed, and the
        # torch baseline then serves block transfers.
        self._block_transfer_kernel: Callable[..., None] | None = None

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            # Not installed: stay on the torch baseline without retrying.
            self._native_bound = True
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        # Every supported build exports the block kernel, so a module without
        # it is a broken install: let the AttributeError propagate rather than
        # serving the torch baseline. ``_native_bound`` stays False, so the
        # next call fails the same way.
        self._block_transfer_kernel = sycl.multi_layer_block_kv_transfer
        self.bind_native(sycl, skip={"multi_layer_block_kv_transfer"})
        self._native_bound = True

    def allocate_block_transfer_output(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate pinned host chunks when the compiled kernel will read them.

        Args:
            shape: Canonical chunk shape.
            dtype: KV element dtype.

        Returns:
            A pinned CPU tensor while the kernel is bound, otherwise the
            :class:`DeviceOps` allocation.
        """
        if self._block_transfer_kernel is None:
            return super().allocate_block_transfer_output(shape, dtype)
        return allocate_pinned_chunk(shape, dtype)

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
        """Run the block transfer on the compiled SYCL kernel when it is bound.

        Operands follow :meth:`DeviceOps.multi_layer_block_kv_transfer`.
        If the extension is not loaded, the torch baseline serves.
        """
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
        if self._block_transfer_kernel is None:
            super().multi_layer_block_kv_transfer(*args)
            return
        transfer_with_address_kernel(self._block_transfer_kernel, *args)

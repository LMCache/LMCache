# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built, a warning is logged and the instance stays on the torch
fallback.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.common.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.device_ops_helper import (
    transfer_with_address_kernel,
)
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)

#: Chunks per kernel launch. The SYCL kernel has no per-launch limit; this
#: matches the CUDA kernel's, and raising it needs validation on XPU hardware.
MAX_CHUNKS_PER_LAUNCH = 4


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def __init__(self) -> None:
        super().__init__()
        self._block_transfer_kernel: Callable[..., None] | None = None

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            self._native_bound = True
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        self._block_transfer_kernel = sycl.multi_layer_block_kv_transfer
        self.bind_native(sycl, skip={"multi_layer_block_kv_transfer"})
        self._native_bound = True

    def multi_layer_block_kv_transfer(
        self,
        paged_kv: DiscoverableKVCache,
        chunks: list[torch.Tensor],
        block_ids: torch.Tensor | list[int],
        device: torch.device | str,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        args = (
            paged_kv,
            chunks,
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
        transfer_with_address_kernel(
            self._block_transfer_kernel,
            *args,
            max_chunks_per_launch=MAX_CHUNKS_PER_LAUNCH,
        )

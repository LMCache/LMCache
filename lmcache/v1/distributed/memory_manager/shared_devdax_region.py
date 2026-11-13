# SPDX-License-Identifier: Apache-2.0
"""A Device-DAX mapping whose extents belong to a memory orchestrator."""

# Standard
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_device_identity, get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.memory_allocators.devdax_memory_allocator import DevDaxMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.memory_orchestrator.api import RegionContract
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


class SharedDevDaxRegion:
    """Map and GPU-register orchestrator-owned bytes without a local allocator.

    This is a mapping, not an L1 memory manager: it only provides views, range
    ownership and cleanup. The caller must drain copies before closing it.
    """

    def __init__(self, config: L1MemoryManagerConfig, contract: RegionContract) -> None:
        """Map the region at the configured device offset and GPU-register it.

        Args:
            config: Supplies ``devdax_path`` and ``devdax_offset_in_bytes``.
            contract: Supplies the region's capacity and extent alignment.

        Raises:
            ValueError: If ``devdax_path`` is not configured.
            RuntimeError: If the range leaves the device, breaks its alignment,
                or cannot be GPU-registered.
            OSError: If the device cannot be opened or mapped.
        """
        if not config.devdax_path:
            raise ValueError("SharedDevDaxRegion requires devdax_path")
        self._path = config.devdax_path
        self._offset = config.devdax_offset_in_bytes
        self._capacity = contract.capacity_bytes
        self._alignment = contract.alignment_bytes
        self._fd, self._mmap, self._mmap_buffer, self._buffer = (
            DevDaxMemoryAllocator.open_devdax_mapping(
                self._path, self._capacity, self._offset
            )
        )
        self._registered = False
        try:
            # Shared direct transfers require registered memory.
            self._registered = current_device_spec.pin_memory(
                self._buffer.data_ptr(), self._capacity
            )
            if not self._registered:
                raise RuntimeError(
                    f"A shared Device-DAX region must be GPU-registered: {self._path}"
                )
        except BaseException:
            self.close()
            raise

    def view(
        self, offset: int, length: int, layout: MemoryLayoutDesc
    ) -> TensorMemoryObj:
        """Return a parentless view of an extent the orchestrator granted.

        Dropping the view never frees the orchestrator's bytes. The region
        must remain open while callers request views or submit copies.

        Args:
            offset: Byte offset of the extent in the region.
            length: Aligned extent length in bytes.
            layout: Shapes and dtypes of the object's tensors.

        Returns:
            A memory object over ``[offset, offset + length)`` of the region.

        Raises:
            ValueError: The extent is negative, misaligned, outside the region,
                or smaller than the layout's payload.
        """
        payload = get_size_bytes(layout.shapes, layout.dtypes)
        if (
            offset < 0
            or offset % self._alignment
            or length < payload
            or offset + length > self._capacity
        ):
            raise ValueError(
                f"extent [{offset}, {offset + length}) for a {payload}-byte "
                f"object does not fit the {self._capacity}-byte region"
            )
        return TensorMemoryObj(
            raw_data=self._buffer[offset : offset + length],
            metadata=MemoryObjMetadata(
                layout.shapes[0],
                layout.dtypes[0],
                offset,
                length,
                1,
                0,
                MemoryFormat.KV_2LTD,
                shapes=list(layout.shapes),
                dtypes=list(layout.dtypes),
            ),
            parent_allocator=None,
        )

    def owns_range(
        self, device_path: str, offset_in_bytes: int, size_in_bytes: int
    ) -> bool:
        """Return whether a device range overlaps this open mapping.

        Args:
            device_path: Candidate device path or alias.
            offset_in_bytes: Start of the candidate range on that device.
            size_in_bytes: Length of the candidate range.

        Returns:
            True when the device identity matches and the ranges overlap.
        """
        identity = get_device_identity(device_path)
        return (
            identity is not None
            and identity == get_device_identity(self._fd)
            and offset_in_bytes < self._offset + self._capacity
            and self._offset < offset_in_bytes + size_in_bytes
        )

    def close(self) -> None:
        """Unregister and unmap once, after the caller's copies drain.

        If exported views prevent unmapping, the region is left mapped.
        Repeated calls after a successful close do nothing. If unregistering
        raises, the mapping is retained so close can be retried.
        """
        if self._fd < 0:
            return
        if self._registered:
            current_device_spec.unpin_memory(self._buffer.data_ptr())
            self._registered = False
        self._buffer = torch.empty(0, dtype=torch.uint8)
        self._mmap_buffer = None
        try:
            self._mmap.close()
        except BufferError:
            logger.warning(
                "Shared Device-DAX region %s still has live views; leaving it mapped",
                self._path,
            )
        os.close(self._fd)
        self._fd = -1

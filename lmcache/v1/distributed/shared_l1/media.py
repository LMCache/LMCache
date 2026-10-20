# SPDX-License-Identifier: Apache-2.0
"""A GPU-registered mapping of one shared DRAM file or Device-DAX region."""

# Standard
from pathlib import Path
import os
import stat

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_device_identity, get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.memory_allocators.devdax_memory_allocator import open_devdax_mapping
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import current_device_spec

logger = init_logger(__name__)


class RegionMedium:
    """Own a shared mapping, its GPU registration, and bounds-checked views.

    Args:
        path: Regular file or Device-DAX path to map.
        capacity_bytes: Region capacity from the orchestrator.
        alignment_bytes: Extent alignment from the orchestrator.
        backend_type: DRAM for a shared file, DEVDAX for a character device.

    Raises:
        ValueError: The backend, backing type, or device capacity is invalid.
        OSError: The backing or Device-DAX size cannot be read or mapped.
        RuntimeError: GPU registration fails; pageable staging is unsupported.
    """

    data_path = "direct-pinned"

    def __init__(
        self,
        path: str,
        capacity_bytes: int,
        alignment_bytes: int,
        backend_type: L1BackendType,
    ) -> None:
        self.path = path
        self.backend_type = backend_type
        self.capacity_bytes = capacity_bytes
        self._alignment_bytes = alignment_bytes
        if backend_type is L1BackendType.DEVDAX:
            device = os.stat(self.path)
            if not stat.S_ISCHR(device.st_mode):
                raise ValueError("A shared DEVDAX L1 requires a character device")
            size = int(
                Path(
                    f"/sys/dev/char/{os.major(device.st_rdev)}:{os.minor(device.st_rdev)}/size"
                ).read_text()
            )
            if size < capacity_bytes:
                raise ValueError(f"{self.path} holds less than the region capacity")
        elif backend_type is L1BackendType.DRAM:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("A shared DRAM L1 requires a regular file")
                if info.st_size < capacity_bytes:
                    os.ftruncate(fd, capacity_bytes)
            finally:
                os.close(fd)
        else:
            raise ValueError("A shared L1 requires a DRAM or DEVDAX configuration")
        self._fd, self._mmap, self._mmap_buffer, self._buffer = open_devdax_mapping(
            self.path, capacity_bytes
        )
        self._base_address = self._buffer.data_ptr()
        self._pinned = False
        try:
            self._pinned = current_device_spec.pin_memory(
                self._base_address, capacity_bytes
            )
            if not self._pinned:
                raise RuntimeError(
                    f"Shared L1 requires GPU-registered memory: {self.path}"
                )
        except BaseException:
            self.close()
            raise

    def view(
        self, offset: int, length: int, layout: MemoryLayoutDesc
    ) -> TensorMemoryObj:
        """Build a view whose lifetime does not free the orchestrator's extent.

        Args:
            offset: Granted byte offset in the mapping.
            length: Aligned extent length.
            layout: Shapes and dtypes of the object's tensors.

        Returns:
            A memory object with no parent allocator.

        Raises:
            ValueError: The extent is misaligned, too small or outside the region.
        """
        payload = get_size_bytes(layout.shapes, layout.dtypes)
        if (
            offset % self._alignment_bytes
            or length < payload
            or offset < 0
            or offset + length > self.capacity_bytes
        ):
            raise ValueError(
                f"extent [{offset}, {offset + length}) for a {payload}-byte "
                f"object does not fit the {self.capacity_bytes}-byte region"
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

    def owns_device(self, device_path: str) -> bool:
        """Return whether a candidate path names this mapped device or file."""
        identity = get_device_identity(device_path)
        return identity is not None and identity == get_device_identity(self._fd)

    def close(self) -> None:
        """Unregister and unmap after copies drain; live views defer unmapping."""
        if self._fd < 0:
            return
        if self._pinned:
            current_device_spec.unpin_memory(self._base_address)
            self._pinned = False
        self._buffer = torch.empty(0, dtype=torch.uint8)
        self._mmap_buffer = None
        try:
            self._mmap.close()
        except BufferError:
            logger.warning(
                "Shared L1 %s still has live views; leaving it mapped", self.path
            )
        os.close(self._fd)
        self._fd = -1

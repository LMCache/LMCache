# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio

# Local
from .connector_client_base import ConnectorClientBase

try:
    # First Party
    from lmcache.lmcache_blkio import LMCacheBlkioClient

    BLKIO_AVAILABLE = True
except ImportError:
    BLKIO_AVAILABLE = False
    LMCacheBlkioClient = None  # type: ignore


class BlkioClient(ConnectorClientBase[LMCacheBlkioClient]):
    """Python client for the native libblkio storage connector.

    Wraps the C++ ``LMCacheBlkioClient`` (pybind11) and integrates it
    with the asyncio event loop via ``ConnectorClientBase``.

    Args:
        device_path: Path to the block device (e.g. ``/dev/nvme0n1``).
        num_workers: Number of I/O worker threads.  Each worker gets
            its own ``io_uring`` instance through libblkio.
        direct_io: Enable ``O_DIRECT`` for bypassing the page cache
            (default ``True``).
        loop: Optional asyncio event loop.  Falls back to the running
            loop when ``None``.
    """

    def __init__(
        self,
        device_path: str,
        num_workers: int,
        direct_io: bool = True,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        if not BLKIO_AVAILABLE:
            raise RuntimeError(
                "BlkioClient requires the C++ libblkio extension. "
                "Install libblkio-dev and rebuild with: pip install -e ."
            )
        native_client = LMCacheBlkioClient(device_path, num_workers, direct_io)
        super().__init__(native_client, loop)

# SPDX-License-Identifier: Apache-2.0
"""
Blkio (libblkio) native L2 adapter config and factory.

Backed by the native C++ libblkio connector (io_uring block device I/O)
wrapped with ``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)


class BlkioL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    libblkio connector (io_uring block device I/O).

    Fields:
    - device_path: path to the block device
      (e.g. ``/dev/nvme0n1``).
    - num_workers: C++ worker threads for I/O
      (default 4).  Each worker gets its own
      io_uring instance through libblkio.
    - direct_io: bypass page cache via O_DIRECT
      (default true).
    """

    def __init__(
        self,
        device_path: str,
        num_workers: int = 4,
        direct_io: bool = True,
    ) -> None:
        """Initialize a BlkioL2AdapterConfig.

        Args:
            device_path: Path to the block device
                (e.g. ``/dev/nvme0n1``).
            num_workers: Number of C++ worker threads for I/O.
                Each worker gets its own io_uring instance
                through libblkio.  Default 4.
            direct_io: If ``True``, bypass the page cache via
                ``O_DIRECT``.  Default ``True``.
        """
        super().__init__()
        self.device_path = device_path
        self.num_workers = num_workers
        self.direct_io = direct_io

    @classmethod
    def from_dict(cls, d: dict) -> "BlkioL2AdapterConfig":
        """Create a :class:`BlkioL2AdapterConfig` from a dictionary.

        Args:
            d: Dictionary with keys ``device_path`` (required),
                ``num_workers`` (optional, default 4), and
                ``direct_io`` (optional, default ``True``).

        Returns:
            A validated :class:`BlkioL2AdapterConfig` instance.

        Raises:
            ValueError: If any field has an invalid type or value.
        """
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path:
            raise ValueError(
                "device_path must be a non-empty string"
            )

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError(
                "num_workers must be a positive integer"
            )

        direct_io = d.get("direct_io", True)
        if not isinstance(direct_io, bool):
            raise ValueError("direct_io must be a boolean")

        return cls(
            device_path=device_path,
            num_workers=num_workers,
            direct_io=direct_io,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Blkio L2 adapter config fields:\n"
            "- device_path (str): path to the block "
            "device (required, e.g. /dev/nvme0n1)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- direct_io (bool): bypass page cache "
            "via O_DIRECT (default true)"
        )


def _create_blkio_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ libblkio connector."""
    try:
        # First Party
        from lmcache.lmcache_blkio import (
            LMCacheBlkioClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Blkio L2 adapter requires the C++ libblkio "
            "extension. Install libblkio-dev and rebuild "
            "with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, BlkioL2AdapterConfig)
    native_client = LMCacheBlkioClient(
        config.device_path,
        config.num_workers,
        config.direct_io,
    )
    logger.info(
        "Created blkio L2 adapter: %s (workers=%d, direct_io=%s)",
        config.device_path,
        config.num_workers,
        config.direct_io,
    )
    return NativeConnectorL2Adapter(native_client)


register_l2_adapter_type("blkio", BlkioL2AdapterConfig)
register_l2_adapter_factory("blkio", _create_blkio_l2_adapter)

# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
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


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    - read_io_depth: threads dedicated to executing reads, and so the
      maximum reads in flight.  0 keeps the legacy path, where reads run
      on the worker threads and the depth against the device therefore
      equals num_workers.  Bytes in flight never exceed read_io_depth x
      object size whatever read_max_bytes_in_flight says.
    - read_max_bytes_in_flight: bytes the connector may keep outstanding
      against the device.  Throughput is set by bytes in flight rather
      than by object count, so this is the figure that decides it.  0 --
      the default -- selects 1536 MiB, chosen for its worst case across
      local and network-latency storage; see connector.h.  Only consulted
      when read_io_depth is positive, and only reachable when that depth
      is large enough to hold the budget in objects.
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        read_io_depth: int = 0,
        read_max_bytes_in_flight: int = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.read_io_depth = read_io_depth
        self.read_max_bytes_in_flight = read_max_bytes_in_flight

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        read_io_depth = d.get("read_io_depth", 0)
        if not isinstance(read_io_depth, int) or read_io_depth < 0:
            raise ValueError("read_io_depth must be a non-negative integer")

        read_max_bytes_in_flight = d.get("read_max_bytes_in_flight", 0)
        if (
            not isinstance(read_max_bytes_in_flight, int)
            or read_max_bytes_in_flight < 0
        ):
            raise ValueError("read_max_bytes_in_flight must be a non-negative integer")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            read_io_depth=read_io_depth,
            read_max_bytes_in_flight=read_max_bytes_in_flight,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)\n"
            "- read_io_depth (int): threads dedicated to "
            "reads, decoupling device queue depth from "
            "num_workers (default 0 = legacy path)\n"
            "- read_max_bytes_in_flight (int): bytes kept "
            "outstanding against the device (default 0 = "
            "1536 MiB)"
        )


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
        config.read_io_depth,
        config.read_max_bytes_in_flight,
    )
    # The connector substitutes its own default when read_io_depth is on
    # and no budget was configured, so ask it rather than reporting the
    # raw 0 the caller passed.
    effective_read_budget = native_client.read_budget_bytes()
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
        "read_ahead=%s, read_io_depth=%d, "
        "read_max_bytes_in_flight=%d)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        config.read_io_depth,
        effective_read_budget,
    )
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "read_io_depth": config.read_io_depth,
            "read_max_bytes_in_flight": effective_read_budget,
        },
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)

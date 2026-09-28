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
from lmcache.v1.distributed.l2_adapters.disk_guard import DiskGuard
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
    - max_capacity_gb: declared L2 capacity in GB, used for usage
      accounting. ``0`` (default) disables it. A value ``> 0`` does
      **not** bound disk usage on its own -- eviction runs only when
      the adapter spec also carries an ``eviction`` block, e.g.
      ``{"eviction": {"eviction_policy": "LRU"}}``. Without one,
      files accumulate past the declared capacity.
    - disk_high_watermark: used share of the filesystem (0-1, as ``df``
      reports it) this adapter must not push the disk past. ``0``
      (default) disables it.
    - disk_min_free_gb: free space in GB this adapter must leave on the
      filesystem. ``0`` (default) disables it.

    The two disk limits shrink the adapter's capacity as the filesystem
    fills, whoever fills it, so they need ``max_capacity_gb`` and an
    ``eviction`` block to act on.
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        disk_high_watermark: float = 0,
        disk_min_free_gb: float = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.disk_high_watermark = disk_high_watermark
        self.disk_min_free_gb = disk_min_free_gb

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
        if max_capacity_gb > 0 and d.get("eviction") is None:
            logger.warning(
                "fs_native: max_capacity_gb=%s is declared for %s but no "
                "'eviction' block is configured, so it is used for usage "
                "accounting only and will NOT bound disk usage. Add an "
                "eviction block to enforce it, e.g. "
                '"eviction": {"eviction_policy": "LRU"}.',
                max_capacity_gb,
                base_path,
            )

        disk_high_watermark = d.get("disk_high_watermark", 0)
        if (
            isinstance(disk_high_watermark, bool)
            or not isinstance(disk_high_watermark, (int, float))
            or not 0 <= disk_high_watermark <= 1
        ):
            raise ValueError("disk_high_watermark must be a number in [0, 1]")
        disk_min_free_gb = d.get("disk_min_free_gb", 0)
        if (
            isinstance(disk_min_free_gb, bool)
            or not isinstance(disk_min_free_gb, (int, float))
            or disk_min_free_gb < 0
        ):
            raise ValueError("disk_min_free_gb must be a non-negative number")
        if (disk_high_watermark > 0 or disk_min_free_gb > 0) and (
            max_capacity_gb <= 0 or d.get("eviction") is None
        ):
            raise ValueError(
                "disk_high_watermark / disk_min_free_gb need max_capacity_gb > 0 "
                "and an 'eviction' block: they work by shrinking the capacity "
                "that eviction enforces"
            )

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            disk_high_watermark=float(disk_high_watermark),
            disk_min_free_gb=float(disk_min_free_gb),
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
            "- max_capacity_gb (float): declared L2 capacity in GB "
            "for usage accounting (default 0 = disabled). Does not "
            "bound disk usage by itself; add an 'eviction' block "
            "to enforce it\n"
            "- disk_high_watermark (float): used share of the filesystem "
            "(0-1) the adapter must not push the disk past (default 0 = "
            "disabled)\n"
            "- disk_min_free_gb (float): free GB the adapter must leave on "
            "the filesystem (default 0 = disabled)"
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
    )
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, read_ahead=%s)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
    )
    disk_guard = None
    if config.disk_high_watermark > 0 or config.disk_min_free_gb > 0:
        disk_guard = DiskGuard(
            config.base_path,
            high_watermark=config.disk_high_watermark,
            min_free_bytes=int(config.disk_min_free_gb * (1024**3)),
        )
        logger.info(
            "FS native L2 adapter %s: disk limits high_watermark=%s min_free_gb=%s",
            config.base_path,
            config.disk_high_watermark or "off",
            config.disk_min_free_gb or "off",
        )
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        disk_guard=disk_guard,
        type_name="FSNativeL2Adapter",
        pad_buffers_to_alignment=config.use_odirect,
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "pad_buffers_to_alignment": config.use_odirect,
        },
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)

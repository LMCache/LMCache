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
import os
import time

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        PersistedObject,
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
    - max_capacity_gb: declared L2 capacity in GB, used for usage
      accounting. ``0`` (default) disables it. A value ``> 0`` does
      **not** bound disk usage on its own -- eviction runs only when
      the adapter spec also carries an ``eviction`` block, e.g.
      ``{"eviction": {"eviction_policy": "LRU"}}``. Without one,
      files accumulate past the declared capacity.
    - recover_on_start: at startup, register the ``.data`` files already in
      ``base_path`` (default true). They stay readable either way (lookup
      checks the file), but only registered files count toward
      ``max_capacity_gb`` and can be evicted; unregistered ones persist
      until removed by hand.
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
        recover_on_start: bool = True,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.recover_on_start = recover_on_start

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

        recover_on_start = d.get("recover_on_start", True)
        if not isinstance(recover_on_start, bool):
            raise ValueError("recover_on_start must be a boolean")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
            recover_on_start=recover_on_start,
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
            "- recover_on_start (bool): register the files already in "
            "base_path at startup so they count toward max_capacity_gb "
            "and can be evicted (default true)"
        )


def _scan_persisted_objects(base_path: str) -> list[PersistedObject]:
    """List the chunk files a previous process left in ``base_path``.

    The native connector writes one flat ``.data`` file per key, with the
    same reversible name as the Python FS adapter. In-flight ``.tmp`` files
    and names that do not decode to an ``ObjectKey`` are skipped. A file
    removed between listing and ``stat`` is skipped too.
    """
    # Lazy imports, as in the factory below, to avoid a circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (  # noqa: PLC0415
        _filename_to_object_key,
    )
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501, PLC0415
        PersistedObject,
    )

    start = time.monotonic()
    found: list[PersistedObject] = []
    skipped = 0
    with os.scandir(base_path) as entries:
        for entry in entries:
            key = _filename_to_object_key(entry.name)
            if key is None:
                skipped += 1
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            found.append(
                PersistedObject(key=key, size=stat.st_size, mtime=stat.st_mtime)
            )
    logger.info(
        "fs_native: found %d persisted objects (%.2f GiB) in %s in %.1f s "
        "(%d entries skipped)",
        len(found),
        sum(obj.size for obj in found) / 1024**3,
        base_path,
        time.monotonic() - start,
        skipped,
    )
    return found


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
    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        pad_buffers_to_alignment=config.use_odirect,
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
            "pad_buffers_to_alignment": config.use_odirect,
            "recover_on_start": config.recover_on_start,
        },
        persisted_object_scanner=(
            (lambda: _scan_persisted_objects(config.base_path))
            if config.recover_on_start
            else None
        ),
    )


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)

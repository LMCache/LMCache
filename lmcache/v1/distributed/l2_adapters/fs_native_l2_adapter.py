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

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
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
from lmcache.v1.distributed.l2_adapters.fs_key_codec import filename_to_object_key

logger = init_logger(__name__)


def _scan_existing_cache_entries(
    base_path: str,
) -> tuple[list[tuple[ObjectKey, int, int]], int]:
    """Scan one native-FS cache directory for persisted entries.

    Args:
        base_path: Directory containing native filesystem cache files.

    Returns:
        A pair of ``(entries, skipped_files)``. Each entry contains its
        decoded key, byte size, and modification timestamp in nanoseconds.

    Raises:
        RuntimeError: If the cache directory itself cannot be scanned.
    """
    recovered: list[tuple[ObjectKey, int, int]] = []
    skipped_files = 0
    try:
        with os.scandir(base_path) as entries:
            for entry in entries:
                if not entry.name.endswith(".data"):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        skipped_files += 1
                        continue
                    key = filename_to_object_key(entry.name)
                    if key is None:
                        skipped_files += 1
                        logger.warning(
                            "Ignoring unrecognized fs_native cache file: %s",
                            entry.path,
                        )
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A concurrent cleanup may remove an entry between scandir
                    # and stat. It no longer contributes to cache usage.
                    continue
                except OSError as e:
                    raise RuntimeError(
                        f"Unable to inspect fs_native cache file {entry.path!r}: {e}"
                    ) from e
                recovered.append((key, stat.st_size, stat.st_mtime_ns))
    except OSError as e:
        raise RuntimeError(
            f"Unable to scan fs_native cache directory {base_path!r}: {e}"
        ) from e
    return recovered, skipped_files


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
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb

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

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
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
            "(default 0 = disabled)"
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
        RecoveredL2Entry,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    try:
        scanned_entries, skipped_files = _scan_existing_cache_entries(config.base_path)
        recovered_entries = [
            RecoveredL2Entry(key=key, size_bytes=size, mtime_ns=mtime_ns)
            for key, size, mtime_ns in scanned_entries
        ]
        logger.info(
            "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
            "read_ahead=%s, recovered_keys=%d, recovered_bytes=%d, skipped=%d)",
            config.base_path,
            config.num_workers,
            config.use_odirect,
            config.read_ahead_size,
            len(recovered_entries),
            sum(entry.size_bytes for entry in recovered_entries),
            skipped_files,
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
                "recovery_skipped_files": skipped_files,
            },
            recovered_entries=recovered_entries,
        )
    except Exception:
        native_client.close()
        raise


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)

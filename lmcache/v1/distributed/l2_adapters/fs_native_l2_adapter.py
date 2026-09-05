# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import fcntl
import os
import uuid

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
from lmcache.v1.distributed.l2_adapters.fs_key_codec import (
    filename_to_object_key,
    object_key_to_filename,
)

logger = init_logger(__name__)

_DISK_UUID_FILENAME = ".lmcache_disk_uuid"
_DISK_UUID_LOCK_FILENAME = ".lmcache_disk_uuid.lock"


def _read_disk_uuid(path: Path) -> str:
    """Read and validate a persisted disk UUID without following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, encoding="ascii") as uuid_file:
            value = uuid_file.read(128).strip()
    except (OSError, UnicodeError) as e:
        raise RuntimeError(f"Unable to read fs_native disk UUID {path!s}: {e}") from e
    try:
        return str(uuid.UUID(value))
    except ValueError as e:
        raise RuntimeError(
            f"Invalid fs_native disk UUID in {path!s}; refusing to remap cache data"
        ) from e


def get_or_create_disk_uuid(base_path: str) -> str:
    """Return the persistent UUID associated with one cache mount.

    Args:
        base_path: Root directory of one independently mounted cache disk.

    Returns:
        Canonical UUID string persisted below ``base_path``.

    Raises:
        RuntimeError: If the directory or identity file cannot be created,
            read, or validated.
    """
    base = Path(base_path)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Unable to create fs_native cache directory {base!s}: {e}"
        ) from e

    uuid_path = base / _DISK_UUID_FILENAME
    lock_path = base / _DISK_UUID_LOCK_FILENAME
    lock_flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        lock_fd = os.open(lock_path, lock_flags, 0o644)
    except OSError as e:
        raise RuntimeError(
            f"Unable to lock fs_native disk identity in {base!s}: {e}"
        ) from e

    temp_path: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if uuid_path.exists() or uuid_path.is_symlink():
            return _read_disk_uuid(uuid_path)

        value = str(uuid.uuid4())
        temp_path = base / f"{_DISK_UUID_FILENAME}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        temp_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temp_fd = os.open(temp_path, temp_flags, 0o644)
        try:
            os.write(temp_fd, f"{value}\n".encode("ascii"))
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
        os.replace(temp_path, uuid_path)
        temp_path = None
        directory_fd = os.open(base, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return value
    except OSError as e:
        raise RuntimeError(
            f"Unable to persist fs_native disk identity in {base!s}: {e}"
        ) from e
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def scan_existing_cache_entries(
    base_path: str,
) -> tuple[list[tuple[ObjectKey, int, int]], int]:
    """Scan the native connector's flat cache directory for recovery.

    Args:
        base_path: Root directory containing cache files.

    Returns:
        ``(entries, skipped_files)``. Each entry contains the key, size, and
        best-effort last-access time ``max(atime_ns, mtime_ns)``.

    Raises:
        RuntimeError: If the cache directory or a candidate file cannot be
            read.
    """
    base = Path(base_path)
    recovered: dict[ObjectKey, tuple[ObjectKey, int, int]] = {}
    skipped_files = 0

    try:
        with os.scandir(base) as entries:
            for entry in entries:
                try:
                    if not entry.name.endswith(".data"):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        skipped_files += 1
                        continue
                    key = filename_to_object_key(entry.name)
                    if key is None or object_key_to_filename(key) != entry.name:
                        skipped_files += 1
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A concurrent cleanup removed the entry after scandir.
                    continue
                except OSError as e:
                    raise RuntimeError(
                        f"Unable to inspect fs_native cache file {entry.path!r}: {e}"
                    ) from e

                last_access_ns = max(stat.st_atime_ns, stat.st_mtime_ns)
                recovered[key] = (key, stat.st_size, last_access_ns)
    except OSError as e:
        raise RuntimeError(
            f"Unable to scan fs_native cache directory {base!s}: {e}"
        ) from e

    return list(recovered.values()), skipped_files


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
        relative_tmp_path = Path(relative_tmp_dir)
        if relative_tmp_path.is_absolute() or ".." in relative_tmp_path.parts:
            raise ValueError(
                "relative_tmp_dir must stay within the fs_native base_path"
            )
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb
        self.placement_id = None

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
    config.placement_id = get_or_create_disk_uuid(config.base_path)
    scanned_entries, skipped_files = scan_existing_cache_entries(config.base_path)
    recovered_entries = [
        RecoveredL2Entry(
            key=key,
            size_bytes=size,
            last_access_ns=last_access_ns,
        )
        for key, size, last_access_ns in scanned_entries
    ]

    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    try:
        adapter = NativeConnectorL2Adapter(
            native_client,
            max_capacity_gb=config.max_capacity_gb,
            type_name="FSNativeL2Adapter",
            extra_status={
                "base_path": config.base_path,
                "placement_id": config.placement_id,
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
    logger.info(
        "Created FS native L2 adapter: %s (placement_id=%s, workers=%d, "
        "odirect=%s, read_ahead=%s, recovered_keys=%d, "
        "recovered_bytes=%d, skipped=%d)",
        config.base_path,
        config.placement_id,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        len(recovered_entries),
        sum(entry.size_bytes for entry in recovered_entries),
        skipped_files,
    )
    return adapter


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

"""
3FS native L2 adapter config and factory.

Backed by the native C++ 3FS connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
        L2AdapterListener,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
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
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)

logger = init_logger(__name__)


class Hf3fsL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++ 3FS connector.

    Fields:
    - mount_point: 3FS mount point directory (required).
    - base_paths: Comma-separated subdirectories under mount_point (required).
    - num_workers: C++ worker threads for I/O (default 4).
    - ior_entries: Max concurrent requests per Ior (default 256, range [128, 1024]).
    - io_depth: Batch control parameter (default 0, range [-128, 128]).
    - numa_id: NUMA node ID (default -1 = current node).
    - iov_size: Per-thread I/O buffer in bytes (default 200MB, range [100MB, 2GB]).
    - enable_key_buffer (bool): Enable in-memory key buffer for accelerated
        exists lookups. When enabled, do_batch_exists() uses a pre-scanned
        hash set instead of filesystem calls. Default: True.
    - per_op_workers: Optional dict mapping lane keys
                (``"lookup"``, ``"retrieve"``, ``"store"``,
                ``"delete"``) to dedicated worker counts.  Ops
                not mentioned use the shared ``num_workers`` pool.
    - max_capacity_gb: declared L2 capacity in GiB across all base_paths,
        used for usage accounting. ``0`` (default) disables it. A value
        ``> 0`` does **not** bound 3FS usage on its own -- watermark
        eviction runs only when the adapter spec also carries an
        ``eviction`` block, e.g.
        ``{"eviction": {"eviction_policy": "LRU"}}``. Without one,
        files accumulate past the declared capacity.
    """

    def __init__(
        self,
        mount_point: str,
        base_paths: str,
        num_workers: int = 4,
        ior_entries: int = 256,
        io_depth: int = 0,
        numa_id: int = -1,
        iov_size: int = 209715200,
        enable_key_buffer: bool = True,
        per_op_workers: dict[str, int] | None = None,
        max_capacity_gb: float = 0,
    ):
        self.mount_point = mount_point
        self.base_paths = base_paths
        self.num_workers = L2AdapterConfigBase._validate_num_workers(num_workers)
        self.ior_entries = ior_entries
        self.io_depth = io_depth
        self.numa_id = numa_id
        self.iov_size = iov_size
        self.enable_key_buffer = enable_key_buffer
        self.per_op_workers = L2AdapterConfigBase._validate_per_op_workers(
            per_op_workers
        )
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "Hf3fsL2AdapterConfig":
        mount_point = d.get("mount_point")
        if not isinstance(mount_point, str) or not mount_point:
            raise ValueError("mount_point must be a non-empty string")

        base_paths = d.get("base_paths")
        if not isinstance(base_paths, str) or not base_paths:
            raise ValueError("base_paths must be a non-empty string")

        # Validate paths exist and are directories
        try:
            mp_path = Path(mount_point).resolve()
            if not mp_path.exists():
                raise ValueError(f"mount_point does not exist: {mount_point}")
            if not mp_path.is_dir():
                raise ValueError(f"mount_point is not a directory: {mount_point}")

            # Validate each base_path is a subdirectory of mount_point
            paths = [p.strip() for p in base_paths.split(",")]
            for path_str in paths:
                if not path_str:
                    raise ValueError("Empty path in base_paths")
                path = Path(path_str).resolve()
                try:
                    path.relative_to(mp_path)
                except ValueError as err:
                    raise ValueError(
                        f"base_path '{path_str}' is not a subdirectory of "
                        f"mount_point '{mount_point}'"
                    ) from err
                if not path.exists():
                    raise ValueError(f"base_path does not exist: {path_str}")
                if not path.is_dir():
                    raise ValueError(f"base_path is not a directory: {path_str}")
        except Exception as e:
            raise ValueError(f"Invalid path configuration: {e}") from e

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        ior_entries = d.get("ior_entries", 256)
        if not isinstance(ior_entries, int) or not (128 <= ior_entries <= 1024):
            raise ValueError("ior_entries must be in range [128, 1024]")

        io_depth = d.get("io_depth", 0)
        if not isinstance(io_depth, int) or not (-128 <= io_depth <= 128):
            raise ValueError("io_depth must be in range [-128, 128]")

        numa_id = d.get("numa_id", -1)
        if not isinstance(numa_id, int):
            raise ValueError("numa_id must be an integer")

        iov_size = d.get("iov_size", 209715200)
        if not isinstance(iov_size, int) or not (104857600 <= iov_size <= 2147483648):
            raise ValueError("iov_size must be in range [100MB, 2GB]")

        enable_key_buffer = d.get("enable_key_buffer", True)
        if not isinstance(enable_key_buffer, bool):
            raise ValueError("enable_key_buffer must be a boolean")

        per_op_workers = L2AdapterConfigBase._parse_per_op_workers_from_dict(d)

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")
        if max_capacity_gb > 0 and d.get("eviction") is None:
            logger.warning(
                "hf3fs: max_capacity_gb=%s is declared for %s but no "
                "'eviction' block is configured, so it is used for usage "
                "accounting only and will NOT bound 3FS usage. Add an "
                "eviction block to enforce it, e.g. "
                '"eviction": {"eviction_policy": "LRU"}.',
                max_capacity_gb,
                base_paths,
            )

        return cls(
            mount_point=mount_point,
            base_paths=base_paths,
            num_workers=num_workers,
            ior_entries=ior_entries,
            io_depth=io_depth,
            numa_id=numa_id,
            iov_size=iov_size,
            enable_key_buffer=enable_key_buffer,
            per_op_workers=per_op_workers,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Hf3fs L2 adapter config fields:\n"
            "- mount_point (str): 3FS mount point directory (required)\n"
            "- base_paths (str): Comma-separated subdirectories under "
            "mount_point (required)\n"
            "- num_workers (int): C++ worker threads (default 4, >0)\n"
            "- ior_entries (int): Max concurrent requests per Ior "
            "(default 256, range [128, 1024])\n"
            "- io_depth (int): Batch control parameter "
            "(default 0, range [-128, 128])\n"
            "- numa_id (int): NUMA node ID (default -1 = current node)\n"
            "- iov_size (int): Per-thread I/O buffer in bytes "
            "(default 200MB, range [100MB, 2GB])\n"
            "- enable_key_buffer (bool): Enable in-memory key buffer for "
            "accelerated exists lookups (default True)\n"
            "- per_op_workers (dict[str, int] | None): Optional dict mapping "
            "lane keys to dedicated worker counts. Valid keys: lookup, "
            "retrieve, store, delete. Ops not mentioned use the shared "
            "num_workers pool.\n"
            "- max_capacity_gb (float): declared L2 capacity in GiB across "
            "all base_paths for usage accounting (default 0 = disabled). "
            "Does not bound 3FS usage by itself; add an 'eviction' block "
            "to enable watermark eviction, e.g. "
            '"eviction": {"eviction_policy": "LRU", '
            '"trigger_watermark": 0.8, "eviction_ratio": 0.2}'
        )


class Hf3fsNativeL2Adapter(NativeConnectorL2Adapter):
    """hf3fs adapter that pre-seeds eviction usage accounting.

    When capacity accounting is enabled (``max_capacity_gb > 0``), the
    adapter hydrates its usage from the ``.data`` files already present
    on the 3FS ``base_paths`` at construction time."""

    def __init__(
        self,
        native_client: Any,
        memory_tags: MemoryTags = MemoryTags(
            MemorySegment.STORAGE, MemoryResidency.REMOTE
        ),
        *,
        max_capacity_gb: float = 0,
        type_name: str = "Hf3fsL2Adapter",
        extra_status: dict[str, Any] | None = None,
        pad_buffers_to_alignment: bool = False,
    ) -> None:
        """Wrap a 3FS native client and defer usage hydration.

        Hydration of pre-existing files happens in ``register_listener``
        (as in ``RawBlockL2Adapter``), not here, so the adapter is
        fully-wired before usage accounting is seeded.

        Args:
            native_client: Pybind-wrapped ``LMCacheHf3fsClient``.
            memory_tags: Memory tags describing the 3FS storage medium;
                defaults to remote storage.
            max_capacity_gb: Declared capacity in GiB; ``> 0`` triggers
                usage hydration from existing files at startup.
            type_name: Stable adapter type label for status reporting.
            extra_status: Extra key-value pairs merged into
                ``report_status()``.
            pad_buffers_to_alignment: See
                ``NativeConnectorL2Adapter.__init__``.
        """
        super().__init__(
            native_client,
            memory_tags=memory_tags,
            max_capacity_gb=max_capacity_gb,
            type_name=type_name,
            extra_status=extra_status,
            pad_buffers_to_alignment=pad_buffers_to_alignment,
        )
        self._seed_usage_from_startup_scan()

    def _seed_usage_from_startup_scan(self) -> None:
        """Charge pre-existing 3FS files to usage accounting (once)."""
        try:
            existing = self._client.get_existing_keys_with_sizes()
        except Exception as e:
            logger.warning("Hf3fs usage hydration failed to read existing keys: %s", e)
            return

        if not existing:
            return

        keys = [key for key, _ in existing]
        sizes = [int(size) for _, size in existing]

        self._scanned_keys: list[ObjectKey] = keys
        self._scanned_sizes: list[int] = sizes

        # Record per-key sizes so a later ``delete()`` debits the correct
        # byte size (the native delete completion carries no sizes).
        with self._lock:
            for key, size in zip(keys, sizes, strict=True):
                if key not in self._key_sizes:
                    self._key_sizes[key] = size

        # Aggregate per-salt deltas before touching
        # ``_bytes_by_cache_salt``, mirroring ``_notify_keys_stored``.
        delta: dict[str, int] = {}
        total_delta = 0
        for key, size in zip(keys, sizes, strict=True):
            delta[key.cache_salt] = delta.get(key.cache_salt, 0) + size
            total_delta += size
        with self._usage_lock:
            self._total_bytes_used += total_delta
            for salt, d in delta.items():
                self._bytes_by_cache_salt[salt] = (
                    self._bytes_by_cache_salt.get(salt, 0) + d
                )

    def register_listener(self, listener: "L2AdapterListener") -> None:
        """Register a listener and seed it with keys already in 3FS.

        Each listener is seeded once with the pre-existing keys (as if
        they were just stored) so eviction policies learn the keys and their recency.

        Args:
            listener: The listener to register; it immediately receives
                the pre-existing keys as if they were just stored.
        """
        super().register_listener(listener)
        keys = getattr(self, "_scanned_keys", None)
        if not keys:
            return
        try:
            listener.on_l2_keys_stored(keys, self._scanned_sizes)
        except Exception as e:
            logger.warning("Hf3fs usage hydration listener bootstrap failed: %s", e)


def _create_hf3fs_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    """Create Hf3FS native L2 adapter."""
    try:
        # First Party
        from lmcache.lmcache_hf3fs import (
            LMCacheHf3fsClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Hf3fs native L2 adapter requires the C++ extension. "
            "Build with: pip install -e ."
        ) from e

    try:
        # Lazy import to keep the C++ extension import out of the hot path.
        # First Party
        from lmcache.lmcache_hf3fs import (
            LMCacheHf3fsClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Hf3fs native L2 adapter requires the C++ extension. "
            "Build with: pip install -e ."
        ) from e

    assert isinstance(config, Hf3fsL2AdapterConfig)
    native_client = LMCacheHf3fsClient(
        config.mount_point,
        config.base_paths,
        config.num_workers,
        config.ior_entries,
        config.io_depth,
        config.numa_id,
        config.iov_size,
        time_out=200,
        enable_key_buffer=config.enable_key_buffer,
        collect_file_sizes=(config.max_capacity_gb > 0),
        per_op_workers=config.per_op_workers,
    )
    logger.info(
        "Created Hf3fs native L2 adapter: %s (workers=%d, paths=%d, "
        "capacity_gb=%s, key_buffer=%s)",
        config.base_paths,
        config.num_workers,
        len(config.base_paths.split(",")),
        config.max_capacity_gb,
        config.enable_key_buffer,
    )
    return Hf3fsNativeL2Adapter(
        native_client,
        MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE),
        type_name="Hf3fsL2Adapter",
        max_capacity_gb=config.max_capacity_gb,
        extra_status={
            "mount_point": config.mount_point,
            "base_paths": config.base_paths,
            "num_workers": config.num_workers,
            "ior_entries": config.ior_entries,
            "io_depth": config.io_depth,
            "per_op_workers": config.per_op_workers,
            "max_capacity_gb": config.max_capacity_gb,
        },
    )


register_l2_adapter_type("hf3fs", Hf3fsL2AdapterConfig)
register_l2_adapter_factory("hf3fs", _create_hf3fs_l2_adapter)

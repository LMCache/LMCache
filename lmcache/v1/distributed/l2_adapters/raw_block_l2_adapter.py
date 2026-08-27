# SPDX-License-Identifier: Apache-2.0
"""
Raw-block L2 adapter for LMCache MP mode.

Uses RawBlockCore as the synchronous durable engine and adapts it to the
non-blocking L2AdapterInterface contract with separate eventfds for store,
lookup, and load.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast
import threading

if TYPE_CHECKING:
    from lmcache.lmcache_native import Bitmap
    from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import EventNotifier, create_event_notifier
from lmcache.v1.storage_backend.raw_block import (
    DEFAULT_IOURING_QUEUE_DEPTH,
    RawBlockCore,
    RawBlockCoreConfig,
    decode_object_key,
    encode_object_key,
    normalize_raw_block_io_engine,
    normalize_raw_block_placement_ids,
    validate_raw_block_io_options,
)

logger = init_logger(__name__)

RawBlockStoreTaskResult = tuple[
    bool,
    list[ObjectKey],
    list[int],
]

_FDP_DATA_PLACEMENT_POLICY_NONE = "none"
_FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_PREFIX = "cache_salt_prefix"
_FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_RANK = "cache_salt_rank"
_FDP_DATA_PLACEMENT_POLICIES = frozenset(
    {
        _FDP_DATA_PLACEMENT_POLICY_NONE,
        _FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_PREFIX,
        _FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_RANK,
    }
)
_FDP_SLOT_REUSE_POLICY_NONE = "none"
_FDP_SLOT_REUSE_POLICY_PID_AFFINITY = "pid_affinity"
_FDP_SLOT_REUSE_POLICIES = frozenset(
    {
        _FDP_SLOT_REUSE_POLICY_NONE,
        _FDP_SLOT_REUSE_POLICY_PID_AFFINITY,
    }
)
_FDP_CACHE_SALT_BUCKET_SEPARATOR = ":"
_FDP_CACHE_SALT_FALLBACK_BUCKET_SAMPLE_LIMIT = 64


def _normalize_fdp_placement_ids(
    placement_ids: Optional[list[int]],
) -> Optional[list[int]]:
    """Validate optional FDP placement identifiers from user configuration."""
    if placement_ids is None:
        return None

    try:
        normalized_placement_ids = normalize_raw_block_placement_ids(
            placement_ids,
            len(placement_ids),
            field_name="fdp_placement_ids",
            allow_none=False,
        )
    except ValueError as e:
        if "placement identifier 0" in str(e):
            logger.warning(
                "raw_block FDP placement identifier 0 is reserved for default "
                "NVMe writes and cannot be configured explicitly"
            )
            raise ValueError("fdp_placement_ids must not contain 0") from e
        raise
    normalized = cast(list[int], normalized_placement_ids)

    if len(normalized) != len(set(normalized)):
        raise ValueError("fdp_placement_ids must not contain duplicates")
    if not normalized:
        raise ValueError("fdp_placement_ids must not be empty")
    return normalized


def _exclude_meta_checkpoint_placement_id(
    placement_ids: list[int],
    meta_checkpoint_placement_id: int | None,
) -> list[int]:
    """Return data placement IDs that exclude the metadata checkpoint ID."""
    if meta_checkpoint_placement_id is None:
        return placement_ids
    return [pid for pid in placement_ids if pid != meta_checkpoint_placement_id]


def _validate_disjoint_fdp_placement_ids(
    *,
    fdp_placement_ids: Optional[list[int]],
    meta_checkpoint_placement_id: int | None,
) -> None:
    """Validate that metadata and KV data FDP placement IDs do not overlap."""
    if meta_checkpoint_placement_id is None or fdp_placement_ids is None:
        return
    if meta_checkpoint_placement_id in fdp_placement_ids:
        raise ValueError(
            "meta_checkpoint_placement_id must not overlap with fdp_placement_ids"
        )


def _default_fdp_data_placement_policy(*, fdp_enabled: bool) -> str:
    """Return the default FDP KV data placement policy."""
    if fdp_enabled:
        return _FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_PREFIX
    return _FDP_DATA_PLACEMENT_POLICY_NONE


def _normalize_fdp_data_placement_policy(
    policy: Any,
    *,
    fdp_enabled: bool,
) -> str:
    """Normalize the FDP KV data placement policy."""
    if policy is None or policy == "":
        return _default_fdp_data_placement_policy(fdp_enabled=fdp_enabled)

    normalized = str(policy).lower()
    if normalized not in _FDP_DATA_PLACEMENT_POLICIES:
        allowed = ", ".join(sorted(_FDP_DATA_PLACEMENT_POLICIES))
        raise ValueError(f"fdp_data_placement_policy must be one of: {allowed}")
    if normalized != _FDP_DATA_PLACEMENT_POLICY_NONE and not fdp_enabled:
        raise ValueError("fdp_data_placement_policy requires fdp_enabled=true")
    return normalized


def _normalize_fdp_slot_reuse_policy(
    policy: Any,
    *,
    fdp_enabled: bool,
) -> str:
    """Normalize the FDP free-slot reuse policy."""
    if policy is None or policy == "":
        if fdp_enabled:
            return _FDP_SLOT_REUSE_POLICY_PID_AFFINITY
        return _FDP_SLOT_REUSE_POLICY_NONE

    normalized = str(policy).lower()
    if normalized not in _FDP_SLOT_REUSE_POLICIES:
        allowed = ", ".join(sorted(_FDP_SLOT_REUSE_POLICIES))
        raise ValueError(f"fdp_slot_reuse_policy must be one of: {allowed}")
    if normalized != _FDP_SLOT_REUSE_POLICY_NONE and not fdp_enabled:
        raise ValueError("fdp_slot_reuse_policy requires fdp_enabled=true")
    return normalized


def _cache_salt_to_fdp_bucket(cache_salt: str) -> str | None:
    """Return the case-insensitive FDP bucket name for a cache salt."""
    if not cache_salt or _FDP_CACHE_SALT_BUCKET_SEPARATOR not in cache_salt:
        return None
    bucket = cache_salt.split(_FDP_CACHE_SALT_BUCKET_SEPARATOR, 1)[0].casefold()
    return bucket or None


def _detect_node_gpu_count() -> int:
    """Return the visible GPU count for cache-salt rank placement quotas."""
    try:
        if not torch_dev.is_available():
            return 0
        return max(0, int(torch_dev.device_count()))
    except Exception as e:
        logger.warning("RawBlockL2Adapter could not detect GPU count: %s", e)
        return 0


def _local_rank_from_kv_rank(kv_rank: int) -> int:
    """Extract the local rank encoded in an ObjectKey KV rank."""
    return int(kv_rank) & 0xFF


def _make_bitmap(size: int) -> "Bitmap":
    # First Party
    from lmcache.lmcache_native import Bitmap

    return Bitmap(size)


class RawBlockL2AdapterConfig(L2AdapterConfigBase):
    """Configuration object for the built-in raw-block MP L2 adapter."""

    def __init__(
        self,
        *,
        device_path: str,
        slot_bytes: int,
        capacity_bytes: int = 0,
        use_odirect: bool = True,
        block_align: int = 4096,
        header_bytes: int = 4096,
        meta_total_bytes: int = 256 * 1024 * 1024,
        meta_magic: str = "LMCIDX01",
        meta_version: int = 1,
        meta_checkpoint_interval_sec: int = 60,
        meta_idle_quiet_ms: int = 100,
        meta_enable_periodic: bool = True,
        load_checkpoint_on_init: bool = True,
        meta_verify_on_load: bool = True,
        enable_zero_copy: bool = True,
        io_engine: str = "posix",
        iouring_queue_depth: int = DEFAULT_IOURING_QUEUE_DEPTH,
        use_uring_cmd: bool = False,
        max_data_transfer_size: int = 0,
        fdp_enabled: bool = False,
        fdp_placement_ids: Optional[list[int]] = None,
        fdp_data_placement_policy: str | None = None,
        fdp_slot_reuse_policy: str | None = None,
        meta_checkpoint_placement_id: int | None = None,
        num_store_workers: int = 2,
        num_lookup_workers: int = 1,
        num_load_workers: int = 4,
    ):
        """Initialize raw-block MP adapter configuration.

        Args:
            device_path: Raw device path or pre-sized file path used for L2.
            slot_bytes: Fixed data-slot size in bytes.
            capacity_bytes: Optional cap on usable bytes; zero uses device size.
            use_odirect: Whether to open the raw path with O_DIRECT.
            block_align: Required power-of-two block alignment in bytes.
            header_bytes: Per-slot header reservation in bytes.
            meta_total_bytes: Reserved metadata checkpoint region size.
            meta_magic: Eight-byte ASCII metadata checkpoint magic.
            meta_version: Metadata checkpoint version.
            meta_checkpoint_interval_sec: Periodic checkpoint interval.
            meta_idle_quiet_ms: Quiet period before periodic checkpoints.
            meta_enable_periodic: Whether to run the checkpoint thread.
            load_checkpoint_on_init: Whether to load existing checkpoint metadata.
            meta_verify_on_load: Whether recovery verifies slot headers.
            enable_zero_copy: Whether to use aligned direct-buffer I/O.
            io_engine: Raw-block I/O engine: ``"posix"`` or ``"io_uring"``.
            iouring_queue_depth: Queue depth for the Rust io_uring engine.
            use_uring_cmd: Whether to use NVMe io_uring_cmd passthrough.
            max_data_transfer_size: Max data transfer size for a single request.
            fdp_enabled: Enable NVMe Flexible Data Placement discovery and
                non-zero placement-identifier registration. ``cache_salt``
                values with ``":"`` bucket prefixes use FDP placement by
                default.
            fdp_placement_ids: Optional non-zero FDP placement identifier list
                for KV data writes. If omitted, all device-reported identifiers
                except 0 and the metadata checkpoint identifier are registered.
            fdp_data_placement_policy: KV data FDP placement policy. ``None``
                defaults to ``"cache_salt_prefix"`` when FDP is enabled and
                ``"none"`` otherwise.
            fdp_slot_reuse_policy: Evicted-slot reuse policy. ``"pid_affinity"``
                prefers a slot last assigned to the same placement identifier
                before falling back to the global free-slot pool. ``None``
                defaults to ``"pid_affinity"`` when FDP is enabled and ``"none"``
                otherwise.
            meta_checkpoint_placement_id: Optional non-zero placement identifier
                for metadata checkpoint writes.
            num_store_workers: Number of store worker threads.
            num_lookup_workers: Number of lookup worker threads.
            num_load_workers: Number of load worker threads.
        """
        super().__init__()
        self.device_path = device_path
        self.slot_bytes = int(slot_bytes)
        self.capacity_bytes = int(capacity_bytes)
        self.use_odirect = bool(use_odirect)
        self.block_align = int(block_align)
        self.header_bytes = int(header_bytes)
        self.meta_total_bytes = int(meta_total_bytes)
        self.meta_magic = meta_magic
        self.meta_version = int(meta_version)
        self.meta_checkpoint_interval_sec = int(meta_checkpoint_interval_sec)
        self.meta_idle_quiet_ms = int(meta_idle_quiet_ms)
        self.meta_enable_periodic = bool(meta_enable_periodic)
        self.load_checkpoint_on_init = bool(load_checkpoint_on_init)
        self.meta_verify_on_load = bool(meta_verify_on_load)
        self.enable_zero_copy = bool(enable_zero_copy)
        self.io_engine = normalize_raw_block_io_engine(io_engine)
        self.iouring_queue_depth = int(iouring_queue_depth)
        validate_raw_block_io_options(
            iouring_queue_depth=self.iouring_queue_depth,
        )
        self.use_uring_cmd = bool(use_uring_cmd)
        self.max_data_transfer_size = int(max_data_transfer_size)
        self.fdp_enabled = bool(fdp_enabled)
        if self.fdp_enabled and (
            self.io_engine != "io_uring" or not self.use_uring_cmd
        ):
            raise ValueError(
                "fdp_enabled requires io_engine='io_uring' and use_uring_cmd=true"
            )
        self.fdp_placement_ids = (
            _normalize_fdp_placement_ids(fdp_placement_ids)
            if self.fdp_enabled
            else None
        )
        self.fdp_data_placement_policy = _normalize_fdp_data_placement_policy(
            fdp_data_placement_policy,
            fdp_enabled=self.fdp_enabled,
        )
        self.fdp_slot_reuse_policy = _normalize_fdp_slot_reuse_policy(
            fdp_slot_reuse_policy,
            fdp_enabled=self.fdp_enabled,
        )
        if meta_checkpoint_placement_id is not None and (
            self.io_engine != "io_uring" or not self.use_uring_cmd
        ):
            raise ValueError(
                "meta_checkpoint_placement_id requires "
                "io_engine='io_uring' and use_uring_cmd=true"
            )
        self.meta_checkpoint_placement_id = normalize_raw_block_placement_ids(
            [meta_checkpoint_placement_id],
            1,
            field_name="meta_checkpoint_placement_id",
        )[0]
        _validate_disjoint_fdp_placement_ids(
            fdp_placement_ids=self.fdp_placement_ids,
            meta_checkpoint_placement_id=self.meta_checkpoint_placement_id,
        )
        self.num_store_workers = int(num_store_workers)
        self.num_lookup_workers = int(num_lookup_workers)
        self.num_load_workers = int(num_load_workers)

    @classmethod
    def from_dict(cls, d: dict) -> "RawBlockL2AdapterConfig":
        """Build and validate a raw-block config from ``--l2-adapter`` JSON."""
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path:
            raise ValueError("device_path must be a non-empty string")
        if "per_tp_device_paths" in d:
            raise ValueError(
                "per_tp_device_paths is not supported in MP raw_block mode"
            )
        if not bool(d.get("persist_enabled", True)):
            raise ValueError("raw_block requires persist_enabled=true")

        slot_bytes = d.get("slot_bytes")
        if not isinstance(slot_bytes, int) or slot_bytes <= 0:
            raise ValueError("slot_bytes must be a positive integer")

        block_align = int(d.get("block_align", 4096))
        header_bytes = int(d.get("header_bytes", 4096))
        meta_total_bytes = int(d.get("meta_total_bytes", 256 * 1024 * 1024))
        capacity_bytes = int(d.get("capacity_bytes", 0))
        io_engine = normalize_raw_block_io_engine(
            d.get("io_engine"),
            use_iouring=d.get("use_iouring"),
            use_uring=d.get("use_uring"),
        )
        iouring_queue_depth = int(
            d.get("iouring_queue_depth", DEFAULT_IOURING_QUEUE_DEPTH)
        )
        use_uring_cmd = bool(d.get("use_uring_cmd", False))
        max_data_transfer_size = int(d.get("max_data_transfer_size", 0))
        fdp_enabled = bool(d.get("fdp_enabled", False))
        meta_checkpoint_placement_id = d.get("meta_checkpoint_placement_id")
        raw_fdp_placement_ids = d.get("fdp_placement_ids")
        if raw_fdp_placement_ids is not None and not isinstance(
            raw_fdp_placement_ids, list
        ):
            raise ValueError("fdp_placement_ids must be a list")
        fdp_placement_ids = raw_fdp_placement_ids if fdp_enabled else None
        fdp_data_placement_policy = d.get("fdp_data_placement_policy")
        fdp_slot_reuse_policy = d.get("fdp_slot_reuse_policy")

        if block_align <= 0 or (block_align & (block_align - 1)) != 0:
            raise ValueError(f"block_align must be a power of 2, got {block_align}")
        if slot_bytes % block_align != 0:
            raise ValueError("slot_bytes must be a multiple of block_align")
        if header_bytes % block_align != 0:
            raise ValueError("header_bytes must be a multiple of block_align")
        if meta_total_bytes % block_align != 0:
            raise ValueError("meta_total_bytes must be a multiple of block_align")
        if slot_bytes < header_bytes + 1:
            raise ValueError("slot_bytes must be >= header_bytes + 1")
        if capacity_bytes > 0 and capacity_bytes <= meta_total_bytes:
            raise ValueError("capacity_bytes must leave space for at least one slot")
        validate_raw_block_io_options(
            iouring_queue_depth=iouring_queue_depth,
        )
        if use_uring_cmd and io_engine != "io_uring":
            raise ValueError("use_uring_cmd requires io_uring io_engine")
        if fdp_enabled and (io_engine != "io_uring" or not use_uring_cmd):
            raise ValueError(
                "fdp_enabled requires io_engine='io_uring' and use_uring_cmd=true"
            )
        if meta_checkpoint_placement_id is not None and (
            io_engine != "io_uring" or not use_uring_cmd
        ):
            raise ValueError(
                "meta_checkpoint_placement_id requires "
                "io_engine='io_uring' and use_uring_cmd=true"
            )

        worker_defaults = {
            "num_store_workers": 2,
            "num_lookup_workers": 1,
            "num_load_workers": 4,
        }
        worker_counts: dict[str, int] = {}
        for field_name, default in worker_defaults.items():
            value = int(d.get(field_name, default))
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0")
            worker_counts[field_name] = value

        return cls(
            device_path=device_path,
            slot_bytes=slot_bytes,
            capacity_bytes=capacity_bytes,
            use_odirect=bool(d.get("use_odirect", True)),
            block_align=block_align,
            header_bytes=header_bytes,
            meta_total_bytes=meta_total_bytes,
            meta_magic=str(d.get("meta_magic", "LMCIDX01")),
            meta_version=int(d.get("meta_version", 1)),
            meta_checkpoint_interval_sec=int(d.get("meta_checkpoint_interval_sec", 60)),
            meta_idle_quiet_ms=int(d.get("meta_idle_quiet_ms", 100)),
            meta_enable_periodic=bool(d.get("meta_enable_periodic", True)),
            load_checkpoint_on_init=bool(d.get("load_checkpoint_on_init", True)),
            meta_verify_on_load=bool(d.get("meta_verify_on_load", True)),
            enable_zero_copy=bool(d.get("enable_zero_copy", True)),
            io_engine=io_engine,
            iouring_queue_depth=iouring_queue_depth,
            use_uring_cmd=use_uring_cmd,
            max_data_transfer_size=max_data_transfer_size,
            fdp_enabled=fdp_enabled,
            fdp_placement_ids=fdp_placement_ids,
            fdp_data_placement_policy=fdp_data_placement_policy,
            fdp_slot_reuse_policy=fdp_slot_reuse_policy,
            meta_checkpoint_placement_id=meta_checkpoint_placement_id,
            num_store_workers=worker_counts["num_store_workers"],
            num_lookup_workers=worker_counts["num_lookup_workers"],
            num_load_workers=worker_counts["num_load_workers"],
        )

    @classmethod
    def help(cls) -> str:
        """Return human-readable raw-block adapter configuration help."""
        return (
            "raw_block L2 adapter config fields:\n"
            "- device_path (str): raw device or file path (required)\n"
            "- slot_bytes (int): slot size in bytes, aligned to block_align "
            "(required)\n"
            "- capacity_bytes (int): optional usable capacity cap "
            "(default 0 = device size)\n"
            "- use_odirect (bool): enable O_DIRECT raw I/O (default true)\n"
            "- block_align (int): required power-of-two block alignment in "
            "bytes (default 4096)\n"
            "- header_bytes (int): per-slot header reservation (default 4096)\n"
            "- meta_total_bytes (int): reserved metadata checkpoint region "
            "(default 256MiB)\n"
            "- meta_magic (str): 8-byte metadata magic (default LMCIDX01)\n"
            "- meta_version (int): metadata version (default 1)\n"
            "- meta_checkpoint_interval_sec (int): periodic checkpoint interval "
            "(default 60)\n"
            "- meta_idle_quiet_ms (int): quiet period before checkpoint (default 100)\n"
            "- meta_enable_periodic (bool): enable periodic checkpointing "
            "(default true)\n"
            "- load_checkpoint_on_init (bool): load existing metadata checkpoint "
            "on startup (default true)\n"
            "- meta_verify_on_load (bool): validate slot headers on recovery "
            "(default true)\n"
            "- enable_zero_copy (bool): use aligned direct buffers when possible "
            "(default true)\n"
            "- io_engine (str): posix or io_uring (default posix)\n"
            "- iouring_queue_depth (int): Rust io_uring queue depth "
            f"(default {DEFAULT_IOURING_QUEUE_DEPTH})\n"
            "- use_uring_cmd (bool): enable NVMe io_uring_cmd path "
            "(default false, requires io_uring as the io_engine)\n"
            "- max_data_transfer_size (int): for a single I/O request "
            "(0: (default) auto detect limit splitting, > 0: explicit split, "
            "< 0: auto detect limit splitting)\n"
            "- fdp_enabled (bool): enable FDP discovery/registration; "
            "cache_salt values with ':' bucket prefixes are assigned to FDP "
            "placement identifiers by default when enabled (default false)\n"
            "- fdp_placement_ids (list[int]): non-zero FDP placement "
            "identifiers for KV data writes; omitted registers all "
            "device-reported non-zero identifiers except the metadata "
            "checkpoint identifier\n"
            "- fdp_data_placement_policy (str): none, cache_salt_prefix, or "
            "cache_salt_rank; defaults to cache_salt_prefix when "
            "fdp_enabled=true\n"
            "- fdp_slot_reuse_policy (str): none or pid_affinity; defaults to "
            "pid_affinity when fdp_enabled=true\n"
            "- meta_checkpoint_placement_id (int): non-zero FDP placement "
            "identifier for metadata checkpoints; requires io_uring_cmd\n"
            "- num_store_workers (int): store worker threads (default 2)\n"
            "- num_lookup_workers (int): lookup worker threads (default 1)\n"
            "- num_load_workers (int): load worker threads (default 4)"
        )

    def to_core_config(self) -> RawBlockCoreConfig:
        """Convert this adapter config to the shared RawBlockCore config."""
        return RawBlockCoreConfig(
            device_path=self.device_path,
            capacity_bytes=self.capacity_bytes,
            block_align=self.block_align,
            header_bytes=self.header_bytes,
            slot_bytes=self.slot_bytes,
            use_odirect=self.use_odirect,
            enable_zero_copy=self.enable_zero_copy,
            meta_total_bytes=self.meta_total_bytes,
            meta_magic=self.meta_magic.encode("ascii"),
            meta_version=self.meta_version,
            meta_checkpoint_interval_sec=self.meta_checkpoint_interval_sec,
            meta_idle_quiet_ms=self.meta_idle_quiet_ms,
            meta_enable_periodic=self.meta_enable_periodic,
            load_checkpoint_on_init=self.load_checkpoint_on_init,
            meta_verify_on_load=self.meta_verify_on_load,
            io_engine=self.io_engine,
            iouring_queue_depth=self.iouring_queue_depth,
            use_uring_cmd=self.use_uring_cmd,
            max_data_transfer_size=self.max_data_transfer_size,
            meta_checkpoint_placement_id=self.meta_checkpoint_placement_id,
            fdp_slot_affinity_enabled=(
                self.fdp_slot_reuse_policy == _FDP_SLOT_REUSE_POLICY_PID_AFFINITY
            ),
        )


class RawBlockL2Adapter(L2AdapterInterface):
    """MP L2 adapter that persists KV objects into raw-block slots."""

    def __init__(
        self,
        config: RawBlockL2AdapterConfig,
        l1_memory_desc: "Optional[L1MemoryDesc]" = None,
    ):
        """Initialize the MP raw-block L2 adapter.

        Args:
            config: Validated raw-block adapter configuration.
            l1_memory_desc: Optional L1 allocation descriptor used to validate
                O_DIRECT alignment compatibility.

        Raises:
            ValueError: If O_DIRECT is enabled and L1 alignment is insufficient.
            RuntimeError: If the shared core cannot open or recover the raw
                device.

        Notes:
            Resources created before an initialization failure are closed before
            the exception is re-raised.
        """
        super().__init__()
        if (
            (config.use_odirect or config.io_engine == "io_uring")
            and l1_memory_desc is not None
            and l1_memory_desc.align_bytes < config.block_align
        ):
            raise ValueError(
                "raw_block requires l1_align_bytes >= block_align when "
                "use_odirect=true or io_engine=io_uring"
            )

        self._closed = False
        self._core: RawBlockCore
        self._store_efd: EventNotifier | None = None
        self._lookup_efd: EventNotifier | None = None
        self._load_efd: EventNotifier | None = None
        self._store_pool: ThreadPoolExecutor
        self._lookup_pool: ThreadPoolExecutor
        self._load_pool: ThreadPoolExecutor
        self._fdp_lock = threading.Lock()
        self._fdp_data_placement_policy = config.fdp_data_placement_policy
        self._fdp_slot_reuse_policy = config.fdp_slot_reuse_policy
        self._fdp_cache_salt_bucket_placements: dict[str, int] = {}
        self._fdp_cache_salt_rank_placements: dict[str, dict[int, int]] = {}
        self._fdp_cache_salt_rank_fallback_buckets: set[str] = set()
        self._fdp_cache_salt_rank_max_placements_per_bucket = _detect_node_gpu_count()
        self._fdp_cache_salt_fallback_count = 0
        self._fdp_cache_salt_fallback_bucket_samples: set[str] = set()
        self._fdp_fallback_warning_emitted = False

        try:
            self._core = RawBlockCore(config.to_core_config(), key_namespace="object")
            self._fdp_enabled = bool(config.fdp_enabled)
            self._fdp_discovered_status: list[tuple[int, int]] = []
            self._fdp_placement_ids: list[int] = []
            if self._fdp_enabled:
                self._configure_fdp(config.fdp_placement_ids)
            if config.io_engine == "io_uring":
                logger.warning(
                    "RawBlockL2Adapter: MP raw_block uses io_uring without "
                    "fixed-buffer registration; zero-copy fixed buffers are "
                    "disabled unless registered by a future MP allocator path"
                )
            self._max_capacity_bytes = int(
                self._core.report_status().get("usable_capacity_bytes", 0)
            )
            self._seed_usage_from_core_snapshot()

            self._store_efd = create_event_notifier()
            self._lookup_efd = create_event_notifier()
            self._load_efd = create_event_notifier()

            self._store_pool = ThreadPoolExecutor(
                max_workers=config.num_store_workers,
                thread_name_prefix="rawblk-store",
            )
            self._lookup_pool = ThreadPoolExecutor(
                max_workers=config.num_lookup_workers,
                thread_name_prefix="rawblk-lookup",
            )
            self._load_pool = ThreadPoolExecutor(
                max_workers=config.num_load_workers,
                thread_name_prefix="rawblk-load",
            )
        except Exception:
            self._cleanup_after_init_failure()
            raise

        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0

        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._store_inflight_tasks: int = 0
        self._lookup_inflight_tasks: int = 0
        self._load_inflight_tasks: int = 0

    def get_store_event_fd(self) -> int:
        """Return the eventfd signaled when store tasks complete."""
        if self._store_efd is None:
            return -1
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled when lookup-and-lock tasks complete."""
        if self._lookup_efd is None:
            return -1
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled when load tasks complete."""
        if self._load_efd is None:
            return -1
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking raw-block store task.

        Args:
            keys: Object keys to persist.
            objects: Memory objects containing payloads for ``keys``.

        Returns:
            Task ID that can be observed through ``pop_completed_store_tasks``.

        Raises:
            ValueError: If either list is empty or the lengths differ.
        """
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._store_inflight_tasks += 1
        try:
            future = self._store_pool.submit(
                self._run_store_task, list(keys), list(objects)
            )
        except Exception:
            with self._lock:
                self._store_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_store_task, task_id))
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Drain and return completed store task results."""
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        """Submit a non-blocking lookup-and-lock task.

        Args:
            keys: Object keys to look up in raw-block L2.

        Returns:
            Task ID whose bitmap can be queried with
            ``query_lookup_and_lock_result``.

        Raises:
            ValueError: If ``keys`` is empty.
        """
        if not keys:
            raise ValueError("keys must be non-empty")
        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._lookup_inflight_tasks += 1
        try:
            future = self._lookup_pool.submit(self._run_lookup_task, list(keys))
        except Exception:
            with self._lock:
                self._lookup_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_lookup_task, task_id, len(keys)))
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed lookup bitmap if available."""
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release L2 locks acquired by lookup-and-lock."""
        encoded_keys = [encode_object_key(key).encoded for key in keys]
        self._core.unlock_many(encoded_keys)

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking raw-block load task.

        Args:
            keys: Object keys to load.
            objects: Caller-provided destination buffers.

        Returns:
            Task ID whose bitmap can be queried with ``query_load_result``.

        Raises:
            ValueError: If either list is empty or the lengths differ.
        """
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._get_next_task_id_locked()
            self._load_inflight_tasks += 1
        try:
            future = self._load_pool.submit(
                self._run_load_task, list(keys), list(objects)
            )
        except Exception:
            with self._lock:
                self._load_inflight_tasks -= 1
            raise
        future.add_done_callback(partial(self._finish_load_task, task_id, len(keys)))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed load bitmap if available."""
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete keys from raw-block L2 and notify listeners for removals."""
        encoded_keys = [encode_object_key(key).encoded for key in keys]
        metas = self._core.get_metadata_many(encoded_keys)
        deleted_bitmap = self._core.delete_many(encoded_keys, force=False)
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for key, meta, deleted in zip(keys, metas, deleted_bitmap, strict=False):
            if not deleted:
                continue
            deleted_keys.append(key)
            deleted_sizes.append(0 if meta is None else int(self._core.slot_bytes))
        if deleted_keys:
            try:
                self._notify_keys_deleted(deleted_keys, deleted_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter delete notification failed: %s", e)

    def register_listener(self, listener: "L2AdapterListener") -> None:
        """Register a listener and seed it with currently indexed keys."""
        super().register_listener(listener)
        keys = self._snapshot_indexed_object_keys()
        if not keys:
            return
        try:
            slot_bytes = int(self._core.slot_bytes)
            listener.on_l2_keys_stored(keys, [slot_bytes] * len(keys))
        except Exception as e:
            logger.warning(
                "RawBlockL2Adapter listener recovery bootstrap failed: %s", e
            )

    def close(self) -> None:
        """Wait for worker pools, close the core, and close eventfds."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._store_pool.shutdown(wait=True)
        self._lookup_pool.shutdown(wait=True)
        self._load_pool.shutdown(wait=True)

        self._core.close()

        with self._lock:
            store_efd = self._store_efd
            lookup_efd = self._lookup_efd
            load_efd = self._load_efd
            self._store_efd = None
            self._lookup_efd = None
            self._load_efd = None

        if store_efd is not None:
            store_efd.close()
        if lookup_efd is not None:
            lookup_efd.close()
        if load_efd is not None:
            load_efd.close()

    def report_status(self) -> dict:
        """Return adapter health, task counters, and core status."""
        core_status = self._core.report_status()
        with self._fdp_lock:
            fdp_cache_salt_bucket_placements = dict(
                self._fdp_cache_salt_bucket_placements
            )
            fdp_cache_salt_rank_placements = {
                bucket: dict(rank_placements)
                for bucket, rank_placements in (
                    self._fdp_cache_salt_rank_placements.items()
                )
            }
            fdp_cache_salt_fallback_count = self._fdp_cache_salt_fallback_count
            fdp_cache_salt_fallback_buckets = sorted(
                self._fdp_cache_salt_fallback_bucket_samples
            )
        with self._lock:
            return {
                "is_healthy": core_status.get("is_healthy", True) and not self._closed,
                "type": "RawBlockL2Adapter",
                "store_inflight_task_count": self._store_inflight_tasks,
                "lookup_inflight_task_count": self._lookup_inflight_tasks,
                "load_inflight_task_count": self._load_inflight_tasks,
                "fdp_enabled": self._fdp_enabled,
                "fdp_discovered_status": list(self._fdp_discovered_status),
                "fdp_placement_ids": list(self._fdp_placement_ids),
                "fdp_data_placement_policy": self._fdp_data_placement_policy,
                "fdp_slot_reuse_policy": self._fdp_slot_reuse_policy,
                "fdp_cache_salt_bucket_separator": _FDP_CACHE_SALT_BUCKET_SEPARATOR,
                "fdp_cache_salt_bucket_placements": (fdp_cache_salt_bucket_placements),
                "fdp_cache_salt_rank_placements": fdp_cache_salt_rank_placements,
                "fdp_cache_salt_rank_max_placements_per_bucket": (
                    self._fdp_cache_salt_rank_max_placements_per_bucket
                ),
                "fdp_cache_salt_fallback_count": fdp_cache_salt_fallback_count,
                "fdp_cache_salt_fallback_buckets": fdp_cache_salt_fallback_buckets,
                "completed_store_task_count": len(self._completed_store_tasks),
                "completed_lookup_task_count": len(self._completed_lookup_tasks),
                "completed_load_task_count": len(self._completed_load_tasks),
                "core": core_status,
            }

    def _configure_fdp(self, configured_ids: Optional[list[int]]) -> None:
        """Fetch and register FDP placement identifiers for this adapter."""
        try:
            discovered = self._core.fetch_fdp_status()
        except Exception as e:
            raise RuntimeError("raw_block FDP status query failed") from e
        if not discovered:
            raise RuntimeError(
                "raw_block FDP enabled but device returned no identifiers"
            )

        self._fdp_discovered_status = [
            (int(pid), int(ruhid)) for pid, ruhid in discovered
        ]
        discovered_ids = [pid for pid, _ in self._fdp_discovered_status]
        device_nonzero_ids = [pid for pid in discovered_ids if pid != 0]
        if not device_nonzero_ids:
            raise RuntimeError(
                "raw_block FDP enabled but device returned no non-zero identifiers"
            )

        meta_checkpoint_placement_id = self._core.meta_checkpoint_placement_id
        device_nonzero_set = set(device_nonzero_ids)
        if (
            meta_checkpoint_placement_id is not None
            and meta_checkpoint_placement_id not in device_nonzero_set
        ):
            raise RuntimeError(
                "raw_block metadata checkpoint placement identifier is not "
                "reported by device identifiers: "
                f"configured={meta_checkpoint_placement_id} "
                f"device={device_nonzero_ids}"
            )

        if configured_ids is not None:
            configured_set = set(configured_ids)
            if not configured_set.issubset(device_nonzero_set):
                raise RuntimeError(
                    "raw_block FDP placement identifier list is not reported by "
                    "device "
                    f"identifiers: configured={configured_ids} "
                    f"device={device_nonzero_ids}"
                )
            self._fdp_placement_ids = list(configured_ids)
        else:
            self._fdp_placement_ids = _exclude_meta_checkpoint_placement_id(
                device_nonzero_ids,
                meta_checkpoint_placement_id,
            )

        if not self._fdp_placement_ids:
            raise RuntimeError(
                "raw_block FDP enabled but no non-zero data placement "
                "identifiers remain after excluding metadata checkpoint placement"
            )

        logger.info(
            "RawBlockL2Adapter registered FDP placement identifiers: %s",
            self._fdp_placement_ids,
        )

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("RawBlockL2Adapter is closed")

    def _get_next_task_id_locked(self) -> L2TaskId:
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _assign_fdp_placement_ids(
        self,
        keys: list[ObjectKey],
    ) -> list[int | None] | None:
        """Return FDP placement identifiers for a store batch.

        cache_salt values containing ":" are grouped by a case-insensitive
        prefix before ":" and assigned exclusive FDP placement identifiers while
        IDs are available. Values without ":" fall back to no directive. When
        the ID pool is exhausted, writes fall back to no directive and a warning
        is emitted once.
        """
        if (
            not self._fdp_enabled
            or self._fdp_data_placement_policy == _FDP_DATA_PLACEMENT_POLICY_NONE
        ):
            return None
        if not self._fdp_placement_ids:
            raise RuntimeError("raw_block FDP placement identifiers are not configured")
        if (
            self._fdp_data_placement_policy
            == _FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_RANK
        ):
            return self._assign_fdp_cache_salt_rank_placement_ids(keys)
        if (
            self._fdp_data_placement_policy
            != _FDP_DATA_PLACEMENT_POLICY_CACHE_SALT_PREFIX
        ):
            raise RuntimeError(
                "raw_block FDP data placement policy is unsupported: "
                f"{self._fdp_data_placement_policy}"
            )

        placement_ids: list[int | None] = []
        for key in keys:
            bucket = _cache_salt_to_fdp_bucket(key.cache_salt)
            if bucket is None:
                placement_ids.append(None)
                continue
            placement_ids.append(self._get_or_assign_fdp_bucket_placement_id(bucket))

        if all(placement_id is None for placement_id in placement_ids):
            return None
        return placement_ids

    def _assign_fdp_cache_salt_rank_placement_ids(
        self,
        keys: list[ObjectKey],
    ) -> list[int | None] | None:
        """Return per-rank FDP placement IDs inside each cache_salt bucket."""
        placement_ids: list[int | None] = []
        for key in keys:
            bucket = _cache_salt_to_fdp_bucket(key.cache_salt)
            if bucket is None:
                placement_ids.append(None)
                continue
            placement_ids.append(
                self._get_or_assign_fdp_rank_placement_id(
                    bucket,
                    _local_rank_from_kv_rank(key.kv_rank),
                )
            )

        if all(placement_id is None for placement_id in placement_ids):
            return None
        return placement_ids

    def _get_or_assign_fdp_bucket_placement_id(self, bucket: str) -> int | None:
        """Return an exclusive placement identifier for a cache_salt bucket."""
        with self._fdp_lock:
            placement_id = self._fdp_cache_salt_bucket_placements.get(bucket)
            if placement_id is not None:
                return placement_id

            if len(self._fdp_cache_salt_bucket_placements) < len(
                self._fdp_placement_ids
            ):
                placement_id = self._fdp_placement_ids[
                    len(self._fdp_cache_salt_bucket_placements)
                ]
                self._fdp_cache_salt_bucket_placements[bucket] = placement_id
                logger.info(
                    "RawBlockL2Adapter assigned FDP placement identifier %d "
                    "to cache_salt bucket %r",
                    placement_id,
                    bucket,
                )
                return placement_id

            self._record_fdp_fallback_bucket_locked(bucket)
            if not self._fdp_fallback_warning_emitted:
                logger.warning(
                    "RawBlockL2Adapter has more cache_salt FDP buckets than "
                    "registered placement identifiers; writes for extra buckets "
                    "will use default NVMe placement without an FDP directive"
                )
                self._fdp_fallback_warning_emitted = True
            return None

    def _get_or_assign_fdp_rank_placement_id(
        self,
        bucket: str,
        rank: int,
    ) -> int | None:
        """Return an exclusive placement identifier for a bucket/rank stream."""
        with self._fdp_lock:
            if bucket in self._fdp_cache_salt_rank_fallback_buckets:
                self._record_fdp_fallback_bucket_locked(bucket)
                return None

            rank_placements = self._fdp_cache_salt_rank_placements.get(bucket)
            if rank_placements is not None:
                placement_id = rank_placements.get(rank)
                if placement_id is not None:
                    return placement_id
                if (
                    len(rank_placements)
                    >= self._fdp_cache_salt_rank_max_placements_per_bucket
                ):
                    self._record_fdp_fallback_bucket_locked(bucket)
                    self._emit_fdp_rank_quota_warning_locked()
                    return None
            elif self._fdp_cache_salt_rank_max_placements_per_bucket <= 0:
                self._fdp_cache_salt_rank_fallback_buckets.add(bucket)
                self._record_fdp_fallback_bucket_locked(bucket)
                self._emit_fdp_rank_quota_warning_locked()
                return None
            elif not self._has_available_fdp_placement_id_locked():
                self._fdp_cache_salt_rank_fallback_buckets.add(bucket)
                self._record_fdp_fallback_bucket_locked(bucket)
                self._emit_fdp_exhausted_warning_locked()
                return None
            else:
                rank_placements = {}
                self._fdp_cache_salt_rank_placements[bucket] = rank_placements

            if not self._has_available_fdp_placement_id_locked():
                self._record_fdp_fallback_bucket_locked(bucket)
                self._emit_fdp_exhausted_warning_locked()
                return None

            placement_id = self._fdp_placement_ids[
                self._fdp_assigned_rank_placement_count_locked()
            ]
            rank_placements[rank] = placement_id
            logger.info(
                "RawBlockL2Adapter assigned FDP placement identifier %d to "
                "cache_salt bucket %r rank %d",
                placement_id,
                bucket,
                rank,
            )
            return placement_id

    def _fdp_assigned_rank_placement_count_locked(self) -> int:
        """Return assigned cache_salt_rank placement count under FDP lock."""
        return sum(
            len(rank_placements)
            for rank_placements in self._fdp_cache_salt_rank_placements.values()
        )

    def _has_available_fdp_placement_id_locked(self) -> bool:
        """Return whether cache_salt_rank can allocate another placement ID."""
        return self._fdp_assigned_rank_placement_count_locked() < len(
            self._fdp_placement_ids
        )

    def _emit_fdp_exhausted_warning_locked(self) -> None:
        """Emit the shared FDP exhaustion warning once under FDP lock."""
        if self._fdp_fallback_warning_emitted:
            return
        logger.warning(
            "RawBlockL2Adapter has more cache_salt FDP buckets/ranks than "
            "registered placement identifiers; writes without assigned "
            "identifiers will use default NVMe placement without an FDP directive"
        )
        self._fdp_fallback_warning_emitted = True

    def _emit_fdp_rank_quota_warning_locked(self) -> None:
        """Emit the cache_salt_rank per-bucket quota warning once under FDP lock."""
        if self._fdp_fallback_warning_emitted:
            return
        logger.warning(
            "RawBlockL2Adapter cache_salt_rank bucket reached the node GPU "
            "count placement quota; extra ranks will use default NVMe placement "
            "without an FDP directive"
        )
        self._fdp_fallback_warning_emitted = True

    def _record_fdp_fallback_bucket_locked(self, bucket: str) -> None:
        """Record fallback telemetry while holding ``self._fdp_lock``."""
        self._fdp_cache_salt_fallback_count += 1
        if (
            len(self._fdp_cache_salt_fallback_bucket_samples)
            < _FDP_CACHE_SALT_FALLBACK_BUCKET_SAMPLE_LIMIT
        ):
            self._fdp_cache_salt_fallback_bucket_samples.add(bucket)

    def _seed_usage_from_core_snapshot(self) -> None:
        """Seed byte counters for entries recovered by RawBlockCore startup."""
        recovered_keys = self._snapshot_indexed_object_keys()
        if not recovered_keys:
            return

        slot_bytes = int(self._core.slot_bytes)
        total_delta = len(recovered_keys) * slot_bytes
        by_salt: dict[str, int] = {}
        for key in recovered_keys:
            by_salt[key.cache_salt] = by_salt.get(key.cache_salt, 0) + slot_bytes

        with self._usage_lock:
            self._total_bytes_used += total_delta
            for salt, delta in by_salt.items():
                self._bytes_by_cache_salt[salt] = (
                    self._bytes_by_cache_salt.get(salt, 0) + delta
                )

    def _snapshot_indexed_object_keys(self) -> list[ObjectKey]:
        """Return decoded ObjectKeys for all indexed raw-block entries."""
        keys: list[ObjectKey] = []
        for encoded_key in self._core.snapshot_indexed_keys():
            try:
                keys.append(decode_object_key(encoded_key))
            except Exception as e:
                logger.warning(
                    "RawBlockL2Adapter could not decode indexed key %r: %s",
                    encoded_key,
                    e,
                )
        return keys

    def _run_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> RawBlockStoreTaskResult:
        """Persist one submitted store batch in the worker pool.

        Args:
            keys: Object keys submitted for storage.
            objects: Payload buffers aligned with ``keys``.

        Returns:
            A 3-tuple containing:

            - task success for the whole batch
            - newly stored object keys
            - raw-block slot byte charges aligned with the newly stored keys
        """
        specs = [encode_object_key(key) for key in keys]
        placement_ids = self._assign_fdp_placement_ids(keys)
        put_result = self._core.put_many(specs, objects, placement_ids=placement_ids)
        stored_encoded = set(put_result.stored_keys)
        slot_bytes = int(self._core.slot_bytes)
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        for key, spec in zip(keys, specs, strict=False):
            if spec.encoded not in stored_encoded:
                continue
            stored_keys.append(key)
            stored_sizes.append(slot_bytes)
        return all(put_result.results), stored_keys, stored_sizes

    def _finish_store_task(
        self,
        task_id: L2TaskId,
        future: Future[RawBlockStoreTaskResult],
    ) -> None:
        success = False
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        bytes_transferred = 0
        try:
            success, stored_keys, stored_sizes = future.result()
            bytes_transferred = sum(stored_sizes)
        except Exception as e:
            logger.error("RawBlockL2Adapter store task %d failed: %s", task_id, e)
        with self._lock:
            self._store_inflight_tasks -= 1
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )
            event_fd = self._store_efd
        if stored_keys:
            try:
                self._notify_keys_stored(stored_keys, stored_sizes)
            except Exception as e:
                logger.warning("RawBlockL2Adapter store notification failed: %s", e)
        self._signal_event_fd(event_fd)

    def _run_lookup_task(self, keys: list[ObjectKey]) -> Bitmap:
        specs = [encode_object_key(key) for key in keys]
        exists = self._core.exists_many([spec.encoded for spec in specs], lock=True)
        bitmap = _make_bitmap(len(keys))
        for i, ok in enumerate(exists):
            if ok:
                bitmap.set(i)
        return bitmap

    def _finish_lookup_task(
        self, task_id: L2TaskId, bitmap_size: int, future: Future[Any]
    ) -> None:
        bitmap = _make_bitmap(bitmap_size)
        try:
            bitmap = future.result()
        except Exception as e:
            logger.error("RawBlockL2Adapter lookup task %d failed: %s", task_id, e)
        with self._lock:
            self._lookup_inflight_tasks -= 1
            self._completed_lookup_tasks[task_id] = bitmap
            event_fd = self._lookup_efd
        self._signal_event_fd(event_fd)

    def _run_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[Bitmap, list[ObjectKey]]:
        specs = [encode_object_key(key) for key in keys]
        results = self._core.load_many_into([spec.encoded for spec in specs], objects)
        bitmap = _make_bitmap(len(keys))
        accessed_keys: list[ObjectKey] = []
        for i, ok in enumerate(results):
            if ok:
                bitmap.set(i)
                accessed_keys.append(keys[i])
        return bitmap, accessed_keys

    def _finish_load_task(
        self, task_id: L2TaskId, bitmap_size: int, future: Future[Any]
    ) -> None:
        bitmap = _make_bitmap(bitmap_size)
        accessed_keys: list[ObjectKey] = []
        try:
            bitmap, accessed_keys = future.result()
        except Exception as e:
            logger.error("RawBlockL2Adapter load task %d failed: %s", task_id, e)
        with self._lock:
            self._load_inflight_tasks -= 1
            self._completed_load_tasks[task_id] = bitmap
            event_fd = self._load_efd
        if accessed_keys:
            try:
                self._notify_keys_accessed(accessed_keys)
            except Exception as e:
                logger.warning("RawBlockL2Adapter access notification failed: %s", e)
        self._signal_event_fd(event_fd)

    def _signal_event_fd(self, event_fd: EventNotifier | None) -> None:
        try:
            if event_fd is not None:
                event_fd.notify()
        except OSError:
            logger.debug("event notifier was closed before signaling")

    def _cleanup_after_init_failure(self) -> None:
        for pool_name in ("_store_pool", "_lookup_pool", "_load_pool"):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
                setattr(self, pool_name, None)

        core = getattr(self, "_core", None)
        if core is not None:
            core.close()

        for fd_name in ("_store_efd", "_lookup_efd", "_load_efd"):
            fd = getattr(self, fd_name, None)
            if fd is not None:
                fd.close()
                setattr(self, fd_name, None)

        self._closed = True


register_l2_adapter_type("raw_block", RawBlockL2AdapterConfig)


def _create_raw_block_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    return RawBlockL2Adapter(config, l1_memory_desc)  # type: ignore[arg-type]


register_l2_adapter_factory("raw_block", _create_raw_block_adapter)

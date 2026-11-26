# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from dataclasses import dataclass, replace
from typing import TypedDict
import threading

# First Party
from lmcache.v1.common.attn_window import DEFAULT_ATTN_WINDOW_DESC, AttnWindowDesc
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import (
    DEFAULT_SHARD_REPLICATION_DESC,
    MemoryLayoutDesc,
    ObjectKey,
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.session import SessionManager


class ShmPoolInfo(TypedDict):
    """Shared-memory pool metadata returned during registration."""

    shm_name: str
    pool_size: int


@dataclass
class _LayoutDescEntry:
    """Stored layout descriptor and its active registration count."""

    chunk_size: int
    token_hasher: TokenHasher
    layout_desc: MemoryLayoutDesc
    ref_count: int
    attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC
    """Cross-chunk attention windows of all object groups, in object-group
    order. Defaults to a single full-attention group."""
    group_layout_descs: dict[int, MemoryLayoutDesc] | None = None
    """Per-group layout descriptors, or ``None`` if all share ``layout_desc``."""
    group_unfull_chunk_layout_descs: dict[int, MemoryLayoutDesc] | None = None
    """Per-group layouts of unfull-chunk objects, or ``None`` if they match
    ``group_layout_descs``."""
    shard_replication_desc: ShardReplicationDesc = DEFAULT_SHARD_REPLICATION_DESC
    """How each object group is spread over the kv ranks; the default stores
    every group per rank."""


def check_chunk_hash_key(ctx: "MPCacheServerContext", key: IPCCacheServerKey) -> None:
    """Reject a chunk-hash key the server cannot use.

    KV-event and coordinator reporting bind every stored chunk to its token
    ids (``MP_TOKENS``) and drop store records that have none. A key carrying
    client-computed chunk hashes has no token ids, so its stores would never
    be reported. Its hashes must also be one per complete chunk of
    ``[0, end)`` at the registration's chunk size (see
    :meth:`IPCCacheServerKey.check_chunk_hash_count`); an unregistered key is
    left to the caller.

    Args:
        ctx: The server context.
        key: Key of an incoming lookup, lock release or transfer.

    Raises:
        ValueError: If ``key`` carries ``chunk_hashes`` while token bindings
            have subscribers, or not one per complete chunk of ``[0, end)``.
    """
    if not key.chunk_hashes:
        return
    if ctx.event_bus.has_subscribers(EventType.MP_TOKENS):
        raise ValueError(
            "Keys carrying client-computed chunk hashes have no token ids, "
            "which KV-event and coordinator reporting need; start the server "
            "with --kv-event-log-size 0 (or --disable-observability) and "
            "without --coordinator-event-reporting or --trace-level events, "
            "or turn off client chunk hashes"
        )
    registry = ctx.layout_desc_registry
    if registry.find(key.model_name, key.world_size) is not None:
        key.check_chunk_hash_count(
            registry.get_chunk_size(key.model_name, key.world_size)
        )


def finish_lookup_read(
    ctx: "MPCacheServerContext",
    key: IPCCacheServerKey,
    obj_keys: list[ObjectKey],
    l1_owners: dict[ObjectKey, int] | None = None,
) -> None:
    """Release a lookup's read locks on ``obj_keys``, each object group the
    count its prefetch reserved (:meth:`ShardReplicationDesc.num_kv_readers`).

    Args:
        ctx: The server context owning the registry and storage manager.
        key: The lookup's IPC key, naming the registration.
        obj_keys: The object keys to release.
        l1_owners: The L1 manager that holds each key, when known.
    """
    desc = ctx.layout_desc_registry.find_shard_replication_desc(
        key.model_name, key.world_size
    )
    by_read_locks: dict[int, list[ObjectKey]] = {}
    for obj_key in obj_keys:
        by_read_locks.setdefault(
            desc.num_kv_readers(obj_key.object_group_id), []
        ).append(obj_key)
    for read_locks, group_keys in by_read_locks.items():
        ctx.storage_manager.finish_read_prefetched(
            group_keys, read_locks=read_locks, l1_owners=l1_owners
        )


class LayoutDescRegistry:
    """Registry of layouts and hashing metadata keyed by model and world size.

    Modules write to this thread-safe registry when KV caches are registered.
    Consumers read the layout, chunk size, and token hasher from the same entry
    so object geometry and key derivation cannot diverge. Multiple worker
    instances can share an entry, which remains until the last matching
    registration is unregistered.
    """

    def __init__(self) -> None:
        # Key: (model_name, world_size) -> layout descriptor entry
        self._registry: dict[tuple[str, int], _LayoutDescEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
        chunk_size: int,
        hash_algorithm: str,
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        group_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        group_unfull_chunk_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        shard_replication_desc: ShardReplicationDesc = DEFAULT_SHARD_REPLICATION_DESC,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Re-registering the same pair increments the active registration
        count. The latest descriptor is retained for lookups.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
            chunk_size: Effective chunk size for this registration.
            hash_algorithm: Hash algorithm for this registration.
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order. Defaults to a single full-attention group.
            group_layout_descs: Maps object_group_id to that group's layout
                (each group is a separate keyed allocation); ``None`` derives
                one entry per object group in ``attn_desc``, all sharing
                ``layout_desc``.
            group_unfull_chunk_layout_descs: Maps object_group_id to the layout of
                that group's unfull-chunk objects; ``None`` when they use
                ``group_layout_descs``.
            shard_replication_desc: How each object group is spread over the kv ranks.
                Empty (the default) stores every group per rank; otherwise it
                describes every object group of ``attn_desc``.

        Raises:
            ValueError: If the chunk size or hash algorithm conflicts with an
                existing registration, or ``shard_replication_desc`` describes
                a different number of object groups than ``attn_desc``.
        """
        key = (model_name, world_size)
        attn_desc = replace(attn_desc, world_size=world_size)
        if (
            shard_replication_desc.num_object_groups
            and shard_replication_desc.num_object_groups != attn_desc.num_object_groups
        ):
            raise ValueError(
                "LayoutDescRegistry: shard_replication_desc describes "
                f"{shard_replication_desc.num_object_groups} object groups but "
                f"attn_desc has {attn_desc.num_object_groups}"
            )
        if group_layout_descs is None:
            group_layout_descs = {
                gid: layout_desc for gid in range(attn_desc.num_object_groups)
            }
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                self._registry[key] = _LayoutDescEntry(
                    chunk_size=chunk_size,
                    token_hasher=TokenHasher(
                        chunk_size=chunk_size, hash_algorithm=hash_algorithm
                    ),
                    layout_desc=layout_desc,
                    ref_count=1,
                    attn_desc=attn_desc,
                    group_layout_descs=group_layout_descs,
                    group_unfull_chunk_layout_descs=group_unfull_chunk_layout_descs,
                    shard_replication_desc=shard_replication_desc,
                )
                return

            if entry.chunk_size != chunk_size:
                raise ValueError(
                    f"KV cache registration {key!r} already uses chunk size "
                    f"{entry.chunk_size}; cannot register chunk size {chunk_size}"
                )
            if entry.token_hasher.hash_algorithm_name != hash_algorithm:
                raise ValueError(
                    f"KV cache registration {key!r} already uses hash algorithm "
                    f"{entry.token_hasher.hash_algorithm_name!r}; cannot register "
                    f"hash algorithm {hash_algorithm!r}"
                )

            entry.layout_desc = layout_desc
            entry.attn_desc = attn_desc
            entry.group_layout_descs = group_layout_descs
            entry.group_unfull_chunk_layout_descs = group_unfull_chunk_layout_descs
            entry.shard_replication_desc = shard_replication_desc
            entry.ref_count += 1

    def unregister(self, model_name: str, world_size: int) -> None:
        """Unregister one layout descriptor registration for a pair.

        The descriptor is removed only when the last active registration for
        the pair is unregistered.

        Args:
            model_name: The model name.
            world_size: The world size.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                return

            if entry.ref_count <= 1:
                self._registry.pop(key)
                return

            entry.ref_count -= 1

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by (model_name, world_size).

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.layout_desc

    def get_chunk_size(self, model_name: str, world_size: int) -> int:
        """Return the effective chunk size registered for a model and world size.

        Args:
            model_name: Registered model name.
            world_size: Registered KV world size.

        Returns:
            The effective chunk size.

        Raises:
            ValueError: If the model and world size are not registered.
        """
        with self._lock:
            return self._registered_entry(model_name, world_size).chunk_size

    def get_token_hasher(self, model_name: str, world_size: int) -> TokenHasher:
        """Return the token hasher registered for a model and world size.

        Args:
            model_name: Registered model name.
            world_size: Registered KV world size.

        Returns:
            The registered token hasher.

        Raises:
            ValueError: If the model and world size are not registered.
        """
        with self._lock:
            return self._registered_entry(model_name, world_size).token_hasher

    def find_group_layout_descs(
        self, model_name: str, world_size: int
    ) -> dict[int, MemoryLayoutDesc] | None:
        """Look up per-object-group layout descriptors, or ``None``."""
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.group_layout_descs

    def find_group_shard_geometry(
        self, model_name: str, world_size: int
    ) -> tuple[tuple[int, ...], ShardReplicationDesc] | None:
        """Read the object group IDs and shard replication from one registration.

        Args:
            model_name: Registered model name.
            world_size: Registered KV world size.

        Returns:
            The group IDs and their replication descriptor, or None when the
            registration is absent or has no groups.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None or not entry.group_layout_descs:
                return None
            return tuple(entry.group_layout_descs), entry.shard_replication_desc

    def find_group_unfull_chunk_layout_descs(
        self, model_name: str, world_size: int
    ) -> dict[int, MemoryLayoutDesc] | None:
        """Look up per-object-group layouts of unfull-chunk objects.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The registered unfull-chunk layouts, the per-group layouts when
            none were registered, or ``None`` if the pair is not registered.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.group_unfull_chunk_layout_descs or entry.group_layout_descs

    def find_attn_desc(self, model_name: str, world_size: int) -> AttnWindowDesc:
        """Look up the attention-window descriptor for a pair.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The :class:`AttnWindowDesc` registered for the pair.

        Raises:
            ValueError: If no descriptor is registered for the pair. Callers
                must register the KV cache (which establishes the pair) before
                looking up its windows.
        """
        with self._lock:
            return self._registered_entry(model_name, world_size).attn_desc

    def find_shard_replication_desc(
        self, model_name: str, world_size: int
    ) -> ShardReplicationDesc:
        """Look up how each object group is spread over the kv ranks.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The :class:`ShardReplicationDesc` registered for the pair; the default
            (every group stored per rank) when the registration gave none.

        Raises:
            ValueError: If no descriptor is registered for the pair.
        """
        with self._lock:
            return self._registered_entry(model_name, world_size).shard_replication_desc

    def _registered_entry(self, model_name: str, world_size: int) -> _LayoutDescEntry:
        """The pair's registration; ``ValueError`` if none. Call under
        ``_lock``."""
        entry = self._registry.get((model_name, world_size))
        if entry is None:
            raise ValueError(
                f"No KV cache registered for model {model_name!r} with world "
                f"size {world_size}"
            )
        return entry


class MPCacheServerContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Immutable server minimum chunk size for KV cache operations.
            Clients resolve model-specific effective sizes before registration.
        hash_algorithm: Hash algorithm for token hashing.
        separate_object_groups: Whether to split kernel groups into one object
            group per sliding-window size at KV-cache registration. Default
            True.
        null_block_id: Engine block ID that denotes absent KV data.
        session_ttl_seconds: Idle seconds before a request session is reaped.
        requires_global_chunk_size: Whether registrations must use the server
            chunk size for CacheBlend or Coordinator compatibility.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
        null_block_id: int = 0,
        session_ttl_seconds: float = SessionManager.DEFAULT_SESSION_TTL,
        requires_global_chunk_size: bool = False,
    ) -> None:
        self._chunk_size = chunk_size
        self._null_block_id = null_block_id
        self._hash_algorithm = hash_algorithm
        self._requires_global_chunk_size = requires_global_chunk_size
        self._separate_object_groups = separate_object_groups
        self._full_sw_kv = full_sw_kv

        self.shm_pool_info: ShmPoolInfo = self._compute_shm_pool_info(
            storage_manager_config
        )
        self._storage_manager = StorageManager(storage_manager_config)
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._session_manager = SessionManager(
            self._token_hasher, ttl=session_ttl_seconds
        )
        self._event_bus = get_event_bus()
        self._layout_desc_registry = LayoutDescRegistry()

    def close(self) -> None:
        """
        Tear down sessions and every storage manager, including its GDS context.
        """
        self._session_manager.close()
        self._storage_manager.close()

    @property
    def chunk_size(self) -> int:
        """Return the immutable server minimum chunk size."""
        return self._chunk_size

    @property
    def hash_algorithm_name(self) -> str:
        """Hash algorithm used for token hashing."""
        return self._hash_algorithm

    def validate_chunk_size(self, chunk_size: int) -> None:
        """Validate an effective chunk size against server-wide constraints.

        Args:
            chunk_size: Effective logical tokens per registered chunk.

        Raises:
            ValueError: If the size is below the server chunk size, or
                differs from the server chunk size when one global size is
                required.
        """
        if chunk_size < self._chunk_size:
            raise ValueError(
                f"chunk_size must be at least the server minimum "
                f"{self._chunk_size}, got {chunk_size}"
            )
        if self._requires_global_chunk_size and chunk_size != self._chunk_size:
            raise ValueError(
                "CacheBlend or Coordinator requires one global chunk size: "
                f"the server uses {self._chunk_size}, but the KV cache "
                f"registration uses {chunk_size}. Configure --chunk-size to a "
                "value compatible with every connected model."
            )

    @property
    def null_block_id(self) -> int:
        """Engine block ID that denotes absent KV data."""
        return self._null_block_id

    @property
    def separate_object_groups(self) -> bool:
        """Whether to split kernel groups into per-sliding-window object groups."""
        return self._separate_object_groups

    @property
    def full_sw_kv(self) -> bool:
        """Whether sliding-window groups cache full per-chunk KV (no window cutting)."""
        return self._full_sw_kv

    @property
    def storage_manager(self) -> StorageManager:
        """The storage manager instance."""
        return self._storage_manager

    def get_read_owners(self, request_id: str) -> dict[ObjectKey, int] | None:
        """Get the exact prefetch owners for a request, or None for legacy single L1.

        Args:
            request_id: Request whose lookup acquired the read locks.

        Raises:
            ValueError: A multi-L1 retrieve has no live lookup session.
        """
        if not self._storage_manager.is_multi_l1:
            return None
        session = self._session_manager.get(request_id)
        if session is None:
            raise ValueError("Multi-L1 retrieve requires a live prefetch session")
        return session.get_prefetch_owners()

    @property
    def token_hasher(self) -> TokenHasher:
        """Return the single-size hasher retained for CacheBlend compatibility."""
        return self._token_hasher

    @property
    def session_manager(self) -> SessionManager:
        """The session manager for request lifecycle tracking."""
        return self._session_manager

    @property
    def event_bus(self) -> EventBus:
        """The event bus for observability events."""
        return self._event_bus

    @property
    def layout_desc_registry(self) -> LayoutDescRegistry:
        """Return the registered layouts and their hashing metadata."""
        return self._layout_desc_registry

    def resolve_obj_keys(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> list[list[ObjectKey]]:
        """Resolve per-object-group object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range; a key carrying
        client-computed chunk hashes supplies them directly. A partial chunk
        past the last chunk boundary is not included; see
        :meth:`resolve_unfull_chunk_obj_keys`.

        Args:
            key: IPC cache key describing model/session/token range.
            object_group_ids: Object group ids to produce keys for.

        Returns:
            The i-th element is the list of ObjectKeys for
            ``object_group_ids[i]``.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``, no KV cache is
                registered for the key's ``(model_name, world_size)``, or
                the key carries chunk hashes while cache events need token
                ids or not one per complete chunk of ``[0, end)`` (see
                :func:`check_chunk_hash_key`).
        """
        check_chunk_hash_key(self, key)
        chunk_size = self.layout_desc_registry.get_chunk_size(
            key.model_name, key.world_size
        )
        token_hasher = self.layout_desc_registry.get_token_hasher(
            key.model_name, key.world_size
        )
        session = self.session_manager.get_or_create(key.request_id, token_hasher)
        session.update_from_key(key)
        if session.lookup_ipc_key is None:
            session.lookup_ipc_key = key.no_worker_id_version()
        # A partial chunk past the last chunk boundary is not a full chunk;
        # an unfull chunk's key comes from resolve_unfull_chunk_obj_keys.
        full_chunk_end = key.end - key.end % chunk_size
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h)
            for h in session.get_hashes(key.start, full_chunk_end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        shard_replication_desc = self.layout_desc_registry.find_shard_replication_desc(
            key.model_name, key.world_size
        )
        return ipc_key_to_object_keys(
            key, chunk_hashes, object_group_ids, shard_replication_desc
        )

    def resolve_unfull_chunk_obj_keys(
        self,
        key: IPCCacheServerKey,
        object_group_ids: list[int],
        chunk_size: int,
    ) -> list[ObjectKey]:
        """Resolve the object key of the unfull chunk that ends ``key``'s range.

        Args:
            key: IPC cache key whose ``end`` is not chunk-aligned (see
                :meth:`IPCCacheServerKey.has_unfull_chunk`).
            object_group_ids: Object group ids to produce keys for.
            chunk_size: Effective full chunk size of the registered context.

        Returns:
            One object key per entry of ``object_group_ids``, in order. Each
            key is tagged with the registration's effective full chunk size.

        Raises:
            ValueError: If ``key.worker_id`` is ``None`` or ``key.end`` is
                chunk-aligned.
        """
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        token_hasher = self.layout_desc_registry.get_token_hasher(
            key.model_name, key.world_size
        )
        session = self.session_manager.get_or_create(key.request_id, token_hasher)
        session.set_tokens(key.token_bytes)
        unfull_hash = TokenHasher.hash_to_bytes(session.get_unfull_chunk_hash(key.end))
        return [
            group_keys[0]
            for group_keys in ipc_key_to_object_keys(
                key,
                [unfull_hash],
                object_group_ids,
                self.layout_desc_registry.find_shard_replication_desc(
                    key.model_name, key.world_size
                ),
                chunk_size,
            )
        ]

    @staticmethod
    def _compute_shm_pool_info(
        storage_manager_config: StorageManagerConfig,
    ) -> ShmPoolInfo:
        """Compute normalized SHM pool metadata from storage config.

        Returns an empty pool (disabled SHM transport) when ``shm_name`` is
        empty or lazy memory mode is enabled. Otherwise strips any leading ``/``
        and ensures the name starts with ``lmcache_l1_pool_``.
        """
        if len(storage_manager_config.l1_manager_configs) != 1:
            return {"shm_name": "", "pool_size": 0}
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        shm_name = mem_cfg.shm_name or ""
        if not shm_name or mem_cfg.use_lazy or mem_cfg.devdax_path:
            return {"shm_name": "", "pool_size": 0}
        bare = shm_name.lstrip("/")
        if not bare.startswith("lmcache_l1_pool_"):
            shm_name = f"lmcache_l1_pool_{bare}"
        return {"shm_name": shm_name, "pool_size": mem_cfg.size_in_bytes}

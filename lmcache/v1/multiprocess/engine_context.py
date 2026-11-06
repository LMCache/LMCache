# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypedDict
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    DEFAULT_ATTN_WINDOW_DESC,
    DEFAULT_SHARD_REPLICATION_DESC,
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ShardReplicationDesc,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class ShmPoolInfo(TypedDict):
    """Shared-memory pool metadata returned during registration."""

    shm_name: str
    pool_size: int


@dataclass(frozen=True)
class _Registration:
    """One active KV cache registration of a ``(model_name, world_size)``."""

    worker_id: int | None
    """The registering worker's id, or ``None`` when the registration does
    not carry one."""
    layout_desc: MemoryLayoutDesc
    attn_desc: AttnWindowDesc
    group_layout_descs: dict[int, MemoryLayoutDesc]
    shard_replication_desc: ShardReplicationDesc


@dataclass
class _LayoutDescEntry:
    """A pair's active registrations and the descriptors lookups use."""

    registrations: list[_Registration]
    """Active registrations, oldest first."""
    layout_desc: MemoryLayoutDesc
    attn_desc: AttnWindowDesc
    """Cross-chunk attention windows of all object groups, in object-group
    order."""
    group_layout_descs: dict[int, MemoryLayoutDesc]
    """Per-group layout descriptors."""
    shard_replication_desc: ShardReplicationDesc
    """How each object group is spread over the kv ranks."""


def _merge_registrations(registrations: list[_Registration]) -> _LayoutDescEntry:
    """Combine a pair's active registrations into the descriptors lookups use.

    A pipeline stage registers only the object groups of its own layers, so
    the registration with the most groups (the latest among ties) describes
    them all; a group id means the same KV on every stage that registers it.
    A group that not every registering worker stores is restricted to the
    workers that do (:meth:`ShardReplicationDesc.storing_workers`).

    Args:
        registrations: The pair's active registrations, oldest first; not
            empty.

    Returns:
        The pair's entry.
    """
    base = max(reversed(registrations), key=lambda r: r.attn_desc.num_object_groups)
    num_groups = base.attn_desc.num_object_groups
    registered = {r.worker_id for r in registrations}
    storing_workers: list[frozenset[int] | None] = []
    for group in range(num_groups):
        owners = {
            r.worker_id for r in registrations if group < r.attn_desc.num_object_groups
        }
        # A registration without a worker id may be any worker.
        if owners == registered or None in owners:
            storing_workers.append(None)
        else:
            storing_workers.append(frozenset(w for w in owners if w is not None))

    shard_replication_desc = base.shard_replication_desc
    if any(workers is not None for workers in storing_workers):
        shard_replication_desc = ShardReplicationDesc(
            group_tp_replicas=tuple(
                shard_replication_desc.tp_replicas(g) for g in range(num_groups)
            ),
            group_dcp_shards=tuple(
                shard_replication_desc.dcp_shards(g) for g in range(num_groups)
            ),
            group_storing_workers=tuple(storing_workers),
        )
    return _LayoutDescEntry(
        registrations=registrations,
        layout_desc=base.layout_desc,
        attn_desc=base.attn_desc,
        group_layout_descs=base.group_layout_descs,
        shard_replication_desc=shard_replication_desc,
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
    """Thread-safe registry mapping (model_name, world_size) to MemoryLayoutDesc.

    Modules write to this registry when KV caches are registered.
    Consumers (e.g. LookupModule) read from it to find layout descriptors
    for prefetch tasks. Multiple worker instances can share the same
    ``(model_name, world_size)`` entry, so the registry keeps the descriptor
    until the last matching registration is unregistered.
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
        attn_desc: AttnWindowDesc = DEFAULT_ATTN_WINDOW_DESC,
        group_layout_descs: dict[int, MemoryLayoutDesc] | None = None,
        shard_replication_desc: ShardReplicationDesc = DEFAULT_SHARD_REPLICATION_DESC,
        worker_id: int | None = None,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Each call adds one active registration of the pair. Lookups use the
        descriptors of the registration with the most object groups (the
        latest among ties). When registering workers store different groups,
        as pipeline stages do, a group is looked up only on the workers that
        registered it (:meth:`ShardReplicationDesc.storing_workers`).

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
            attn_desc: Cross-chunk attention windows of all object groups, in
                object-group order. Defaults to a single full-attention group.
            group_layout_descs: Maps object_group_id to that group's layout
                (each group is a separate keyed allocation); ``None`` derives
                one entry per object group in ``attn_desc``, all sharing
                ``layout_desc``.
            shard_replication_desc: How each object group is spread over the kv ranks.
                Empty (the default) stores every group per rank; otherwise it
                describes every object group of ``attn_desc``.
            worker_id: The registering worker's id within ``world_size`` (the
                ``worker_id`` its store keys carry), or ``None`` when the
                registration does not carry one; its groups then count as
                stored by every worker.

        Raises:
            ValueError: If ``shard_replication_desc`` describes a different number of
                object groups than ``attn_desc``.
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
        registration = _Registration(
            worker_id=worker_id,
            layout_desc=layout_desc,
            attn_desc=attn_desc,
            group_layout_descs=group_layout_descs,
            shard_replication_desc=shard_replication_desc,
        )
        with self._lock:
            entry = self._registry.get(key)
            registrations = [] if entry is None else entry.registrations
            self._registry[key] = _merge_registrations([*registrations, registration])

    def unregister(
        self, model_name: str, world_size: int, worker_id: int | None = None
    ) -> None:
        """Unregister one layout descriptor registration for a pair.

        Removes the most recent registration made with ``worker_id``, if any.
        The descriptor is removed only when the last active registration for
        the pair is unregistered.

        Args:
            model_name: The model name.
            world_size: The world size.
            worker_id: The worker id the registration was made with.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                return
            registrations = entry.registrations
            matches = [
                i for i, r in enumerate(registrations) if r.worker_id == worker_id
            ]
            if not matches:
                return
            remaining = registrations[: matches[-1]] + registrations[matches[-1] + 1 :]
            if not remaining:
                self._registry.pop(key)
                return
            self._registry[key] = _merge_registrations(remaining)

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

    def find_group_layout_descs(
        self, model_name: str, world_size: int
    ) -> dict[int, MemoryLayoutDesc] | None:
        """Look up per-object-group layout descriptors, or ``None``."""
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.group_layout_descs

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
        chunk_size: Minimum chunk size for KV cache operations. vLLM connector
            startup may negotiate this upward to satisfy model-specific KV
            geometry.
        hash_algorithm: Hash algorithm for token hashing.
        separate_object_groups: Whether to split kernel groups into one object
            group per sliding-window size at KV-cache registration. Default
            False.
        null_block_id: Engine block ID that denotes absent KV data.
        session_ttl_seconds: Idle seconds before a request session is reaped.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
        null_block_id: int = 0,
        session_ttl_seconds: float = SessionManager.DEFAULT_SESSION_TTL,
    ) -> None:
        self._configured_chunk_size = chunk_size
        self._chunk_size = chunk_size
        self._null_block_id = null_block_id
        self._hash_algorithm = hash_algorithm
        self._session_ttl_seconds = session_ttl_seconds
        self._chunk_size_lock = threading.Lock()
        self._chunk_size_selected = False
        self._chunk_size_finalized = False
        self._chunk_size_bind_listeners: list[Callable[[int], None]] = []
        self._next_chunk_size_bind_listener = 0
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
        """Chunk size for KV cache operations."""
        return self._chunk_size

    @property
    def finalized_chunk_size(self) -> int | None:
        """Return the immutable chunk size, or ``None`` before negotiation."""
        with self._chunk_size_lock:
            return self._chunk_size if self._chunk_size_finalized else None

    def finalize_chunk_size(self) -> int:
        """Return the current chunk size and prevent future renegotiation."""
        with self._chunk_size_lock:
            self._chunk_size_selected = True
            self._finalize_chunk_size_locked()
            return self._chunk_size

    @property
    def hash_algorithm_name(self) -> str:
        """Hash algorithm used for token hashing."""
        return self._hash_algorithm

    def negotiate_chunk_size(self, required_alignment: int) -> int:
        """Negotiate the server chunk size against model geometry.

        Args:
            required_alignment: The least common multiple of all positive
                cacheable ``tokens_per_block`` values for the registering
                model.

        Returns:
            The concrete server chunk size.

        Raises:
            ValueError: If ``required_alignment`` is not positive, if an
                already-finalized chunk size is incompatible with
                ``required_alignment``, or if a resize is requested after
                sessions already exist.
        """
        if required_alignment < 1:
            raise ValueError(
                f"required_chunk_alignment must be positive, got {required_alignment}"
            )
        with self._chunk_size_lock:
            if self._chunk_size_selected:
                if self._chunk_size % required_alignment == 0:
                    self._finalize_chunk_size_locked()
                    return self._chunk_size
                raise ValueError(
                    f"LMCache chunk size {self._chunk_size} must be a multiple "
                    f"of required chunk alignment {required_alignment}. This MP "
                    f"server has already selected its chunk size; models that "
                    f"require incompatible alignments must use separate MP servers."
                )

            resolved = _round_up_to_multiple(
                self._configured_chunk_size, required_alignment
            )
            if resolved != self._chunk_size:
                if self._session_manager.active_count():
                    raise ValueError(
                        "Cannot negotiate LMCache chunk size after sessions "
                        "have started"
                    )
                self._bind_chunk_size_locked(resolved)
            self._chunk_size_selected = True
            self._finalize_chunk_size_locked()
            if resolved != self._configured_chunk_size:
                logger.info(
                    "Negotiated LMCache MP chunk size to %d "
                    "(configured_min=%d, required_alignment=%d)",
                    resolved,
                    self._configured_chunk_size,
                    required_alignment,
                )
            return resolved

    def _finalize_chunk_size_locked(self) -> None:
        """Notify dependents, then commit the current size as immutable.

        A listener may perform a remote update and fail transiently. Keep the
        size unfinalized until every listener succeeds so the next negotiation
        retries the notification instead of returning a false success.
        """
        if self._chunk_size_finalized:
            return
        listeners = list(self._chunk_size_bind_listeners)
        while self._next_chunk_size_bind_listener < len(listeners):
            listener = listeners[self._next_chunk_size_bind_listener]
            listener(self._chunk_size)
            self._next_chunk_size_bind_listener += 1
        self._chunk_size_finalized = True

    def add_chunk_size_bind_listener(self, listener: Callable[[int], None]) -> None:
        """Register a callback for chunk-size changes during negotiation."""
        self._chunk_size_bind_listeners.append(listener)

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
        """The token hasher for computing chunk hashes."""
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
        """Registry mapping (model_name, world_size) to MemoryLayoutDesc."""
        return self._layout_desc_registry

    def resolve_obj_keys(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> list[list[ObjectKey]]:
        """Resolve per-object-group object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.
            object_group_ids: Object group ids to produce keys for.

        Returns:
            The i-th element is the list of ObjectKeys for
            ``object_group_ids[i]``.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``, or no KV cache is
                registered for the key's ``(model_name, world_size)``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(key.token_bytes)
        if session.lookup_ipc_key is None:
            session.lookup_ipc_key = key.no_worker_id_version()
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        shard_replication_desc = self.layout_desc_registry.find_shard_replication_desc(
            key.model_name, key.world_size
        )
        return ipc_key_to_object_keys(
            key, chunk_hashes, object_group_ids, shard_replication_desc
        )

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

    def _bind_chunk_size_locked(self, chunk_size: int) -> None:
        """Initialize runtime state that depends on a concrete chunk size."""
        previous_session_manager = getattr(self, "_session_manager", None)
        destroy_listeners = (
            list(previous_session_manager.destroy_listeners)
            if previous_session_manager is not None
            else []
        )
        if previous_session_manager is not None:
            previous_session_manager.close()
        self._chunk_size = chunk_size
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=self._hash_algorithm
        )
        self._session_manager = SessionManager(
            self._token_hasher, ttl=self._session_ttl_seconds
        )
        self._session_manager.destroy_listeners.extend(destroy_listeners)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to a positive multiple of *multiple*."""
    return ((value + multiple - 1) // multiple) * multiple

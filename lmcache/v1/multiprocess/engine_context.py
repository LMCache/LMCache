# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


@dataclass
class _LayoutDescEntry:
    """Stored layout descriptor and its active registration count."""

    layout_desc: MemoryLayoutDesc
    ref_count: int


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
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Re-registering the same pair increments the active registration
        count. The latest descriptor is retained for lookups.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                self._registry[key] = _LayoutDescEntry(
                    layout_desc=layout_desc,
                    ref_count=1,
                )
                return

            entry.layout_desc = layout_desc
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


class MPCacheEngineContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Chunk size for KV cache operations.
        hash_algorithm: Hash algorithm for token hashing.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ) -> None:
        self._chunk_size = chunk_size
        self._storage_manager = StorageManager(storage_manager_config)
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._session_manager = SessionManager(self._token_hasher)
        self._event_bus = get_event_bus()
        self._layout_desc_registry = LayoutDescRegistry()

    @property
    def chunk_size(self) -> int:
        """Chunk size for KV cache operations."""
        return self._chunk_size

    @property
    def storage_manager(self) -> StorageManager:
        """The storage manager instance."""
        return self._storage_manager

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

    def resolve_obj_keys(self, key: IPCCacheEngineKey) -> list[ObjectKey]:
        """Resolve object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.

        Returns:
            Resolved object keys for the requested token range.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes)

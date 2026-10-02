# SPDX-License-Identifier: Apache-2.0
"""Hidden-state cache operations for the MPCacheServer.

Hidden states are per-token activations a pipelined engine wants back on a
prefix hit, so it can skip re-running the stage that produced them. They ride
the same chunk hashes as KV but live under their own model namespace, and the
server moves them as pickled CPU tensors rather than through a GPU transfer
context -- there is no paged device-side buffer to point at.
"""

# Standard
from dataclasses import replace
import pickle
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler

logger = init_logger(__name__)

HIDDEN_NAMESPACE_SUFFIX = "##hidden"
"""Appended to the model name so hidden-state objects never collide with the
KV objects of the same model. Mirrors the ``##query`` namespace of qstore."""

_OBJECT_GROUP_ID = 0
"""Hidden states are a single object group: one tensor per chunk, no kernel
groups and no sliding windows."""

_PREFETCH_TIMEOUT_S = 5.0


def hidden_model_name(model_name: str) -> str:
    """Return the hidden-state namespace for ``model_name``."""
    return f"{model_name}{HIDDEN_NAMESPACE_SUFFIX}"


class HiddenStateModule(InstanceLivenessTarget):
    """Stores and serves per-chunk hidden states keyed by the KV chunk hashes.

    A worker declares its layer set and hidden size once, at registration; every
    object for that model then has the same shape and the server never needs the
    layer set on the store or retrieve path.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        # instance_id -> (hidden model name, world size), so unregister can
        # drop the right layout-registry entry without the worker resending it.
        self._registrations: dict[int, tuple[str, int]] = {}
        self._lock = threading.Lock()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    @request_handler(HandlerType.SYNC)
    def register_hidden_state(
        self,
        instance_id: int,
        model_name: str,
        world_size: int,
        num_layers: int,
        hidden_size: int,
        dtype_name: str,
    ) -> bool:
        """Declare the hidden-state layout for a worker.

        Args:
            instance_id: Worker instance identifier.
            model_name: The KV model name; the hidden namespace is derived.
            world_size: World size of the engine.
            num_layers: Number of layers the worker will send per token.
            hidden_size: Per-layer hidden dimension.
            dtype_name: Torch dtype name without the ``torch.`` prefix.

        Returns:
            ``True`` when the layout was registered.
        """
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            logger.error("Unknown hidden-state dtype %r", dtype_name)
            return False
        if num_layers <= 0 or hidden_size <= 0:
            logger.error(
                "Invalid hidden-state layout (num_layers=%d, hidden_size=%d)",
                num_layers,
                hidden_size,
            )
            return False

        name = hidden_model_name(model_name)
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([self._ctx.chunk_size, num_layers, hidden_size])],
            dtypes=[dtype],
        )
        self._ctx.layout_desc_registry.register(name, world_size, layout_desc)
        with self._lock:
            self._registrations[instance_id] = (name, world_size)
        logger.info(
            "Registered hidden-state layout for %s (world_size=%d, layers=%d, "
            "hidden_size=%d, dtype=%s)",
            name,
            world_size,
            num_layers,
            hidden_size,
            dtype_name,
        )
        return True

    @request_handler(HandlerType.SYNC)
    def unregister_hidden_state(self, instance_id: int) -> None:
        """Drop a worker's hidden-state registration.

        Args:
            instance_id: Worker instance identifier.
        """
        with self._lock:
            entry = self._registrations.pop(instance_id, None)
        if entry is None:
            return
        self._ctx.layout_desc_registry.unregister(*entry)

    @request_handler(HandlerType.BLOCKING)
    def lookup_hidden_state(self, key: IPCCacheServerKey, instance_id: int) -> int:
        """Return how many leading tokens of ``key`` have hidden states cached.

        The engine calls this before committing to a KV hit: a KV prefix that
        reaches further than the hidden-state prefix cannot be used in full,
        because the stage downstream would have nothing to consume.

        Args:
            key: Cache key describing the token range to probe.
            instance_id: Worker instance identifier.

        Returns:
            Covered token count, always a multiple of the chunk size.
        """
        row = self._prefetch_row(key, instance_id)
        if row is None:
            return 0
        handle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchTaskSpec(key_groups=[row], lock_mode=PrefetchLockMode.NO_LOCK),
            external_request_id=key.request_id,
        )
        if not self._ctx.storage_manager.wait_prefetch_status(
            handle, _PREFETCH_TIMEOUT_S
        ):
            return 0
        found = self._ctx.storage_manager.query_prefetch_status(handle)
        if not found:
            return 0
        return found[0].count_leading_ones() * self._ctx.chunk_size

    @request_handler(HandlerType.BLOCKING)
    def store_hidden_state(
        self, key: IPCCacheServerKey, instance_id: int, cpu_data: bytes
    ) -> bool:
        """Write one pickled CPU tensor per chunk of ``key``.

        Args:
            key: Cache key describing the token range being stored.
            instance_id: Worker instance identifier.
            cpu_data: Pickled ``list[torch.Tensor]``, one entry per chunk, in
                token order, each shaped like the registered layout.

        Returns:
            ``True`` when every reserved object was written.
        """
        layout_desc = self._layout_desc(instance_id)
        if layout_desc is None:
            return False
        obj_keys = self._resolve_obj_keys(key, instance_id)
        if not obj_keys:
            return False

        chunks: list[torch.Tensor] = pickle.loads(cpu_data)
        reserved = self._ctx.storage_manager.reserve_write(obj_keys, layout_desc)
        written: list[ObjectKey] = []
        try:
            for idx, obj_key in enumerate(obj_keys):
                memory_obj = reserved.get(obj_key)
                if memory_obj is None or memory_obj.tensor is None:
                    continue
                if idx >= len(chunks):
                    logger.error(
                        "Hidden-state store is missing chunk %d of %d "
                        "(instance_id=%d)",
                        idx,
                        len(obj_keys),
                        instance_id,
                    )
                    continue
                if chunks[idx].shape != memory_obj.tensor.shape:
                    logger.error(
                        "Hidden-state store chunk shape mismatch (instance_id=%d, "
                        "chunk_index=%d, chunk_shape=%s, object_shape=%s)",
                        instance_id,
                        idx,
                        tuple(chunks[idx].shape),
                        tuple(memory_obj.tensor.shape),
                    )
                    continue
                memory_obj.tensor.copy_(chunks[idx])
                written.append(obj_key)
        finally:
            if written:
                self._ctx.storage_manager.finish_write(written)

        return len(written) == len(reserved)

    @request_handler(HandlerType.BLOCKING)
    def retrieve_hidden_state(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[bytes, int]:
        """Return the cached hidden states for the leading prefix of ``key``.

        A short read is not an error: the caller gets the prefix that was
        actually cached and recomputes the rest.

        Args:
            key: Cache key describing the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            ``(pickled list[torch.Tensor], covered token count)``.
        """
        row = self._prefetch_row(key, instance_id)
        if row is None:
            return b"", 0
        # One lock, not key.num_kv_readers: this call takes the lock and
        # releases it before returning, so it is its own only reader. Each of
        # the other ranks retrieving the same object does the same.
        handle = self._ctx.storage_manager.submit_prefetch_task(
            PrefetchTaskSpec(key_groups=[row], num_kv_readers=1),
            external_request_id=key.request_id,
        )
        if not self._ctx.storage_manager.wait_prefetch_status(
            handle, _PREFETCH_TIMEOUT_S
        ):
            return b"", 0
        found = self._ctx.storage_manager.query_prefetch_status(handle)
        if not found:
            return b"", 0
        hit_chunks = found[0].count_leading_ones()
        if hit_chunks == 0:
            return b"", 0

        prefix = row.keys[:hit_chunks]
        try:
            with self._ctx.storage_manager.read_prefetched_results(
                prefix
            ) as memory_objs:
                if not memory_objs or len(memory_objs) != hit_chunks:
                    return b"", 0
                chunks = []
                for memory_obj in memory_objs:
                    if memory_obj.tensor is None:
                        return b"", 0
                    chunks.append(memory_obj.tensor.cpu().clone())
                return pickle.dumps(chunks), hit_chunks * self._ctx.chunk_size
        finally:
            self._ctx.storage_manager.finish_read_prefetched(prefix)

    def drop_instance_state(self, instance_id: int) -> None:
        """Release the registration of a reaped worker."""
        self.unregister_hidden_state(instance_id)

    def report_status(self) -> dict:
        """Return the number of registered hidden-state instances."""
        with self._lock:
            return {"hidden_state_registrations": len(self._registrations)}

    def close(self) -> None:
        """Release every outstanding layout registration."""
        with self._lock:
            entries = list(self._registrations.values())
            self._registrations.clear()
        for entry in entries:
            self._ctx.layout_desc_registry.unregister(*entry)

    def _layout_desc(self, instance_id: int) -> MemoryLayoutDesc | None:
        """Return the registered layout for ``instance_id``, or ``None``."""
        with self._lock:
            entry = self._registrations.get(instance_id)
        if entry is None:
            logger.error(
                "Hidden-state request from unregistered instance %d", instance_id
            )
            return None
        return self._ctx.layout_desc_registry.find(*entry)

    def _resolve_obj_keys(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[ObjectKey]:
        """Map an IPC key onto hidden-state object keys.

        The chunk hashes come from the same session the KV path uses, so a
        hidden-state object is addressed by exactly the prefix that produced
        it. Only the model name and the rank differ: hidden states are
        replicated across tensor-parallel ranks, so they are stored once under
        rank 0 rather than once per rank.

        Unlike ``MPCacheServerContext.resolve_obj_keys`` this never touches
        ``session.lookup_ipc_key`` -- that field belongs to the KV lookup, and
        a hidden-state request may well arrive first.
        """
        with self._lock:
            entry = self._registrations.get(instance_id)
        if entry is None:
            return []
        name, _ = entry
        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            self._ctx.token_hasher.hash_to_bytes(h)
            for h in session.get_hashes(key.start, key.end)
        ]
        hidden_key = replace(key, model_name=name, worker_id=0, world_size=1)
        return ipc_key_to_object_keys(hidden_key, chunk_hashes, [_OBJECT_GROUP_ID])[0]

    def _prefetch_row(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> GroupedObjectKeys | None:
        """Build the single prefetch row for ``key``, or ``None`` if unusable."""
        layout_desc = self._layout_desc(instance_id)
        if layout_desc is None:
            return None
        obj_keys = self._resolve_obj_keys(key, instance_id)
        if not obj_keys:
            return None
        return GroupedObjectKeys(
            keys=obj_keys,
            object_group_id=_OBJECT_GROUP_ID,
            layout_desc=layout_desc,
            sliding_window_size=FULL_ATTENTION_WINDOW_CHUNKS,
        )

# SPDX-License-Identifier: Apache-2.0
"""Bytes-level KV cache storage helpers for the multiprocess server."""

# Standard
from collections.abc import AsyncIterable, Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Protocol, cast
import asyncio
import math
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle, StorageManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher

ModelResolver = Callable[[str], tuple[MemoryLayoutDesc, int]]


class _ClosableIterator(Protocol):
    """Iterator protocol extension for generators that can be closed."""

    def close(self) -> None:
        """Close the iterator and run its cleanup handlers."""


@dataclass(frozen=True)
class StoreBytesResult:
    """Outcome of a bytes-level store request submitted via the HTTP API.

    Attributes:
        total_tokens: Whole-chunk token count represented by the request.
        total_chunks: Whole-chunk count represented by the request.
        stored_tokens: Leading whole-chunk token count that was persisted.
        stored_chunks: Leading whole-chunk count that was persisted. This may
            be less than ``total_chunks`` if some keys could not be reserved.
    """

    total_tokens: int
    total_chunks: int
    stored_tokens: int
    stored_chunks: int


@dataclass(frozen=True)
class KVBytesShard:
    """One retrieved KV cache worker shard.

    Attributes:
        chunk_index: Zero-based token chunk index in the retrieved prefix.
        worker_id: Tensor-parallel worker shard index within the chunk.
        data: Raw bytes for the shard's ``MemoryObj``.
    """

    chunk_index: int
    worker_id: int
    data: bytes


class RetrieveBytesResult:
    """Lazy result of a bytes-level retrieve request.

    The result describes the longest cached prefix and exposes shard bytes
    through :meth:`iter_shards`. Callers must either consume
    :meth:`iter_shards` to completion or call :meth:`close` to release read
    locks held by the storage manager.

    Args:
        total_tokens: Whole-chunk token count represented by the request.
        total_chunks: Whole-chunk count represented by the request.
        hit_tokens: Whole-chunk token count available in cache.
        hit_chunks: Whole-chunk count available in cache.
        world_size: Tensor-parallel world size used by the stored shards.
        per_shard_shape: Shape of each retrieved worker shard.
        dtype: Dtype of each retrieved shard.
        shard_iter_factory: Factory that yields retrieved shard bytes.
        close_callback: Idempotent callback that releases unread locks.
    """

    def __init__(
        self,
        total_tokens: int,
        total_chunks: int,
        hit_tokens: int,
        hit_chunks: int,
        world_size: int,
        per_shard_shape: tuple[int, int, int, int],
        dtype: torch.dtype,
        shard_iter_factory: Callable[[], Iterator[KVBytesShard]],
        close_callback: Callable[[], None],
    ) -> None:
        self.total_tokens = total_tokens
        self.total_chunks = total_chunks
        self.hit_tokens = hit_tokens
        self.hit_chunks = hit_chunks
        self.world_size = world_size
        self.per_shard_shape = per_shard_shape
        self.dtype = dtype
        self._shard_iter_factory = shard_iter_factory
        self._close_callback = close_callback
        self._active_iterator: Iterator[KVBytesShard] | None = None

    def iter_shards(self) -> Iterator[KVBytesShard]:
        """Yield retrieved worker shards and release locks when exhausted."""
        iterator = self._shard_iter_factory()
        self._active_iterator = iterator
        try:
            yield from iterator
        finally:
            if self._active_iterator is iterator:
                self._active_iterator = None

    def close(self) -> None:
        """Release read locks if the shard iterator was not fully consumed."""
        active_iterator = self._active_iterator
        if active_iterator is not None and hasattr(active_iterator, "close"):
            cast(_ClosableIterator, active_iterator).close()
        self._close_callback()


async def store_kv_bytes_by_tokens(
    *,
    model_name: str,
    tokens: list[int],
    chunks: AsyncIterable[bytes],
    full_shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    cache_salt: str,
    chunk_size: int,
    token_hasher: TokenHasher,
    storage_manager: StorageManager,
    resolve_model: ModelResolver,
) -> StoreBytesResult:
    """Store KV cache bytes from an async stream of full-token chunks.

    Args:
        model_name: Registered model name.
        tokens: Token sequence the bytes are keyed by.
        chunks: Async byte-stream yielding full KV chunks in order.
        full_shape: Shape of the complete client tensor represented by
            ``chunks``. Must equal ``[2, L, total_tokens, D]`` for the
            server's registered model layout.
        dtype: Dtype of the complete client tensor represented by ``chunks``.
        cache_salt: Per-namespace isolation salt.
        chunk_size: Server token chunk size.
        token_hasher: Token hasher used by the owning ``MPCacheEngine``.
        storage_manager: Storage manager used by the owning ``MPCacheEngine``.
        resolve_model: Callback that maps ``model_name`` to
            ``(layout_desc, world_size)``.

    Returns:
        How many tokens and chunks were actually persisted.

    Raises:
        KeyError: If ``model_name`` is not registered.
        ValueError: If the chunk stream, shape, dtype, or registered model
            layout violates the v1 bytes-level contract.
    """
    layout_desc, world_size = resolve_model(model_name)

    chunk_hashes = token_hasher.compute_chunk_hashes(tokens)
    total_chunks = len(chunk_hashes)
    total_tokens = total_chunks * chunk_size
    if total_chunks == 0:
        return StoreBytesResult(0, 0, 0, 0)

    per_shard_shape, per_shard_dtype = _get_single_group_layout(layout_desc)
    per_shard_bytes = math.prod(per_shard_shape) * per_shard_dtype.itemsize
    expected_chunk_bytes = world_size * per_shard_bytes
    if dtype != per_shard_dtype:
        raise ValueError(
            f"payload dtype {dtype} does not match registered KV dtype "
            f"{per_shard_dtype}"
        )

    ipc_key = IPCCacheEngineKey(
        model_name=model_name,
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(tokens[:total_tokens]),
        start=0,
        end=total_tokens,
        request_id="http-store",
        cache_salt=cache_salt,
    )
    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)

    d_per_worker = per_shard_shape[3]
    expected_full_shape = (
        per_shard_shape[0],
        per_shard_shape[1],
        total_tokens,
        d_per_worker * world_size,
    )
    if full_shape != expected_full_shape:
        raise ValueError(
            f"payload shape {full_shape} does not match expected {expected_full_shape}"
        )

    written_keys: list[ObjectKey] = []
    seen_chunks = 0
    async for chunk_payload in chunks:
        if seen_chunks >= total_chunks:
            raise ValueError(f"received more chunks than expected ({total_chunks})")
        if len(chunk_payload) != expected_chunk_bytes:
            raise ValueError(
                f"chunk {seen_chunks} byte length {len(chunk_payload)} "
                f"does not match expected {expected_chunk_bytes}"
            )

        chunk_keys = obj_keys[seen_chunks * world_size : (seen_chunks + 1) * world_size]
        written_keys.extend(
            await asyncio.to_thread(
                _store_chunk_payload,
                chunk_payload,
                chunk_keys,
                per_shard_shape,
                per_shard_dtype,
                d_per_worker,
                world_size,
                chunk_size,
                storage_manager,
                layout_desc,
            )
        )
        seen_chunks += 1

    if seen_chunks != total_chunks:
        raise ValueError(f"expected {total_chunks} chunks, got {seen_chunks}")

    stored_chunks = _count_leading_complete_chunks(
        obj_keys,
        set(written_keys),
        world_size,
    )
    return StoreBytesResult(
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        stored_tokens=stored_chunks * chunk_size,
        stored_chunks=stored_chunks,
    )


def retrieve_kv_bytes_by_tokens(
    *,
    model_name: str,
    tokens: list[int],
    cache_salt: str,
    chunk_size: int,
    token_hasher: TokenHasher,
    storage_manager: StorageManager,
    resolve_model: ModelResolver,
) -> RetrieveBytesResult:
    """Retrieve KV cache bytes for the longest cached prefix of ``tokens``.

    Args:
        model_name: Registered model name.
        tokens: Token sequence to retrieve.
        cache_salt: Per-namespace isolation salt.
        chunk_size: Server token chunk size.
        token_hasher: Token hasher used by the owning ``MPCacheEngine``.
        storage_manager: Storage manager used by the owning ``MPCacheEngine``.
        resolve_model: Callback that maps ``model_name`` to
            ``(layout_desc, world_size)``.

    Returns:
        Prefix metadata plus a lazy shard iterator. The iterator is empty on a
        miss. Callers must consume the iterator or call ``close``.

    Raises:
        KeyError: If ``model_name`` is not registered.
        ValueError: If the registered model violates the v1 bytes-level
            limitations.
    """
    layout_desc, world_size = resolve_model(model_name)

    chunk_hashes = token_hasher.compute_chunk_hashes(tokens)
    total_chunks = len(chunk_hashes)
    total_tokens = total_chunks * chunk_size
    if total_chunks == 0:
        return RetrieveBytesResult(
            total_tokens=0,
            total_chunks=0,
            hit_tokens=0,
            hit_chunks=0,
            world_size=world_size,
            per_shard_shape=(0, 0, 0, 0),
            dtype=torch.uint8,
            shard_iter_factory=lambda: iter(()),
            close_callback=lambda: None,
        )

    per_shard_shape, per_shard_dtype = _get_single_group_layout(layout_desc)
    ipc_key = IPCCacheEngineKey(
        model_name=model_name,
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(tokens[:total_tokens]),
        start=0,
        end=total_tokens,
        request_id="http-retrieve",
        cache_salt=cache_salt,
    )
    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)

    handle = storage_manager.submit_prefetch_task(
        obj_keys,
        layout_desc,
        extra_count=0,
        external_request_id=ipc_key.request_id,
    )
    total_hit_keys = _wait_prefetch(storage_manager, handle)
    hit_chunks = total_hit_keys // world_size
    hit_tokens = hit_chunks * chunk_size
    locked_obj_keys = obj_keys[:total_hit_keys]
    if hit_chunks == 0:
        if locked_obj_keys:
            storage_manager.finish_read_prefetched(locked_obj_keys)
        return RetrieveBytesResult(
            total_tokens=total_tokens,
            total_chunks=total_chunks,
            hit_tokens=0,
            hit_chunks=0,
            world_size=world_size,
            per_shard_shape=per_shard_shape,
            dtype=per_shard_dtype,
            shard_iter_factory=lambda: iter(()),
            close_callback=lambda: None,
        )

    hit_obj_keys = locked_obj_keys[: hit_chunks * world_size]
    remainder_obj_keys = locked_obj_keys[len(hit_obj_keys) :]

    started = False
    closed = False

    def close_unstarted() -> None:
        nonlocal closed
        if closed:
            return
        if not started and locked_obj_keys:
            storage_manager.finish_read_prefetched(locked_obj_keys)
        closed = True

    def iter_shards() -> Generator[KVBytesShard, None, None]:
        nonlocal closed, started
        if closed:
            return
        started = True
        read_succeeded = False
        try:
            with storage_manager.read_prefetched_results(hit_obj_keys) as memory_objs:
                if memory_objs is None:
                    return
                for idx, memory_obj in enumerate(memory_objs):
                    chunk_idx = idx // world_size
                    worker_id = idx % world_size
                    shard_tensor = _memory_obj_tensor(
                        memory_obj,
                        per_shard_shape,
                        per_shard_dtype,
                    )
                    yield KVBytesShard(
                        chunk_index=chunk_idx,
                        worker_id=worker_id,
                        data=_tensor_to_bytes(shard_tensor),
                    )
                read_succeeded = True
        finally:
            if read_succeeded:
                storage_manager.finish_read_prefetched(locked_obj_keys)
            elif remainder_obj_keys:
                storage_manager.finish_read_prefetched(remainder_obj_keys)
            closed = True

    return RetrieveBytesResult(
        total_tokens=total_tokens,
        total_chunks=total_chunks,
        hit_tokens=hit_tokens,
        hit_chunks=hit_chunks,
        world_size=world_size,
        per_shard_shape=per_shard_shape,
        dtype=per_shard_dtype,
        shard_iter_factory=iter_shards,
        close_callback=close_unstarted,
    )


def _wait_prefetch(storage_manager: StorageManager, handle: PrefetchHandle) -> int:
    """Wait until ``handle`` finishes and return the total hit count."""
    while True:
        status = storage_manager.query_prefetch_status(handle)
        if status is not None:
            return status
        time.sleep(0.01)


def _store_chunk_payload(
    chunk_payload: bytes,
    chunk_keys: list[ObjectKey],
    per_shard_shape: tuple[int, int, int, int],
    per_shard_dtype: torch.dtype,
    d_per_worker: int,
    world_size: int,
    chunk_size: int,
    storage_manager: StorageManager,
    layout_desc: MemoryLayoutDesc,
) -> list[ObjectKey]:
    """Write one full chunk into reserved worker-shard memory objects."""
    chunk_tensor = torch.frombuffer(
        bytearray(chunk_payload),
        dtype=per_shard_dtype,
    ).reshape(
        (
            per_shard_shape[0],
            per_shard_shape[1],
            chunk_size,
            d_per_worker * world_size,
        )
    )
    reserved = storage_manager.reserve_write(
        chunk_keys,
        layout_desc,
        "all",
    )
    written_keys: list[ObjectKey] = []
    try:
        for worker_id in range(world_size):
            obj_key = chunk_keys[worker_id]
            memory_obj = reserved.get(obj_key)
            if memory_obj is None:
                continue
            d_start = worker_id * d_per_worker
            d_end = d_start + d_per_worker
            shard = chunk_tensor[:, :, :, d_start:d_end]
            _memory_obj_tensor(
                memory_obj,
                per_shard_shape,
                per_shard_dtype,
            ).copy_(shard)
            written_keys.append(obj_key)
    finally:
        if reserved:
            storage_manager.finish_write(list(reserved.keys()))
    return written_keys


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Materialize a tensor as dtype-preserving raw bytes."""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _memory_obj_tensor(
    memory_obj: MemoryObj,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the shaped tensor view exposed by ``memory_obj``."""
    tensor = memory_obj.tensor
    if tensor is None:
        raise ValueError("bytes-level KV access requires tensor-backed memory")
    if tensor.dtype != dtype:
        raise ValueError(f"memory dtype {tensor.dtype} does not match {dtype}")
    return tensor.reshape(shape)


def _get_single_group_layout(
    layout_desc: MemoryLayoutDesc,
) -> tuple[tuple[int, int, int, int], torch.dtype]:
    """Validate that ``layout_desc`` matches the v1 bytes-API contract."""
    if len(layout_desc.shapes) != 1:
        raise ValueError(
            "bytes-level KV access currently supports a single KV layer group "
            "(homogeneous attention only); hybrid-attention models with "
            f"multiple layer groups are not supported in v1 (got "
            f"{len(layout_desc.shapes)} groups)"
        )
    shape = layout_desc.shapes[0]
    if len(shape) != 4:
        raise ValueError(
            "bytes-level KV access requires the KV_2LTD layout "
            "(shape [2, num_layers, num_tokens, hidden_dim]); "
            f"got a {len(shape)}-D shape {tuple(shape)}"
        )
    return (shape[0], shape[1], shape[2], shape[3]), layout_desc.dtypes[0]


def _count_leading_complete_chunks(
    obj_keys: list[ObjectKey],
    written_keys: set[ObjectKey],
    world_size: int,
) -> int:
    """Count leading chunks where every worker shard is present."""
    stored_chunks = 0
    for chunk_start in range(0, len(obj_keys), world_size):
        chunk_keys = obj_keys[chunk_start : chunk_start + world_size]
        if len(chunk_keys) != world_size:
            break
        if not all(obj_key in written_keys for obj_key in chunk_keys):
            break
        stored_chunks += 1
    return stored_chunks

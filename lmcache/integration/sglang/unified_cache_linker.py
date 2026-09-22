# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree UnifiedCacheLinker backed by the LMCache multiprocess server.

SGLang owns the tree and all device allocation. This adapter owns MP
registrations, remote read leases, and the lifetime of asynchronous transfers.
One MP logical chunk represents one SGLang page (or one recurrent checkpoint).
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar
import hashlib
import json
import threading
import time
import uuid

# Third Party
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

logger = init_logger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


def _serialized(
    method: Callable[Concatenate[LMCacheLinker, P], R],
) -> Callable[Concatenate[LMCacheLinker, P], R]:
    @wraps(method)
    def call(self: LMCacheLinker, *args: P.args, **kwargs: P.kwargs) -> R:
        with self.state_lock:
            return method(self, *args, **kwargs)

    return call


def _content_hash(value: str) -> bytes:
    return hashlib.sha256(b"sglang-linker-v1\0" + value.encode()).digest()


def _ranges(indices: set[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index in sorted(indices):
        if result and result[-1][1] == index:
            result[-1] = (result[-1][0], index + 1)
        else:
            result.append((index, index + 1))
    return result


@dataclass
class _Pool:
    entry: Any
    model_name: str
    instance_id: int
    tensors: dict[str, torch.Tensor]
    context: TransferContext
    kernel_groups: int


@dataclass
class _Lease:
    pool: _Pool
    key: IPCCacheServerKey
    page_keys: tuple[str, ...]
    held: set[int] = field(default_factory=set)
    lookup_future: MessagingFuture[Any] | None = None
    status_future: MessagingFuture[Any] | None = None
    resolved: bool = False
    releases: dict[tuple[int, int], MessagingFuture[Any]] = field(default_factory=dict)


@dataclass
class _Retrieve:
    lease: _Lease
    start: int
    end: int
    block_ids: list[int]


@dataclass
class _Operation:
    future: MessagingFuture[bool]
    session_id: str
    event: Any


@dataclass
class _Batch:
    operations: list[_Operation]
    request_ids: list[str] = field(default_factory=list)
    waiting_streams: set[int] = field(default_factory=set)


class _LoadCounter:
    """All-layer event barrier; storage objects need not be layerwise."""

    def __init__(self, linker: LMCacheLinker) -> None:
        self.linker = linker
        self.consumer_index = -1

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def wait_until(self, threshold: int) -> None:
        # Also covers a load whose originating request was aborted after the
        # tree adopted its slots. A later matching request still needs it.
        self.linker.wait_for_loads()


class LMCacheLinker(UnifiedCacheLinker):
    """Use LMCache MP as an external store for SGLang's unified radix tree.

    Args:
        server_args: SGLang model/topology configuration.
        params: Worker-owned cache pools and allocation metadata.
        components: Active unified tree component types.
        extra_config: ``server_url`` (default tcp://127.0.0.1:5555),
            ``namespace``, ``timeout`` (seconds), and ``heartbeat_interval``.

    Raises:
        ValueError: Configuration, pool layout, or server chunk size is invalid.
        RuntimeError: Registration or a required load fails. Loads fail closed;
            uninitialized destinations are never acknowledged as reusable KV.
    """

    def __init__(
        self,
        server_args: Any,
        params: Any,
        *,
        components: set[Any],
        extra_config: dict[str, Any],
    ) -> None:
        unknown = set(extra_config) - {
            "server_url",
            "namespace",
            "timeout",
            "heartbeat_interval",
        }
        if unknown:
            raise ValueError(f"Unknown LMCache linker options: {sorted(unknown)}")
        self.timeout = float(extra_config.get("timeout", 60))
        heartbeat_interval = float(extra_config.get("heartbeat_interval", 10))
        if self.timeout <= 0 or heartbeat_interval <= 0:
            raise ValueError("timeout and heartbeat_interval must be positive")
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=params.page_size,
            params=params,
            components=components,
        )
        group = params.attn_tp_cache_group or params.tp_cache_group
        tp_rank, tp_size = 0, int(server_args.tp_size)
        if torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group)
            tp_size = torch.distributed.get_world_size(group)
        identity = {
            "schema": "sglang-linker-v1",
            "model": server_args.model_path,
            "revision": getattr(server_args, "revision", None),
            "namespace": str(extra_config.get("namespace", "default")),
            "page_size": params.page_size,
            "tp": [tp_rank, tp_size],
            "pp": [params.pp_rank, params.pp_size],
            "cp": [params.attn_cp_rank, params.attn_cp_size],
        }
        self.client = RequestClientFactory.create(
            str(extra_config.get("server_url", "tcp://127.0.0.1:5555"))
        )
        self.pools: dict[PoolName, _Pool] = {}
        self.state_lock = threading.RLock()
        self.layer_done_counter = _LoadCounter(self)
        self._lookups: dict[str, list[_Lease]] = {}
        # Own leases independently of request queues, including timed-out RPCs.
        self._leases: dict[str, _Lease] = {}
        self._queued: dict[str, tuple[list[_Retrieve], list[_Lease]]] = {}
        self._loads: deque[_Batch] = deque()
        self._stores: deque[_Batch] = deque()
        self._completed_loads: deque[list[str]] = deque()
        self._completed_stores: deque[bool] = deque()
        self._generation = 0
        self._closed = False
        self._fatal: BaseException | None = None
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        try:
            if self.client.get_chunk_size().result(self.timeout) != 1:
                raise ValueError(
                    "LMCacheLinker needs a dedicated MP server with --chunk-size 1; "
                    "one logical chunk contains one complete SGLang page."
                )
            for entry in self.pool_group.entries:
                tensors = entry.get_page_tensors()
                if not tensors or any(t.device.type != "cuda" for t in tensors):
                    raise ValueError("LMCacheLinker requires CUDA device page tensors")
                pool_identity = {
                    **identity,
                    "pool": str(entry.name),
                    "layout": entry.get_page_layout(),
                }
                model_name = (
                    "sglang-linker:"
                    + hashlib.sha256(
                        json.dumps(pool_identity, sort_keys=True).encode()
                    ).hexdigest()
                )
                instance_id = uuid.uuid4().int & ((1 << 63) - 1)
                caches = {f"page_{i}": t for i, t in enumerate(tensors)}
                buckets: dict[tuple[Any, ...], list[int]] = {}
                for i, tensor in enumerate(tensors):
                    buckets.setdefault((*tensor.shape[1:], tensor.dtype), []).append(i)
                specs = [
                    EngineGroupInfo(
                        engine_group_id=0,
                        layer_indices=tuple(indices),
                        tokens_per_block=1,
                    )
                    for indices in buckets.values()
                ]
                context = create_transfer_context(
                    caches,
                    instance_id=instance_id,
                    req_client=self.client,
                    mode="lmcache_driven",
                )
                pool = _Pool(
                    entry, model_name, instance_id, caches, context, len(specs)
                )
                self.pools[entry.name] = pool
                context.register(
                    caches,
                    model_name,
                    1,
                    1,
                    self.timeout,
                    layout_hints={"tokens_per_block": 1},
                    engine_group_infos=specs,
                    engine_type=EngineType.SGLANG,
                )
            self._heartbeat = threading.Thread(
                target=self._keep_alive,
                args=(heartbeat_interval,),
                daemon=True,
                name="lmcache-linker-heartbeat",
            )
            self._heartbeat.start()
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Cleaning up an incomplete linker registration failed")
            raise
        logger.info("LMCacheLinker registered %d device pools", len(self.pools))

    @_serialized
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Reserve every candidate page and return jointly restorable boundaries."""
        self._check_open()
        self.release_request(rid)
        expanded = self.pool_group.resolve_transfers(transfers)
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        if not expanded or kv is None or not kv.keys:
            return []
        keys = list(kv.keys)
        leases: list[_Lease] = []
        self._lookups[rid] = leases
        candidates = set(range(1, len(keys) + 1))
        try:
            for transfer in expanded:
                pool = self.pools[transfer.name]
                if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                    lease = self._lookup_pages(pool, keys)
                    leases.append(lease)
                    candidates.intersection_update(range(1, len(lease.held) + 1))
                else:
                    # Trailing windows can exist at sparse node boundaries.
                    # Probe pages independently; a missing earlier checkpoint
                    # must not hide a later restorable boundary.
                    window = len(transfer.keys or ())
                    present = set()
                    for index, key in enumerate(keys):
                        lease = self._lookup_pages(pool, [key])
                        leases.append(lease)
                        if lease.held:
                            present.add(index)
                    candidates = {
                        end
                        for end in candidates
                        if all(i in present for i in range(max(0, end - window), end))
                    }
                if not candidates:
                    break
        except BaseException:
            self.release_request(rid)
            raise
        if not candidates:
            self.release_request(rid)
        return sorted(candidates)

    @_serialized
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue destinations; hold remote leases until the next batch starts."""
        self._check_open()
        if rid in self._queued:
            raise ValueError(f"Duplicate queued LMCache load: {rid}")
        leases = self._lookups.get(rid, [])
        pages = {
            (lease.pool.entry.name, key): (lease, index)
            for lease in leases
            for index, key in enumerate(lease.page_keys)
            if index in lease.held
        }
        tasks: list[_Retrieve] = []
        for transfer in self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        ):
            pool = self.pools[transfer.name]
            blocks = pool.entry.get_page_indices(transfer.host_indices)
            keys = list(transfer.keys or ())
            if len(blocks) != len(keys):
                raise ValueError("Load keys and destination page counts differ")
            for key, block in zip(keys, blocks, strict=True):
                item = pages.get((transfer.name, key))
                if item is None:
                    return False
                lease, index = item
                if tasks and tasks[-1].lease is lease and tasks[-1].end == index:
                    tasks[-1].end += 1
                    tasks[-1].block_ids.append(block)
                else:
                    tasks.append(_Retrieve(lease, index, index + 1, [block]))
        if not tasks:
            return False
        self._lookups.pop(rid, None)
        self._queued[rid] = (tasks, leases)
        return True

    @_serialized
    def start_layer_wise_loading(self) -> int:
        """Submit queued retrieves and expose their all-layer event barrier."""
        self._check_open()
        if not self._queued:
            return self._generation if self._loads else -1
        queued, self._queued = self._queued, {}
        batch = _Batch([], list(queued))
        self._loads.append(batch)
        self._generation += 1
        try:
            for tasks, leases in queued.values():
                for task in tasks:
                    pool = task.lease.pool
                    key = replace(
                        task.lease.key, worker_id=0, start=task.start, end=task.end
                    )
                    event = pool.context.create_recorded_event()
                    future = pool.context.submit_retrieve(
                        key.request_id,
                        key,
                        pool.tensors,
                        [list(task.block_ids) for _ in range(pool.kernel_groups)],
                        event,
                        1,
                    )
                    batch.operations.append(_Operation(future, key.request_id, event))
                    task.lease.held.difference_update(range(task.start, task.end))
                self._release_leases(
                    leases,
                    retained_sessions={op.session_id for op in batch.operations},
                )
        except BaseException as error:
            # A submitted RPC can still be writing destinations. Never report
            # completion or drop the batch on an unknown transport outcome.
            self._fatal = error
            raise
        return self._generation

    @_serialized
    def wait_for_loads(self) -> None:
        """Order the current forward stream after all outstanding load batches."""
        self._check_open()
        stream = torch_dev.current_stream()
        stream_id = int(stream.cuda_stream)
        for batch in self._loads:
            if stream_id in batch.waiting_streams:
                continue
            for operation in batch.operations:
                future = operation.future
                ok = (
                    future.wait_on_stream(stream, self.timeout)
                    if isinstance(future, DeviceMessagingFuture)
                    else future.result(self.timeout)
                )
                if not ok:
                    self._fatal = RuntimeError(
                        "LMCache retrieve failed; KV is not ready"
                    )
                    raise self._fatal
            batch.waiting_streams.add(stream_id)

    @_serialized
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel only a queue the caller has already removed from its tree."""
        queued = self._queued.pop(rid, None)
        if queued is None:
            return False
        self._release_leases(queued[1])
        return True

    @_serialized
    def release_request(self, rid: str) -> None:
        """Release unused lookup pins; queued/in-flight loads keep their owners."""
        self._release_leases(self._lookups.pop(rid, []))

    @_serialized
    def num_completed_loads(self) -> int:
        """Count FIFO batches whose server response and GPU DMA both finished."""
        self._poll()
        return len(self._completed_loads)

    @_serialized
    def pop_completed_load(self) -> list[str]:
        """Acknowledge the oldest completed batch to SGLang's node-lock owner."""
        self._poll()
        return self._completed_loads.popleft()

    @_serialized
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Submit ordered GPU reads; acknowledge only after every pool completes."""
        self._check_open()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return False
        batch = _Batch([])
        self._stores.append(batch)
        try:
            for transfer in expanded:
                pool = self.pools[transfer.name]
                keys = list(transfer.keys or ())
                blocks = pool.entry.get_page_indices(transfer.host_indices)
                if len(keys) != len(blocks):
                    raise ValueError("Store keys and source page counts differ")
                key = self._make_key(pool, keys)
                event = pool.context.create_recorded_event()
                future = pool.context.submit_store(
                    key.request_id,
                    key,
                    pool.tensors,
                    [list(blocks) for _ in range(pool.kernel_groups)],
                    event,
                    1,
                )
                batch.operations.append(_Operation(future, key.request_id, event))
        except BaseException as error:
            self._fatal = error
            raise
        return True

    @_serialized
    def num_completed_offloads(self) -> int:
        """Count completed stores without releasing any in-flight source slots."""
        self._poll()
        return len(self._completed_stores)

    @_serialized
    def pop_completed_offload(self) -> bool:
        """Consume the oldest store's all-pool success result."""
        self._poll()
        return self._completed_stores.popleft()

    @_serialized
    def reset(self) -> None:
        """Drain device work and leases while preserving external cached pages."""
        for rid in list(self._queued):
            self.cancel_queued_load(rid)
        for rid in list(self._lookups):
            self.release_request(rid)
        for batch in (*self._loads, *self._stores):
            for operation in batch.operations:
                operation.future.result(self.timeout)
        # Do not swallow transfer errors: the caller must not recycle slots
        # after a timeout whose remote GPU-write outcome is unknown.
        self._poll()
        self._release_leases(list(self._leases.values()))
        self._completed_loads.clear()
        self._completed_stores.clear()
        self.layer_done_counter.consumer_index = -1

    @_serialized
    def close(self) -> None:
        """Drain work, unregister GPU allocations, and close transport resources."""
        if self._closed:
            return
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=self.timeout + 1)
        for rid in list(self._queued):
            self.cancel_queued_load(rid)
        for rid in list(self._lookups):
            self.release_request(rid)
        # Closing after a negative retrieve result is safe once its device
        # event completes. A timeout/transport exception still propagates:
        # unregistering memory while its remote DMA outcome is unknown is not.
        for batch in (*self._loads, *self._stores):
            for operation in batch.operations:
                operation.future.result(self.timeout)
            for session_id in {op.session_id for op in batch.operations}:
                self._finish_session(session_id)
        self._loads.clear()
        self._stores.clear()
        self._release_leases(list(self._leases.values()))
        for pool in self.pools.values():
            future = pool.context.unregister()
            if future is not None:
                future.result(self.timeout)
            pool.context.close()
        self.client.close()
        self._closed = True

    def _make_key(self, pool: _Pool, keys: list[str]) -> IPCCacheServerKey:
        return IPCCacheServerKey(
            model_name=pool.model_name,
            world_size=1,
            worker_id=0,
            token_ids=(),
            start=0,
            end=len(keys),
            request_id="sglang-linker-" + uuid.uuid4().hex,
            num_kv_readers=1,
            chunk_hashes=tuple(map(_content_hash, keys)),
        )

    def _lookup_pages(self, pool: _Pool, keys: list[str]) -> _Lease:
        key = self._make_key(pool, keys).no_worker_id_version()
        lease = _Lease(pool, key, tuple(keys))
        self._leases[key.request_id] = lease
        lease.lookup_future = self.client.lookup(key, 1)
        self._resolve_lease(lease)
        return lease

    def _resolve_lease(self, lease: _Lease) -> None:
        if lease.resolved:
            return
        if lease.lookup_future is None:
            return
        lease.lookup_future.result(self.timeout)
        deadline = time.monotonic() + self.timeout
        while True:
            if lease.status_future is None:
                lease.status_future = self.client.query_prefetch_status(
                    lease.key.request_id
                )
            count = lease.status_future.result(self.timeout)
            lease.status_future = None
            if count is not None:
                lease.held = set(range(int(count)))
                lease.resolved = True
                return
            if time.monotonic() >= deadline:
                raise LMCacheTimeoutError("LMCache linker prefetch timed out")
            time.sleep(0.001)

    def _release_leases(
        self, leases: list[_Lease], *, retained_sessions: set[str] | None = None
    ) -> None:
        for lease in leases:
            self._resolve_lease(lease)
            for start, end in _ranges(lease.held):
                span = (start, end)
                if span not in lease.releases:
                    lease.releases[span] = self.client.free_lookup_locks(
                        replace(lease.key, start=start, end=end), 1
                    )
                # Retain an unacknowledged unlock rather than send it twice:
                # read locks are shared counters, not idempotent lease IDs.
                lease.releases[span].result(self.timeout)
                lease.held.difference_update(range(start, end))
                del lease.releases[span]
            if lease.key.request_id not in (retained_sessions or ()):
                self.client.end_session(lease.key.request_id).result(self.timeout)
                self._leases.pop(lease.key.request_id, None)

    def _poll(self) -> None:
        self._check_open()
        for queue, loading in ((self._loads, True), (self._stores, False)):
            while queue:
                batch = queue[0]
                if not all(op.future.query() for op in batch.operations):
                    break
                results = [op.future.result(0) for op in batch.operations]
                if loading and not all(results):
                    self._fatal = RuntimeError(
                        "LMCache retrieve failed; refusing KV reuse"
                    )
                    raise self._fatal
                for session_id in {op.session_id for op in batch.operations}:
                    self._finish_session(session_id)
                queue.popleft()
                if loading:
                    self._completed_loads.append(batch.request_ids)
                else:
                    self._completed_stores.append(all(results))

    def _finish_session(self, session_id: str) -> None:
        lease = self._leases.get(session_id)
        if lease is not None:
            self._release_leases([lease])
        else:
            self.client.end_session(session_id).result(self.timeout)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("LMCacheLinker is closed")
        if self._fatal is not None:
            raise RuntimeError(
                "LMCacheLinker transfer state is unsafe"
            ) from self._fatal

    def _keep_alive(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                for pool in self.pools.values():
                    if not self.client.ping(pool.instance_id).result(self.timeout):
                        raise RuntimeError("LMCache server lost a GPU registration")
            except BaseException as error:
                self._fatal = error
                logger.exception("LMCache linker heartbeat failed")
                return

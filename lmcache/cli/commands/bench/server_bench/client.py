# SPDX-License-Identifier: Apache-2.0
"""Client and data models for ``lmcache bench server``."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Any, Literal
import hashlib
import time

# Third Party
import zmq

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.bench.server_bench.allocator import (
    BlockAllocator,
    GroupBlocks,
    RequestBlocks,
)
from lmcache.cli.commands.bench.server_bench.buffers import KVCacheBuffers
from lmcache.cli.commands.bench.server_bench.config import (
    BenchConfig,
    WorkerSpec,
)
from lmcache.cli.commands.bench.server_bench.helpers import (
    _DEFAULT_RPC_TIMEOUT_S,
    _INSTANCE_ID_BASE,
    _MODEL_NAME,
    _TIMEOUT,
    _get_chunk_size,
    _make_key,
    _poll_prefetch_status,
    _query_checksum,
    _send_end_session,
    _send_lookup,
    _wait_for_result,
)
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import (
    DTYPE_MAP,
    format_kvcache_shape_spec,
    parse_kvcache_shape_spec,
)
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    create_transfer_context,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.transfer_context import TransferContext
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import IPCEvent
    from lmcache.v1.multiprocess.transport.base import RequestClient


@dataclass(frozen=True)
class LookupResult:
    """Result of one LOOKUP, including its hit range and latency."""

    hit_chunks: int
    total_chunks: int
    latency_ms: float
    error: str | None = None

    def __post_init__(self) -> None:
        if self.total_chunks < 0:
            raise ValueError(
                f"total_chunks must be non-negative, got {self.total_chunks}"
            )
        if not 0 <= self.hit_chunks <= self.total_chunks:
            raise ValueError(
                "hit_chunks must be between zero and total_chunks, got "
                f"{self.hit_chunks}/{self.total_chunks}"
            )
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

    @property
    def succeeded(self) -> bool:
        """Return whether LOOKUP completed without an error."""
        return self.error is None

    @property
    def is_full_hit(self) -> bool:
        """Return whether every requested chunk was found."""
        return (
            self.succeeded
            and self.total_chunks > 0
            and (self.hit_chunks == self.total_chunks)
        )

    @property
    def is_full_miss(self) -> bool:
        """Return whether no requested chunk was found."""
        return self.succeeded and self.total_chunks > 0 and self.hit_chunks == 0

    @property
    def is_partial_hit(self) -> bool:
        """Return whether LOOKUP found some but not all requested chunks."""
        return self.succeeded and 0 < self.hit_chunks < self.total_chunks


@dataclass(frozen=True)
class TransferResult:
    """Aggregate STORE or RETRIEVE result for its target Worker ranks."""

    operation: str
    token_count: int
    latency_ms: float
    attempted_worker_ranks: tuple[int, ...]
    successful_worker_ranks: tuple[int, ...]
    failed_worker_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError(
                f"token_count must be non-negative, got {self.token_count}"
            )
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

        attempted = set(self.attempted_worker_ranks)
        successful = set(self.successful_worker_ranks)
        failed = set(self.failed_worker_ranks)
        if len(attempted) != len(self.attempted_worker_ranks):
            raise ValueError("attempted_worker_ranks must not contain duplicates")
        if len(successful) != len(self.successful_worker_ranks):
            raise ValueError("successful_worker_ranks must not contain duplicates")
        if len(failed) != len(self.failed_worker_ranks):
            raise ValueError("failed_worker_ranks must not contain duplicates")
        if successful & failed:
            raise ValueError("successful and failed Worker ranks must be disjoint")
        if attempted != successful | failed:
            raise ValueError(
                "successful and failed Worker ranks must exactly partition "
                "attempted_worker_ranks"
            )

    @property
    def succeeded(self) -> bool:
        """Return whether every attempted Worker completed successfully."""
        return not self.failed_worker_ranks


@dataclass(frozen=True)
class RequestContext:
    """Tokens and request-exclusive group blocks shared by all TP workers."""

    sequence_id: int
    request_id: str
    request_kind: str
    token_ids: tuple[int, ...]
    num_full_tokens: int
    total_chunks: int
    chunk_size: int
    block_size: int
    block_allocation: RequestBlocks

    def blocks(self, start_token: int, token_count: int) -> RequestBlocks:
        """Slice all group allocations by a validated token range."""
        start = start_token // self.block_size
        end = (start_token + token_count) // self.block_size
        return RequestBlocks(
            tuple(
                GroupBlocks(group.group_id, group.block_ids[start:end])
                for group in self.block_allocation.groups
            )
        )


@dataclass
class WorkerContext:
    """Worker routing, owned KV buffers, and their registered transfer context."""

    spec: WorkerSpec
    buffers: "KVCacheBuffers"
    transfer_context: "TransferContext"


class ServerBenchClient:
    """Manage Workers and data-plane operations for one benchmark run."""

    def __init__(
        self,
        config: BenchConfig,
        log: Callable[[str], None],
        client_index: int = 0,
    ) -> None:
        """Create a client without connecting to the Server.

        Args:
            config: Benchmark configuration.
            log: Progress logger.
            client_index: Position among concurrent clients. Each client
                registers its Workers under its own block of instance IDs.

        Raises:
            ValueError: If ``client_index`` is negative.
        """
        if client_index < 0:
            raise ValueError(f"client_index must be non-negative, got {client_index}")
        self._config = config
        self._log = log
        self._client_index = client_index
        self._zmq_context: Any | None = None
        self._req_client: "RequestClient | None" = None
        self._workers: list[WorkerContext] = []
        self._transfer_executor: ThreadPoolExecutor | None = None
        self._transfer_uncertain = False
        self._started = False
        self._requests: dict[int, RequestContext] = {}
        self._allocator = BlockAllocator(0, 0)

        self._chunk_size = 0
        self._block_size = 0
        self._blocks_in_chunk = 0
        self._num_blocks = 0
        self._num_engine_group_infos = 0
        self._kv_world_size = 0

    def start(self) -> None:
        """Connect, allocate worker-owned KV buffers, and register Workers.

        Raises:
            ValueError: If the KV layout is invalid.
            RuntimeError: If startup fails or the client is already started.
        """
        if self._started:
            raise RuntimeError("ServerBenchClient has already been started")

        try:
            self._initialize()
            self._transfer_executor = ThreadPoolExecutor(max_workers=len(self._workers))
            # Start all transfer threads outside measured operations.
            ready = Event()
            try:
                warmup = [
                    self._transfer_executor.submit(ready.wait) for _ in self._workers
                ]
            finally:
                ready.set()
            for future in warmup:
                future.result(timeout=self._config.transfer_timeout_s)
        except BaseException:
            self.close()
            raise

        self._allocator = BlockAllocator(
            self._num_engine_group_infos,
            self._num_blocks,
            allocation_order=self._config.block_allocation,
        )
        self._started = True

    def create_request(
        self,
        sequence_id: int,
        request_id: str,
        request_kind: str,
        *,
        token_ids: Sequence[int],
    ) -> RequestContext | None:
        """Create a request with exclusive blocks reserved until release_request.

        Args:
            sequence_id: Sequence ID used in logs.
            request_id: Server session ID.
            request_kind: Request kind used in logs.
            token_ids: Request tokens, rounded down to full chunks.

        Returns:
            The request, or ``None`` if it has no full chunk.

        Raises:
            RuntimeError: If the client is unavailable or free blocks are insufficient.
        """
        self._require_started()
        token_ids = tuple(token_ids)
        num_full_tokens = (len(token_ids) // self._chunk_size) * self._chunk_size
        if num_full_tokens == 0:
            self._log(
                "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
                % (sequence_id, request_kind, len(token_ids), self._chunk_size)
            )
            return None

        num_blocks = num_full_tokens // self._block_size
        binding = self._allocator.allocate(num_blocks)
        request = RequestContext(
            sequence_id=sequence_id,
            request_id=request_id,
            request_kind=request_kind,
            token_ids=token_ids,
            num_full_tokens=num_full_tokens,
            total_chunks=num_full_tokens // self._chunk_size,
            chunk_size=self._chunk_size,
            block_size=self._block_size,
            block_allocation=binding,
        )
        self._requests[id(request)] = request
        return request

    def release_request(self, request: RequestContext) -> None:
        """End the session and free blocks without clearing KV; invalidate request."""
        if not self._started:
            raise RuntimeError("ServerBenchClient must be started before use")
        self._validate_request(request)
        try:
            if not self._transfer_uncertain:
                _send_end_session(self._require_started(), request.request_id)
        finally:
            if not self._transfer_uncertain:
                self._allocator.release(request.block_allocation)
            # Uncertain transfers keep blocks reserved until close.
            del self._requests[id(request)]

    def lookup(self, request: RequestContext) -> LookupResult:
        """Run LOOKUP and wait for its hit count.

        Args:
            request: Request to look up.

        Returns:
            LOOKUP result and latency.

        Raises:
            RuntimeError: If the client is not started.
        """
        req_client = self._check_request_ready(request)

        lookup_key = _make_key(
            request.token_ids,
            request.request_id,
            start=0,
            end=request.num_full_tokens,
            world_size=self._kv_world_size,
            num_kv_readers=sum(w.spec.retrieve_enabled for w in self._workers)
            // self._kv_world_size,
        )
        started_at = time.monotonic()
        if not _send_lookup(req_client, lookup_key, tp_size=len(self._workers)):
            latency_ms = (time.monotonic() - started_at) * 1000
            self._log(
                "  [seq %d/%s] LOOKUP timeout"
                % (request.sequence_id, request.request_kind)
            )
            return LookupResult(
                hit_chunks=0,
                total_chunks=request.total_chunks,
                latency_ms=latency_ms,
                error="timeout",
            )

        hit_chunks = _poll_prefetch_status(req_client, lookup_key.request_id)
        if hit_chunks is None:
            hit_chunks = 0
        latency_ms = (time.monotonic() - started_at) * 1000
        self._log(
            "  [seq %d/%s] LOOKUP: %d/%d chunks hit (%.1f ms)"
            % (
                request.sequence_id,
                request.request_kind,
                hit_chunks,
                request.total_chunks,
                latency_ms,
            )
        )
        return LookupResult(
            hit_chunks=hit_chunks,
            total_chunks=request.total_chunks,
            latency_ms=latency_ms,
        )

    def store(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """STORE a token range concurrently on write-enabled workers.

        Args:
            request: Request to store.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Returns:
            Worker results, including failed ranks on timeout, or ``None``
            for an empty range. A timeout or transfer exception prevents
            further operations until the client is closed.

        Raises:
            RuntimeError: If the request is inactive or the client is unavailable.
            ValueError: If a non-empty range is invalid or not chunk-aligned.
        """
        return self._transfer("store", request, start_token, token_count)

    def retrieve(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """RETRIEVE a token range concurrently on read-enabled workers.

        Args:
            request: Request to retrieve.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Returns:
            Worker results, including failed ranks on timeout, or ``None``
            for an empty range. A timeout or transfer exception prevents
            further operations until the client is closed.

        Raises:
            RuntimeError: If the request is inactive or the client is unavailable.
            ValueError: If a non-empty range is invalid or not chunk-aligned.
        """
        return self._transfer("retrieve", request, start_token, token_count)

    def fill_deterministic_kv(self, request: RequestContext) -> None:
        """Fill request blocks with reproducible pseudorandom KV, then synchronize."""
        self._check_request_ready(request)
        for worker in self._workers:
            worker.buffers.fill_deterministic_kv(
                request.block_allocation.as_block_ids(),
                request.token_ids[: request.num_full_tokens],
                worker.spec.kv_worker_id,
            )
            worker.buffers.synchronize()

    def zero_destination(
        self, request: RequestContext, start_token: int, token_count: int
    ) -> None:
        """Clear and synchronize reader destination blocks before RETRIEVE.

        Args:
            request: Request to clear.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Raises:
            RuntimeError: If the client is not started.
            ValueError: If the range is invalid or not chunk-aligned.
        """
        self._check_request_ready(request)
        self._validate_token_range(request, start_token, token_count)
        for worker in self._workers:
            if worker.spec.retrieve_enabled:
                worker.buffers.set_blocks_to_zero(
                    request.blocks(start_token, token_count).as_block_ids()
                )
                worker.buffers.synchronize()

    def compute_checksums(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
        *,
        operation: Literal["store", "retrieve"],
    ) -> dict[int, list[str]] | None:
        """Compute per-receiver checksums for a request range.

        Both transfer modes hash local worker buffers.
        Store checksums provide the reference for a subsequent retrieve:
        MLA receivers share rank 0's reference, while other TP ranks use
        their own writer's reference.

        Args:
            request: Request to hash.
            start_token: Start offset in the request.
            token_count: Number of tokens.
            operation: ``"store"`` hashes writers before STORE;
                ``"retrieve"`` hashes receivers after RETRIEVE.

        Returns:
            Receiver rank to per-chunk digests, or ``None`` for an empty
            range or if any required checksum is unavailable.

        Raises:
            RuntimeError: If the client is not started.
            ValueError: If a non-empty range is invalid or not chunk-aligned.
        """
        self._check_request_ready(request)
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        workers = [
            worker
            for worker in self._workers
            if (
                worker.spec.store_enabled
                if operation == "store"
                else worker.spec.retrieve_enabled
            )
        ]
        checksums: dict[int, list[str]] = {}
        for worker in workers:
            checksums[worker.spec.rank] = [
                worker.buffers.checksum(
                    request.blocks(offset, self._chunk_size).as_block_ids()
                )
                for offset in range(
                    start_token, start_token + token_count, self._chunk_size
                )
            ]

        if operation == "store":
            writer_checksums = {
                worker.spec.kv_worker_id: checksums[worker.spec.rank]
                for worker in workers
            }
            checksums = {
                worker.spec.rank: writer_checksums[worker.spec.kv_worker_id]
                for worker in self._workers
                if worker.spec.retrieve_enabled
            }

        if checksums:
            all_chunks = [digest for parts in checksums.values() for digest in parts]
            digest = hashlib.md5("".join(all_chunks).encode()).hexdigest()[:16]
            self._log(
                "  [seq %d/%s] CHECKSUM: %s (%d ranks, %d chunks)"
                % (
                    request.sequence_id,
                    request.request_kind,
                    digest,
                    len(checksums),
                    len(all_chunks),
                )
            )
        return checksums or None

    def verify_server_checksums(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
        *,
        operation: Literal["store", "retrieve"],
    ) -> bool | None:
        """Compare local and HTTP digests; return None for empty or disabled checks."""
        self._check_request_ready(request)
        self._validate_token_range(request, start_token, token_count)
        if token_count == 0:
            return None
        if not self._config.uses_handle_transfer or not self._config.http_url:
            self._log("  SERVER CHECKSUM SKIP: requires lmcache_driven mode and --url")
            return None

        allocation = request.blocks(start_token, token_count)
        matched = True
        for worker in self._workers:
            enabled = (
                worker.spec.store_enabled
                if operation == "store"
                else worker.spec.retrieve_enabled
            )
            if not enabled:
                continue
            expected: dict[int, list[str]] = {}
            for offset in range(
                start_token, start_token + token_count, self._chunk_size
            ):
                for index, digest in worker.buffers.tensor_checksums(
                    request.blocks(offset, self._chunk_size).as_block_ids()
                ).items():
                    expected.setdefault(index, []).append(digest)
            queried: dict[tuple[int, ...], dict[str, list[str]] | None] = {}
            for group, blocks in zip(
                worker.buffers.engine_group_infos, allocation.groups, strict=True
            ):
                if blocks.block_ids not in queried:
                    queried[blocks.block_ids] = _query_checksum(
                        self._config.http_url,
                        list(blocks.block_ids),
                        self._blocks_in_chunk,
                        worker.spec.instance_id,
                    )
                actual = queried[blocks.block_ids]
                # The legacy endpoint hashes every tensor; select this group's results.
                for index in group.layer_indices:
                    if (
                        actual is None
                        or actual.get(f"layer_{index}") != expected[index]
                    ):
                        matched = False
                        self._log(
                            "  SERVER CHECKSUM FAIL: rank %d tensor %d (%s)"
                            % (worker.spec.rank, index, operation)
                        )
        self._log("  SERVER CHECKSUM %s" % ("OK" if matched else "FAIL"))
        return matched

    def close(self) -> None:
        """Wait for transfer threads, then synchronously release resources.

        Cleanup is idempotent, including after partial startup. A blocked
        native transfer can delay shutdown until it returns.
        """
        self._started = False
        if self._transfer_executor is not None:
            self._transfer_executor.shutdown(wait=True, cancel_futures=True)
            self._transfer_executor = None

        if self._req_client is not None:
            for worker in self._workers:
                instance_id = worker.spec.instance_id
                try:
                    worker.transfer_context.flush_inflight_stores()
                except Exception as exc:
                    self._log(
                        "  [warning] TransferContext flush failed for iid %d: %s"
                        % (instance_id, exc)
                    )
                try:
                    future = worker.transfer_context.unregister()
                    if future is not None:
                        future.result(timeout=_DEFAULT_RPC_TIMEOUT_S)
                        self._log("[iid %d] UNREGISTER_KV_CACHE: OK" % instance_id)
                except Exception as exc:
                    self._log(
                        "  [warning] UNREGISTER_KV_CACHE failed for "
                        "iid %d: %s" % (instance_id, exc)
                    )
        for worker in self._workers:
            try:
                worker.transfer_context.close()
            except Exception as exc:
                self._log(
                    "  [warning] TransferContext close failed for iid %d: %s"
                    % (worker.spec.instance_id, exc)
                )

        # Drop tensors after contexts release their local transfer resources.
        for worker in self._workers:
            worker.buffers.release()

        if self._req_client is not None:
            try:
                self._req_client.close()
            except Exception as exc:
                self._log("  [warning] Request client close failed: %s" % exc)
            finally:
                self._req_client = None
        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception as exc:
                self._log("  [warning] ZMQ context close failed: %s" % exc)
            finally:
                self._zmq_context = None

        self._workers.clear()
        self._requests.clear()
        self._allocator = BlockAllocator(0, 0)
        self._transfer_uncertain = False

    def _transfer(
        self,
        operation: Literal["store", "retrieve"],
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """Run either public transfer operation with shared validation and waiting."""
        self._check_request_ready(request)
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        workers = [
            worker
            for worker in self._workers
            if (
                worker.spec.store_enabled
                if operation == "store"
                else worker.spec.retrieve_enabled
            )
        ]
        attempted = [worker.spec.rank for worker in workers]
        successful: list[int] = []
        failed: list[int] = []
        success_status = "stored" if operation == "store" else "retrieved"
        status = success_status
        timed_out = False
        started_at = time.monotonic()
        assert self._transfer_executor is not None
        deadline = started_at + self._config.transfer_timeout_s
        pending: list[Future[str]] = []
        for worker in workers:
            # Record on the caller's producer stream before dispatching.
            event = worker.transfer_context.create_recorded_event()
            self._transfer_uncertain = True
            pending.append(
                self._transfer_executor.submit(
                    self._transfer_worker,
                    worker,
                    operation,
                    request,
                    start_token,
                    token_count,
                    event,
                    deadline,
                )
            )
        for worker, future in zip(workers, pending, strict=True):
            try:
                worker_status = future.result(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except (FuturesTimeoutError, TimeoutError):
                future.cancel()
                worker_status = "timeout"
            if worker_status == success_status:
                successful.append(worker.spec.rank)
            else:
                failed.append(worker.spec.rank)
                status = worker_status
            timed_out |= worker_status == "timeout"
        self._transfer_uncertain = timed_out
        if timed_out:
            status = "timeout"
        latency_ms = (time.monotonic() - started_at) * 1000
        result = TransferResult(
            operation=operation,
            token_count=token_count,
            latency_ms=latency_ms,
            attempted_worker_ranks=tuple(attempted),
            successful_worker_ranks=tuple(successful),
            failed_worker_ranks=tuple(failed),
        )
        self._log(
            "  [seq %d/%s] %s: %s (%d tokens, %.1f ms, %d %s)"
            % (
                request.sequence_id,
                request.request_kind,
                operation.upper(),
                status,
                token_count,
                latency_ms,
                len(attempted),
                "writers" if operation == "store" else "workers",
            )
        )
        return result

    def _transfer_worker(
        self,
        worker: WorkerContext,
        operation: Literal["store", "retrieve"],
        request: RequestContext,
        start_token: int,
        token_count: int,
        event: IPCEvent | None,
        deadline: float,
    ) -> str:
        """Submit and wait on one worker; its submit may itself block."""
        key = _make_key(
            request.token_ids,
            request.request_id,
            start=start_token,
            end=start_token + token_count,
            worker_id=worker.spec.kv_worker_id,
            world_size=self._kv_world_size,
        )
        block_ids = request.blocks(start_token, token_count).as_block_ids()
        submit = (
            worker.transfer_context.submit_store
            if operation == "store"
            else worker.transfer_context.submit_retrieve
        )
        device = next(iter(worker.buffers.kv_caches.values())).device
        with torch_dev.device(device) if device.type != "cpu" else nullcontext():
            future = submit(
                request.request_id,
                key,
                worker.buffers.kv_caches,
                block_ids,
                event,
                self._blocks_in_chunk,
            )
            reply = _wait_for_result(
                future,
                timeout_s=max(0.0, deadline - time.monotonic()),
                retain_refs=(event,) if event is not None else (),
            )
        if reply is _TIMEOUT:
            return "timeout"
        return (
            ("stored" if operation == "store" else "retrieved")
            if reply
            else f"{operation}_failed"
        )

    def _initialize(self) -> None:
        """Allocate resources and register Workers."""

        config = self._config
        use_gpu = config.is_gpu
        use_handle = config.uses_handle_transfer
        if use_gpu and not torch_dev.is_available():
            raise RuntimeError("--mode gpu requires CUDA")
        if use_handle and not use_gpu:
            self._log(
                "  [info] --transfer-mode=lmcache_driven on cpu mode: "
                "using REGISTER_KV_CACHE + STORE/RETRIEVE over POSIX SHM"
            )

        self._log(
            "Connecting to LMCache MP Server at %s (mode=%s) ..."
            % (config.rpc_url, config.mode)
        )
        self._zmq_context = zmq.Context()
        self._zmq_context.setsockopt(zmq.LINGER, 0)
        self._req_client = RequestClientFactory.create(
            config.rpc_url,
            context=self._zmq_context,
        )

        self._chunk_size = _get_chunk_size(self._req_client)
        self._log("Server chunk_size = %d" % self._chunk_size)

        tensor_groups = parse_kvcache_shape_spec(config.kvcache_shape_spec)
        self._num_engine_group_infos = len(tensor_groups) or 1
        self._log(
            "Resolved KV shape spec: %s" % format_kvcache_shape_spec(tensor_groups)
        )

        first = tensor_groups[0]
        nb_vals = {group.shape_desc.nb for group in tensor_groups}
        bs_vals = {group.shape_desc.bs for group in tensor_groups}
        if len(nb_vals) > 1 or len(bs_vals) > 1:
            raise ValueError(
                "All groups must share NB and BS (paged KV requires uniform "
                "block geometry). Got NB=%s BS=%s" % (sorted(nb_vals), sorted(bs_vals))
            )

        num_tensors = sum(group.num_layers for group in tensor_groups)
        spec_nb = getattr(first.shape_desc, "nb", 0) or 0
        spec_bs = getattr(first.shape_desc, "bs", 0) or 0
        self._num_blocks = spec_nb if spec_nb > 0 else config.num_blocks
        self._block_size = spec_bs if spec_bs > 0 else config.block_size
        if self._chunk_size % self._block_size != 0:
            raise ValueError(
                "Server chunk_size %d must be a multiple of block_size %d"
                % (self._chunk_size, self._block_size)
            )
        self._blocks_in_chunk = self._chunk_size // self._block_size
        if spec_nb and spec_nb != config.num_blocks:
            self._log(
                "  [info] spec nb=%d overrides --num-blocks=%d"
                % (spec_nb, config.num_blocks)
            )
        if spec_bs and spec_bs != config.block_size:
            self._log(
                "  [info] spec bs=%d overrides --block-size=%d"
                % (spec_bs, config.block_size)
            )

        heads_set = {group.shape_desc.nh for group in tensor_groups}
        hs_set = {group.shape_desc.hs for group in tensor_groups}
        kv_size_set = {group.shape_desc.kv_size for group in tensor_groups}
        dtype_set = {group.dtype for group in tensor_groups}
        num_heads_display: int | str = (
            first.shape_desc.nh if len(heads_set) == 1 else "mixed"
        )
        head_size_display: int | str = (
            first.shape_desc.hs if len(hs_set) == 1 else "mixed"
        )
        kv_size_display: int | str = (
            first.shape_desc.kv_size if len(kv_size_set) == 1 else "mixed"
        )
        if len(dtype_set) == 1:
            dtype_string = next(
                (name for name, dtype in DTYPE_MAP.items() if dtype == first.dtype),
                "float16",
            )
        else:
            dtype_string = "mixed"

        layout_hints: LayoutHints = {"kv_layout": "NHD"}

        self._log(
            "KV shape: %d tensors, %s heads x %s, dtype=%s, "
            "blocks=%dx%d, kv=%s"
            % (
                num_tensors,
                num_heads_display,
                head_size_display,
                dtype_string,
                self._num_blocks,
                self._block_size,
                kv_size_display,
            )
        )

        use_mla = config.use_mla or (
            isinstance(kv_size_display, int) and kv_size_display == 1
        )
        if use_mla and isinstance(kv_size_display, int) and kv_size_display != 1:
            raise ValueError(
                "--use-mla requires --kvcache-shape-spec with kv_size=1 "
                "(single-plane MLA group), got kv_size=%s" % kv_size_display
            )
        self._kv_world_size = 1 if use_mla else config.tp_size

        for rank in range(config.tp_size):
            instance_id = _INSTANCE_ID_BASE + self._client_index * config.tp_size + rank
            kv_worker_id = 0 if use_mla else rank
            buffers = KVCacheBuffers(
                groups=tensor_groups,
                device=None if use_gpu else "cpu",
                block_size=self._block_size,
            )
            kv_caches = buffers.kv_caches
            self._log(
                "[rank %d] Allocated %d KV tensors on %s"
                % (rank, len(kv_caches), next(iter(kv_caches.values())).device)
            )
            # CPU bench tensors can coexist with process-global accelerator
            # support. Keep this workaround local until factory dispatch is
            # made device-aware.
            transfer_context: TransferContext
            if not use_gpu and config.transfer_mode in ("auto", "engine_driven"):
                transfer_context = EngineDrivenTransferContext(
                    instance_id, self._req_client
                )
            else:
                transfer_context = create_transfer_context(
                    kv_caches,
                    mode=config.transfer_mode,
                    instance_id=instance_id,
                    req_client=self._req_client,
                )

            worker = WorkerContext(
                spec=WorkerSpec(
                    rank=rank,
                    kv_worker_id=kv_worker_id,
                    kv_world_size=self._kv_world_size,
                    instance_id=instance_id,
                    store_enabled=(rank == 0) if use_mla else True,
                    retrieve_enabled=True,
                ),
                buffers=buffers,
                transfer_context=transfer_context,
            )
            self._workers.append(worker)

            if (
                isinstance(transfer_context, EngineDrivenTransferContext)
                and len(tensor_groups) > 1
            ):
                raise ValueError(
                    "engine_driven supports only one KV group; "
                    "use --transfer-mode lmcache_driven for multiple groups"
                )

            try:
                transfer_context.register(
                    kv_caches,
                    _MODEL_NAME,
                    self._kv_world_size,
                    self._blocks_in_chunk,
                    _DEFAULT_RPC_TIMEOUT_S,
                    layout_hints=layout_hints,
                    engine_group_infos=buffers.engine_group_infos,
                    engine_type=EngineType.VLLM,
                )
            except TimeoutError:
                raise RuntimeError(
                    "REGISTER_KV_CACHE failed for rank %d (instance_id=%d)"
                    % (rank, instance_id)
                ) from None
            self._log("[rank %d] REGISTER_KV_CACHE: OK" % rank)

        self._log("")

    def _require_started(self) -> "RequestClient":
        """Return the live multiprocess client or raise before startup."""
        if not self._started or self._req_client is None:
            raise RuntimeError("ServerBenchClient must be started before use")
        if self._transfer_uncertain:
            raise RuntimeError(
                "transfer completion is uncertain; close the client before reuse"
            )
        return self._req_client

    def _check_request_ready(self, request: RequestContext) -> "RequestClient":
        self._validate_request(request)
        return self._require_started()

    def _validate_request(self, request: RequestContext) -> None:
        if self._requests.get(id(request)) is not request:
            raise RuntimeError("request is not active on this client")

    def _validate_token_range(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> None:
        """Validate one chunk-aligned range within ``request``."""
        if start_token < 0 or token_count < 0:
            raise ValueError("token range must be non-negative")
        if start_token + token_count > request.num_full_tokens:
            raise ValueError("token range exceeds request")
        if start_token % self._chunk_size or token_count % self._chunk_size:
            raise ValueError("token range must be chunk-aligned")

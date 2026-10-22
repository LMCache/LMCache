# SPDX-License-Identifier: Apache-2.0
"""Client and data models for ``lmcache bench server``."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
import hashlib
import time

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.bench.server_bench.config import (
    BenchConfig,
    WorkerSpec,
)
from lmcache.cli.commands.bench.server_bench.helpers import _IMPORT_ERROR

# The thin CLI can load its parser without the full runtime dependencies.
if _IMPORT_ERROR is None:
    # First Party
    from lmcache.cli.commands.bench.server_bench.buffers import KVCacheBuffers

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.transfer_context import TransferContext
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
    """Resolved token and block geometry for one request."""

    sequence_id: int
    request_id: str
    request_kind: str
    token_ids: tuple[int, ...]
    num_full_tokens: int
    total_chunks: int
    chunk_size: int
    block_offset: int
    num_blocks: int


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
        self._started = False

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
        except BaseException:
            self.close()
            raise

        self._started = True

    def create_request(
        self,
        sequence_id: int,
        request_id: str,
        request_kind: str,
    ) -> RequestContext | None:
        """Create a request with resolved token and block ranges.

        Args:
            sequence_id: Sequence ID used to generate tokens and block offsets.
            request_id: Server session ID.
            request_kind: Request kind used in logs.

        Returns:
            The request, or ``None`` if it has no full chunk.

        Raises:
            RuntimeError: If the client is not started.
        """
        self._require_started()

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import _build_token_ids

        token_ids = tuple(_build_token_ids(sequence_id, self._config.num_tokens))
        num_full_tokens = (len(token_ids) // self._chunk_size) * self._chunk_size
        if num_full_tokens == 0:
            self._log(
                "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
                % (sequence_id, request_kind, len(token_ids), self._chunk_size)
            )
            return None

        num_blocks = num_full_tokens // self._block_size
        usable_blocks = max(self._num_blocks - num_blocks, 1)
        return RequestContext(
            sequence_id=sequence_id,
            request_id=request_id,
            request_kind=request_kind,
            token_ids=token_ids,
            num_full_tokens=num_full_tokens,
            total_chunks=num_full_tokens // self._chunk_size,
            chunk_size=self._chunk_size,
            block_offset=(sequence_id * num_blocks) % usable_blocks,
            num_blocks=num_blocks,
        )

    def lookup(self, request: RequestContext) -> LookupResult:
        """Run LOOKUP and wait for its hit count.

        Args:
            request: Request to look up.

        Returns:
            LOOKUP result and latency.

        Raises:
            RuntimeError: If the client is not started.
        """
        req_client = self._require_started()

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _make_key,
            _poll_prefetch_status,
            _send_lookup,
        )

        lookup_key = _make_key(
            request.token_ids,
            request.request_id,
            start=0,
            end=request.num_full_tokens,
            world_size=self._kv_world_size,
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
        """STORE a token range on every write-enabled Worker.

        Args:
            request: Request to store.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Returns:
            Worker results, or ``None`` for an empty range.

        Raises:
            RuntimeError: If the client is not started.
            ValueError: If a non-empty range is invalid or not chunk-aligned.
        """
        return self._transfer("store", request, start_token, token_count)

    def retrieve(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """RETRIEVE a token range on every read-enabled Worker.

        Args:
            request: Request to retrieve.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Returns:
            Worker results, or ``None`` for an empty range.

        Raises:
            RuntimeError: If the client is not started.
            ValueError: If a non-empty range is invalid or not chunk-aligned.
        """
        return self._transfer("retrieve", request, start_token, token_count)

    def zero_destination(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> None:
        """Clear an engine-driven destination range before RETRIEVE.

        This is a no-op in handle mode.

        Args:
            request: Request to clear.
            start_token: Start offset in the request.
            token_count: Number of tokens.

        Raises:
            RuntimeError: If the client is not started.
            ValueError: If a non-empty data-mode range is invalid.
        """
        self._require_started()
        if token_count == 0 or self._config.uses_handle_transfer:
            return
        self._validate_token_range(request, start_token, token_count)

        block_offset = request.block_offset + (start_token // self._block_size)
        num_blocks = token_count // self._block_size
        for worker in self._workers:
            if worker.spec.retrieve_enabled:
                worker.buffers.zero_blocks(block_offset, num_blocks)

    def compute_checksums(
        self,
        request: RequestContext,
        start_token: int,
        token_count: int,
        *,
        operation: Literal["store", "retrieve"],
    ) -> dict[int, list[str]] | None:
        """Compute per-receiver checksums for a request range.

        Data mode hashes worker buffers; handle mode queries the Server.
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
        self._require_started()
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _query_checksum,
        )

        block_offset = request.block_offset + (start_token // self._block_size)
        num_blocks = token_count // self._block_size
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
            if self._config.uses_handle_transfer:
                if not self._config.http_url:
                    return None
                parts = _query_checksum(
                    self._config.http_url.rstrip("/"),
                    block_offset,
                    num_blocks,
                    self._block_size,
                    self._chunk_size,
                    instance_id=worker.spec.instance_id,
                )
            else:
                parts = worker.buffers.checksums(
                    block_offset, num_blocks, self._chunk_size
                )
            if not parts:
                return None
            checksums[worker.spec.rank] = parts

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

    def end_session(self, request: RequestContext) -> None:
        """End the request's Server-side session."""
        req_client = self._require_started()

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import _send_end_session

        _send_end_session(req_client, request.request_id)

    def close(self) -> None:
        """Close transfer contexts before releasing their worker-owned buffers.

        Cleanup is idempotent, including after partial startup.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _DEFAULT_RPC_TIMEOUT_S,
        )

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
        self._started = False

    def _transfer(
        self,
        operation: Literal["store", "retrieve"],
        request: RequestContext,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """Run either public transfer operation with shared validation and waiting."""
        self._require_started()
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _TIMEOUT,
            _make_key,
            _wait_for_result,
        )

        attempted: list[int] = []
        successful: list[int] = []
        failed: list[int] = []
        is_store = operation == "store"
        success_status = "stored" if is_store else "retrieved"
        status = success_status
        started_at = time.monotonic()
        block_offset = request.block_offset + (start_token // self._block_size)
        num_blocks = token_count // self._block_size
        block_ids = list(range(block_offset, block_offset + num_blocks))
        for worker in self._workers:
            enabled = (
                worker.spec.store_enabled if is_store else worker.spec.retrieve_enabled
            )
            if not enabled:
                continue
            attempted.append(worker.spec.rank)
            key = _make_key(
                request.token_ids,
                request.request_id,
                start=start_token,
                end=start_token + token_count,
                worker_id=worker.spec.kv_worker_id,
                world_size=self._kv_world_size,
            )
            block_ids_per_group = [
                block_ids.copy() for _ in range(self._num_engine_group_infos)
            ]
            event = worker.transfer_context.create_recorded_event()
            submit = (
                worker.transfer_context.submit_store
                if is_store
                else worker.transfer_context.submit_retrieve
            )
            future = submit(
                request.request_id,
                key,
                worker.buffers.kv_caches,
                block_ids_per_group,
                event,
                self._blocks_in_chunk,
            )
            reply = _wait_for_result(
                future, retain_refs=(event,) if event is not None else ()
            )
            if reply is _TIMEOUT:
                worker_status = "timeout"
            else:
                worker_status = success_status if reply else f"{operation}_failed"
            if worker_status == success_status:
                successful.append(worker.spec.rank)
            else:
                failed.append(worker.spec.rank)
                status = worker_status

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
                "writers" if is_store else "workers",
            )
        )
        return result

    def _initialize(self) -> None:
        """Allocate resources and register Workers."""

        # Keep heavy imports out of the thin CLI package path.
        # Third Party
        import zmq

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _DEFAULT_RPC_TIMEOUT_S,
            _INSTANCE_ID_BASE,
            _MODEL_NAME,
            DTYPE_MAP,
            _get_chunk_size,
        )
        from lmcache.utils import EngineType
        from lmcache.v1.gpu_connector.utils import LayoutHints
        from lmcache.v1.kv_layer_groups import (
            format_kvcache_shape_spec,
            parse_kvcache_shape_spec,
        )
        from lmcache.v1.multiprocess.transfer_context import (
            EngineDrivenTransferContext,
            create_transfer_context,
        )
        from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

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

        layer_groups = parse_kvcache_shape_spec(config.kvcache_shape_spec)
        self._num_engine_group_infos = len(layer_groups) or 1
        self._log(
            "Resolved KV shape spec: %s" % format_kvcache_shape_spec(layer_groups)
        )

        first = layer_groups[0]
        nb_vals = {group.shape_desc.nb for group in layer_groups}
        bs_vals = {group.shape_desc.bs for group in layer_groups}
        if len(nb_vals) > 1 or len(bs_vals) > 1:
            raise ValueError(
                "All groups must share NB and BS (paged KV requires uniform "
                "block geometry). Got NB=%s BS=%s" % (sorted(nb_vals), sorted(bs_vals))
            )

        num_layers = sum(group.num_layers for group in layer_groups)
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

        heads_set = {group.shape_desc.nh for group in layer_groups}
        hs_set = {group.shape_desc.hs for group in layer_groups}
        kv_size_set = {group.shape_desc.kv_size for group in layer_groups}
        dtype_set = {group.dtype for group in layer_groups}
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
            "Each request: %d tokens (%d full chunks)"
            % (
                config.num_tokens + 1,
                (config.num_tokens + 1) // self._chunk_size,
            )
        )
        self._log(
            "KV shape: %d layers, %s heads x %s, dtype=%s, "
            "blocks=%dx%d, kv=%s"
            % (
                num_layers,
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
                groups=layer_groups,
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
                and len(layer_groups) > 1
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
        return self._req_client

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

# SPDX-License-Identifier: Apache-2.0
"""Runtime ownership and operation results for ``lmcache bench server``."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import mmap
import os

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.bench.server_bench.config import (
    BenchConfig,
    WorkerSpec,
)

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.mq import MessageQueueClient


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
class BenchRequest:
    """Resolved token and block geometry for one data-plane request.

    ``label`` is used only in progress output. Runtime operations never branch
    on it, so callers may use any descriptive value.
    """

    seq_no: int
    request_id: str
    label: str
    token_ids: tuple[int, ...]
    num_full_tokens: int
    total_chunks: int
    chunk_size: int
    block_offset: int
    num_blocks: int


@dataclass
class WorkerRuntime:
    """Live resources owned by one simulated Worker.

    Attributes:
        spec: Logical Worker identity and STORE/RETRIEVE routing roles.
        kv_tensors: Paged KV tensors retained for the Worker lifetime.
        ipc_wrappers: IPC descriptions retained with their source tensors.
        server_pool: Mapping of the Server-owned SHM pool for that Worker.
        shm_mappings: Client-owned POSIX SHM mappings as ``(address, size)``.
    """

    spec: WorkerSpec
    kv_tensors: "list[torch.Tensor]"
    ipc_wrappers: list[Any]
    server_pool: "mmap.mmap | None" = None
    shm_mappings: list[tuple[int, int]] = field(default_factory=list)


class BenchRuntime:
    """Own the live client, Workers, and resources for one benchmark run.

    Construction has no external side effects. Call :meth:`start` before
    running operations and :meth:`close` when the run ends. The runtime keeps
    the CLI independent from ZMQ, tensor allocation, Worker registration, IPC,
    SHM, and multi-Worker dispatch details.
    """

    def __init__(
        self,
        config: BenchConfig,
        log: Callable[[str], None],
    ) -> None:
        """Create a stopped runtime.

        Args:
            config: Resolved configuration for one server-bench run.
            log: Progress logger that already applies the CLI quiet policy.
        """
        self._config = config
        self._log = log
        self._zmq_context: Any | None = None
        self._client: "MessageQueueClient | None" = None
        self._workers: list[WorkerRuntime] = []
        self._registered_instance_ids: list[int] = []
        self._shm_names: list[str] = []
        self._started = False

        self._chunk_size = 0
        self._block_size = 0
        self._num_blocks = 0
        self._num_engine_group_infos = 0
        self._kv_world_size = 0

    def start(self) -> None:
        """Connect to the Server, allocate KV resources, and register Workers.

        Raises:
            ValueError: If the KV layout is invalid or incompatible with the
                selected MLA routing.
            RuntimeError: If GPU mode is unavailable, registration fails, or
                this runtime has already been started.
        """
        if self._started:
            raise RuntimeError("BenchRuntime has already been started")

        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

        self._started = True

    def create_request(
        self,
        seq_no: int,
        request_id: str,
        label: str,
    ) -> BenchRequest | None:
        """Resolve tokens and block locations for one request.

        Args:
            seq_no: Synthetic sequence number used to generate tokens and
                choose Device blocks.
            request_id: Server session identifier.
            label: Descriptive text used only in progress output.

        Returns:
            A resolved request, or ``None`` when it contains no full Server
            chunk.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
        """
        self._require_started()

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import _build_token_ids

        token_ids = tuple(_build_token_ids(seq_no, self._config.num_tokens))
        num_full_tokens = (len(token_ids) // self._chunk_size) * self._chunk_size
        if num_full_tokens == 0:
            self._log(
                "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
                % (seq_no, label, len(token_ids), self._chunk_size)
            )
            return None

        num_blocks = num_full_tokens // self._block_size
        usable_blocks = max(self._num_blocks - num_blocks, 1)
        return BenchRequest(
            seq_no=seq_no,
            request_id=request_id,
            label=label,
            token_ids=token_ids,
            num_full_tokens=num_full_tokens,
            total_chunks=num_full_tokens // self._chunk_size,
            chunk_size=self._chunk_size,
            block_offset=(seq_no * num_blocks) % usable_blocks,
            num_blocks=num_blocks,
        )

    def lookup(self, request: BenchRequest) -> LookupResult:
        """Run one scheduler-scoped LOOKUP and wait for its hit count.

        Args:
            request: Resolved request returned by :meth:`create_request`.

        Returns:
            The hit count, total chunk count, latency, and timeout status.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
        """
        client = self._require_started()

        # Standard
        import time

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
        if not _send_lookup(client, lookup_key, tp_size=len(self._workers)):
            latency_ms = (time.monotonic() - started_at) * 1000
            self._log("  [seq %d/%s] LOOKUP timeout" % (request.seq_no, request.label))
            return LookupResult(
                hit_chunks=0,
                total_chunks=request.total_chunks,
                latency_ms=latency_ms,
                error="timeout",
            )

        hit_chunks = _poll_prefetch_status(client, lookup_key.request_id)
        if hit_chunks is None:
            hit_chunks = 0
        latency_ms = (time.monotonic() - started_at) * 1000
        self._log(
            "  [seq %d/%s] LOOKUP: %d/%d chunks hit (%.1f ms)"
            % (
                request.seq_no,
                request.label,
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
        request: BenchRequest,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """STORE a token range through every write-enabled Worker.

        Args:
            request: Resolved request returned by :meth:`create_request`.
            start_token: Token offset within the request.
            token_count: Number of tokens to store.

        Returns:
            Aggregate per-Worker transfer status, or ``None`` for an empty
            range.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
            ValueError: If the token range is outside the request or is not
                chunk-aligned.
        """
        client = self._require_started()
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        # Standard
        import time

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _make_key,
            _send_store,
        )

        attempted: list[int] = []
        successful: list[int] = []
        failed: list[int] = []
        status = "stored"
        started_at = time.monotonic()
        block_offset = request.block_offset + (start_token // self._block_size)
        for worker in self._workers:
            if not worker.spec.store_enabled:
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
            worker_status = _send_store(
                client,
                key,
                block_offset=block_offset,
                block_size=self._block_size,
                num_engine_group_infos=self._num_engine_group_infos,
                use_gpu=self._config.is_gpu,
                use_handle=self._config.uses_handle_transfer,
                client_tensors=self._data_tensors(worker),
                chunk_size=self._chunk_size,
                server_pool=worker.server_pool,
                instance_id=worker.spec.instance_id,
            )
            if worker_status == "stored":
                successful.append(worker.spec.rank)
            else:
                failed.append(worker.spec.rank)
                status = worker_status

        latency_ms = (time.monotonic() - started_at) * 1000
        result = TransferResult(
            operation="store",
            token_count=token_count,
            latency_ms=latency_ms,
            attempted_worker_ranks=tuple(attempted),
            successful_worker_ranks=tuple(successful),
            failed_worker_ranks=tuple(failed),
        )
        self._log(
            "  [seq %d/%s] STORE: %s (%d tokens, %.1f ms, %d writers)"
            % (
                request.seq_no,
                request.label,
                status,
                token_count,
                latency_ms,
                len(attempted),
            )
        )
        return result

    def retrieve(
        self,
        request: BenchRequest,
        start_token: int,
        token_count: int,
    ) -> TransferResult | None:
        """RETRIEVE a token range through every read-enabled Worker.

        Args:
            request: Resolved request returned by :meth:`create_request`.
            start_token: Token offset within the request.
            token_count: Number of tokens to retrieve.

        Returns:
            Aggregate per-Worker transfer status, or ``None`` for an empty
            range.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
            ValueError: If the token range is outside the request or is not
                chunk-aligned.
        """
        client = self._require_started()
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        # Standard
        import time

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _make_key,
            _send_retrieve,
        )

        attempted: list[int] = []
        successful: list[int] = []
        failed: list[int] = []
        status = "retrieved"
        started_at = time.monotonic()
        block_offset = request.block_offset + (start_token // self._block_size)
        hit_chunks = token_count // self._chunk_size
        for worker in self._workers:
            if not worker.spec.retrieve_enabled:
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
            worker_status = _send_retrieve(
                client,
                key,
                self._chunk_size,
                hit_chunks,
                block_offset=block_offset,
                block_size=self._block_size,
                num_engine_group_infos=self._num_engine_group_infos,
                use_gpu=self._config.is_gpu,
                use_handle=self._config.uses_handle_transfer,
                client_tensors=self._data_tensors(worker),
                server_pool=worker.server_pool,
                instance_id=worker.spec.instance_id,
            )
            if worker_status == "retrieved":
                successful.append(worker.spec.rank)
            else:
                failed.append(worker.spec.rank)
                status = worker_status

        latency_ms = (time.monotonic() - started_at) * 1000
        result = TransferResult(
            operation="retrieve",
            token_count=token_count,
            latency_ms=latency_ms,
            attempted_worker_ranks=tuple(attempted),
            successful_worker_ranks=tuple(successful),
            failed_worker_ranks=tuple(failed),
        )
        self._log(
            "  [seq %d/%s] RETRIEVE: %s (%d tokens, %.1f ms, %d workers)"
            % (
                request.seq_no,
                request.label,
                status,
                token_count,
                latency_ms,
                len(attempted),
            )
        )
        return result

    def zero_destination(
        self,
        request: BenchRequest,
        start_token: int,
        token_count: int,
    ) -> None:
        """Clear an engine-driven destination range before RETRIEVE.

        Handle mode intentionally remains a no-op to preserve the current
        baseline behavior.

        Args:
            request: Resolved request returned by :meth:`create_request`.
            start_token: Token offset within the request.
            token_count: Number of tokens to clear.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
            ValueError: If the token range is outside the request or is not
                chunk-aligned.
        """
        self._require_started()
        if token_count == 0 or self._config.uses_handle_transfer:
            return
        self._validate_token_range(request, start_token, token_count)

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _zero_fill_client_blocks,
        )

        block_offset = request.block_offset + (start_token // self._block_size)
        num_blocks = token_count // self._block_size
        for worker in self._workers:
            if worker.spec.retrieve_enabled:
                _zero_fill_client_blocks(
                    worker.kv_tensors,
                    block_offset,
                    num_blocks,
                )

    def compute_checksums(
        self,
        request: BenchRequest,
        start_token: int,
        token_count: int,
    ) -> list[str] | None:
        """Compute checksums for a request range without exposing hardware.

        Engine-driven mode hashes write-enabled Worker tensors locally.
        Handle mode queries the Server checksum endpoint for the registered
        Device slots.

        Args:
            request: Resolved request returned by :meth:`create_request`.
            start_token: Token offset within the request.
            token_count: Number of tokens to hash.

        Returns:
            One digest per Server chunk, or ``None`` if checksums are not
            available.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
            ValueError: If the token range is outside the request or is not
                chunk-aligned.
        """
        self._require_started()
        if token_count == 0:
            return None
        self._validate_token_range(request, start_token, token_count)

        # Standard
        import hashlib

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _compute_client_checksums,
            _query_checksum,
        )

        block_offset = request.block_offset + (start_token // self._block_size)
        num_blocks = token_count // self._block_size
        checksums: list[str] | None
        if self._config.uses_handle_transfer:
            if not self._config.http_url or not self._workers:
                return None
            checksums = _query_checksum(
                self._config.http_url.rstrip("/"),
                block_offset,
                num_blocks,
                self._block_size,
                self._chunk_size,
                instance_id=self._workers[0].spec.instance_id,
            )
        else:
            parts: list[str] = []
            for worker in self._workers:
                if not worker.spec.store_enabled:
                    continue
                parts.extend(
                    _compute_client_checksums(
                        worker.kv_tensors,
                        block_offset,
                        num_blocks,
                        self._block_size,
                        self._chunk_size,
                    )
                )
            checksums = parts or None

        if checksums:
            digest = hashlib.md5("".join(checksums).encode()).hexdigest()[:16]
            self._log(
                "  [seq %d/%s] CHECKSUM: %s (%d chunks)"
                % (request.seq_no, request.label, digest, len(checksums))
            )
        return checksums

    def end_session(self, request: BenchRequest) -> None:
        """Release Server-side state associated with one request session.

        Args:
            request: Resolved request returned by :meth:`create_request`.

        Raises:
            RuntimeError: If :meth:`start` has not completed.
        """
        client = self._require_started()

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import _send_end_session

        _send_end_session(client, request.request_id)

    def close(self) -> None:
        """Unregister Workers and release all local runtime resources.

        The method is idempotent and can clean up a partially completed
        :meth:`start`, making it safe to call from a ``finally`` block.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _send_unregister_kv_cache,
        )

        if self._client is not None:
            for instance_id in self._registered_instance_ids:
                try:
                    ok = _send_unregister_kv_cache(
                        self._client,
                        instance_id=instance_id,
                        use_handle=self._config.uses_handle_transfer,
                    )
                    self._log(
                        "[iid %d] UNREGISTER_KV_CACHE: %s"
                        % (instance_id, "OK" if ok else "FAIL")
                    )
                except Exception as exc:
                    self._log(
                        "  [warning] UNREGISTER_KV_CACHE failed for "
                        "iid %d: %s" % (instance_id, exc)
                    )

        for worker in self._workers:
            if worker.server_pool is None:
                continue
            try:
                worker.server_pool.close()
            except (BufferError, ValueError):
                pass

        # Drop tensor/buffer owners before unmapping their client-side SHM.
        for worker in self._workers:
            worker.kv_tensors.clear()
            worker.ipc_wrappers.clear()
        if any(worker.shm_mappings for worker in self._workers):
            # First Party
            from lmcache.v1.platform.cpu.shm import shm_munmap

            for worker in self._workers:
                for address, size in worker.shm_mappings:
                    try:
                        shm_munmap(address, size)
                    except OSError:
                        pass

        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:
                self._log("  [warning] client close failed: %s" % exc)
            finally:
                self._client = None
        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception as exc:
                self._log("  [warning] ZMQ context close failed: %s" % exc)
            finally:
                self._zmq_context = None

        if self._shm_names:
            # First Party
            from lmcache.v1.platform.cpu.shm import shm_unlink

            for name in self._shm_names:
                try:
                    shm_unlink(name)
                except OSError:
                    pass

        self._registered_instance_ids.clear()
        self._workers.clear()
        self._shm_names.clear()
        self._started = False

    def _initialize(self) -> None:
        """Allocate and register every resource required by the runtime."""

        # Heavy imports stay inside start so the slim lmcache-cli package can
        # still import the command parser and show its install hint.
        # Third Party
        import zmq

        # First Party
        from lmcache.cli.commands.bench.server_bench.helpers import (
            _INSTANCE_ID_BASE,
            DTYPE_MAP,
            _allocate_cpu_shm_kv_cache,
            _allocate_gpu_kv_cache,
            _get_chunk_size,
            _is_mla_kv_size,
            _send_register_kv_cache,
            shm_open_pool_as_mmap,
        )
        from lmcache.v1.kv_layer_groups import (
            format_kvcache_shape_spec,
            parse_kvcache_shape_spec,
        )
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo
        from lmcache.v1.multiprocess.mq import MessageQueueClient

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
        self._client = MessageQueueClient(config.rpc_url, self._zmq_context)

        self._chunk_size = _get_chunk_size(self._client)
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

        layout_hints = {
            "num_layers": num_layers,
            "num_heads": num_heads_display,
            "head_size": head_size_display,
            "num_blocks": self._num_blocks,
            "block_size": self._block_size,
            "dtype": dtype_string,
            "kv_size": kv_size_display,
        }
        engine_group_infos = [
            EngineGroupInfo(
                engine_group_id=group_index,
                layer_indices=tuple(group.layer_indices),
                tokens_per_block=self._block_size,
            )
            for group_index, group in enumerate(layer_groups)
        ]

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
            isinstance(kv_size_display, int) and _is_mla_kv_size(kv_size_display)
        )
        if (
            use_mla
            and isinstance(kv_size_display, int)
            and not _is_mla_kv_size(kv_size_display)
        ):
            raise ValueError(
                "--use-mla requires --kvcache-shape-spec with kv_size=1 "
                "(single-plane MLA group), got kv_size=%s" % kv_size_display
            )
        self._kv_world_size = 1 if use_mla else config.tp_size

        if use_gpu:
            # First Party
            from lmcache.v1.platform.kv_wrap import wrap_one_kv_cache
        else:
            # First Party
            from lmcache.v1.platform.cpu.shm import CpuShmTensorWrapper

        for rank in range(config.tp_size):
            instance_id = _INSTANCE_ID_BASE + rank
            kv_worker_id = 0 if use_mla else rank
            if use_gpu:
                allocated = _allocate_gpu_kv_cache(groups=layer_groups)
                self._log(
                    "[rank %d] Allocated %d GPU tensors on %s"
                    % (rank, len(allocated), allocated[0].device)
                )
                kv_wrappers = [wrap_one_kv_cache(tensor) for tensor in allocated]
                client_kv_tensors = allocated
                rank_shm_mappings: list[tuple[int, int]] = []
            else:
                shm_prefix = CpuShmTensorWrapper.SHM_NAME_PREFIX + "%s_r%d" % (
                    os.getpid(),
                    rank,
                )
                (
                    cpu_tensors,
                    cpu_wrappers,
                    rank_shm_names,
                    rank_shm_mappings,
                ) = _allocate_cpu_shm_kv_cache(
                    groups=layer_groups,
                    shm_prefix=shm_prefix,
                )
                self._shm_names.extend(rank_shm_names)
                self._log(
                    "[rank %d] Allocated %d CPU SHM tensors (prefix=%s)"
                    % (rank, len(cpu_tensors), shm_prefix)
                )
                kv_wrappers = list(cpu_wrappers)
                client_kv_tensors = cpu_tensors

            worker = WorkerRuntime(
                spec=WorkerSpec(
                    rank=rank,
                    kv_worker_id=kv_worker_id,
                    kv_world_size=self._kv_world_size,
                    instance_id=instance_id,
                    store_enabled=(rank == 0) if use_mla else True,
                    retrieve_enabled=True,
                ),
                kv_tensors=client_kv_tensors,
                ipc_wrappers=kv_wrappers,
                shm_mappings=rank_shm_mappings,
            )
            self._workers.append(worker)

            register_result = _send_register_kv_cache(
                self._client,
                instance_id=instance_id,
                world_size=self._kv_world_size,
                layout_hints=layout_hints,
                kv_caches=kv_wrappers if use_handle else None,
                use_gpu=use_gpu,
                use_handle=use_handle,
                engine_group_infos=engine_group_infos,
            )
            self._log(
                "[rank %d] REGISTER_KV_CACHE: %s"
                % (rank, "OK" if register_result else "FAIL")
            )
            if not register_result:
                raise RuntimeError(
                    "REGISTER_KV_CACHE failed for rank %d (instance_id=%d)"
                    % (rank, instance_id)
                )
            self._registered_instance_ids.append(instance_id)

            rank_server_pool: mmap.mmap | None = None
            if not use_handle and not isinstance(register_result, bool):
                shm_name = getattr(register_result, "shm_name", "")
                pool_size = getattr(register_result, "pool_size", 0)
                if shm_name and pool_size > 0:
                    rank_server_pool = shm_open_pool_as_mmap(shm_name, pool_size)
            worker.server_pool = rank_server_pool

        self._log("")

    def _require_started(self) -> "MessageQueueClient":
        """Return the live client or raise before successful startup."""
        if not self._started or self._client is None:
            raise RuntimeError("BenchRuntime must be started before use")
        return self._client

    def _validate_token_range(
        self,
        request: BenchRequest,
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

    def _data_tensors(self, worker: WorkerRuntime) -> "list[torch.Tensor] | None":
        """Return tensors only for the engine-driven transfer path."""
        if self._config.uses_handle_transfer:
            return None
        return worker.kv_tensors

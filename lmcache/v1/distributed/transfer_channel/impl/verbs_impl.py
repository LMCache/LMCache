# SPDX-License-Identifier: Apache-2.0
"""Native libibverbs transfer channel for peer host-DRAM L1 caches."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable
import importlib
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
    TransferChannelServer,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelReadResult,
)
from lmcache.v1.distributed.transfer_channel.factory import (
    register_transfer_channel_factory,
)

_MAX_CHUNK_BYTES = 1 << 30
_MAX_GID_INDEX = 255
_MAX_QUEUE_DEPTH = (1 << 32) - 1
_MAX_HANDSHAKE_TIMEOUT_MS = (1 << 31) - 1
logger = init_logger(__name__)


def _load_native() -> Any:
    try:
        return importlib.import_module("lmcache.rdma_l1_ops")
    except ImportError as error:
        raise RuntimeError(
            "The native RDMA L1 extension is unavailable. Build LMCache with "
            "BUILD_WITH_RDMA_L1=1."
        ) from error


def _parse_devices(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("RDMA device must be a non-empty string")
    devices = [device.strip() for device in value.split(",")]
    if any(not device for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("RDMA devices must be a comma-separated list of unique names")
    return devices


def _parse_gid_indices(value: Any, fallback: int, rail_count: int) -> list[int]:
    if isinstance(fallback, bool) or not isinstance(fallback, int):
        raise ValueError("RDMA GID index must be an integer")
    if value is None or value == "":
        indices = [fallback] * rail_count
    elif isinstance(value, str):
        try:
            indices = [int(index.strip()) for index in value.split(",")]
        except ValueError as error:
            raise ValueError(
                "RDMA GID indices must be comma-separated integers"
            ) from error
    elif isinstance(value, (list, tuple)) and all(
        isinstance(index, int) and not isinstance(index, bool) for index in value
    ):
        indices = list(value)
    else:
        raise ValueError("RDMA GID indices must be a comma-separated string or list")
    if len(indices) != rail_count:
        raise ValueError("RDMA GID index count must match RDMA device count")
    if any(index < -1 or index > _MAX_GID_INDEX for index in indices):
        raise ValueError("RDMA GID indices must be in [-1, 255]")
    return indices


def _offset_url_port(url: str, offset: int) -> str:
    scheme = ""
    address = url
    if "://" in url:
        scheme, address = url.split("://", 1)
    host, separator, raw_port = address.rpartition(":")
    if not separator or not host:
        raise ValueError(f"Invalid RDMA URL: {url!r}")
    try:
        port = int(raw_port) + offset
    except ValueError as error:
        raise ValueError(f"Invalid RDMA URL: {url!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"RDMA URL port is outside [1, 65535]: {port}")
    prefix = f"{scheme}://" if scheme else ""
    return f"{prefix}{host}:{port}"


def _close_all(resources: list[Any]) -> Exception | None:
    first_error: Exception | None = None
    for resource in reversed(resources):
        try:
            resource.close()
        except Exception as error:  # noqa: BLE001 - preserve first teardown error
            if first_error is None:
                first_error = error
    return first_error


@dataclass
class _NativeTask:
    rail_index: int
    task_id: int
    chunk_count: int
    finished: bool = False
    succeeded: bool = False


@dataclass
class _ReadTask:
    object_rails: list[set[int]]
    native_tasks: list[_NativeTask]


class VerbsTransferChannelServer(TransferChannelServer):
    """The native contexts own the listeners; this exposes their identity."""

    def __init__(
        self,
        listen_url: str,
        advertise_url: str,
        l1_memory_desc: L1MemoryDesc,
    ) -> None:
        self.listen_url = listen_url
        self.advertise_url = advertise_url
        self.l1_memory_desc = l1_memory_desc

    def close(self) -> None:
        return


class VerbsTransferChannelClient(TransferChannelClient):
    """One logical client striped across one RC connection per HCA."""

    def __init__(
        self,
        native_clients: list[Any],
        reconnect: Callable[[], list[Any]],
        queue_depth: int = _MAX_QUEUE_DEPTH,
    ) -> None:
        if not native_clients:
            raise ValueError("RDMA client requires at least one rail")
        self._natives = native_clients
        self._reconnect = reconnect
        self._queue_depth = queue_depth
        self._outstanding_chunks = [0] * len(native_clients)
        self._tasks: dict[int, _ReadTask] = {}
        self._next_task_id = 1
        self._needs_reconnect = False
        self._teardown_pending = False
        self._closed = False
        self._lock = threading.Lock()

    @staticmethod
    def _chunk_count(size: int, rail_count: int) -> int:
        minimum = (size + _MAX_CHUNK_BYTES - 1) // _MAX_CHUNK_BYTES
        balanced = ((minimum + rail_count - 1) // rail_count) * rail_count
        return min(size, max(rail_count, balanced))

    def _stripe(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> tuple[list[tuple[list[int], list[int], list[int]]], list[set[int]]]:
        if not local_addresses:
            raise ValueError("RDMA read batch must not be empty")
        if len(local_addresses) != len(remote_addresses):
            raise ValueError("Local and remote RDMA address counts must match")

        rail_count = len(self._natives)
        local_offsets: list[list[int]] = [[] for _ in range(rail_count)]
        remote_offsets: list[list[int]] = [[] for _ in range(rail_count)]
        sizes: list[list[int]] = [[] for _ in range(rail_count)]
        object_rails: list[set[int]] = []
        for local, remote in zip(local_addresses, remote_addresses, strict=True):
            if not local.is_valid() or not remote.is_valid():
                raise ValueError("RDMA read addresses must be valid")
            if local.size != remote.size:
                raise ValueError("Local and remote RDMA object sizes must match")
            chunk_count = self._chunk_count(local.size, rail_count)
            chunk_size, remainder = divmod(local.size, chunk_count)
            consumed = 0
            used_rails: set[int] = set()
            for chunk_index in range(chunk_count):
                size = chunk_size + (chunk_index < remainder)
                rail_index = chunk_index % rail_count
                local_offsets[rail_index].append(local.offset + consumed)
                remote_offsets[rail_index].append(remote.offset + consumed)
                sizes[rail_index].append(size)
                used_rails.add(rail_index)
                consumed += size
            object_rails.append(used_rails)
        return (
            list(zip(local_offsets, remote_offsets, sizes, strict=True)),
            object_rails,
        )

    def _fail_all_tasks(self) -> None:
        for task in self._tasks.values():
            for native_task in task.native_tasks:
                native_task.finished = True
                native_task.succeeded = False
        self._outstanding_chunks = [0] * len(self._natives)

    def _retry_pending_teardown(self) -> Exception | None:
        close_error = _close_all(self._natives)
        if close_error is not None:
            return close_error
        self._fail_all_tasks()
        self._teardown_pending = False
        self._needs_reconnect = True
        return None

    def _recover_if_needed(self) -> None:
        if self._teardown_pending:
            close_error = self._retry_pending_teardown()
            if close_error is not None:
                raise close_error
        self._needs_reconnect |= any(not native.healthy for native in self._natives)
        if not self._needs_reconnect:
            return
        has_active_tasks = any(
            any(not native_task.finished for native_task in task.native_tasks)
            for task in self._tasks.values()
        )
        if has_active_tasks:
            raise RuntimeError("Cannot reconnect RDMA rails with reads in flight")
        close_error = _close_all(self._natives)
        if close_error is not None:
            raise close_error
        rail_count = len(self._natives)
        replacement = self._reconnect()
        if len(replacement) != rail_count:
            _close_all(replacement)
            raise RuntimeError("RDMA reconnect changed the rail count")
        self._natives = replacement
        self._outstanding_chunks = [0] * rail_count
        self._needs_reconnect = False

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit a batch of RDMA reads from remote into local L1 memory.

        Args:
            local_addresses: Local destination addresses in registered L1 memory.
            remote_addresses: Remote source addresses, one per local address with a
                matching size.

        Returns:
            A task ID to pass to ``query_read_status``.
            If a partial submit cannot be quiesced immediately, its task stays
            nonterminal until teardown succeeds, then completes as failed.

        Raises:
            RuntimeError: If the client is closed, the connection cannot be
                recovered, or an RDMA rail rejects the transfer state.
            ValueError: If the batch or address pairs are invalid or exceed a rail's
                configured limits.
            OverflowError: If native byte or task-ID accounting overflows.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("RDMA transfer client is closed")
            self._recover_if_needed()
            batches, object_rails = self._stripe(local_addresses, remote_addresses)
            for rail_index, (_, _, sizes) in enumerate(batches):
                if (
                    self._outstanding_chunks[rail_index] + len(sizes)
                    > self._queue_depth
                ):
                    raise RuntimeError("RDMA send queue is full")

            native_tasks: list[_NativeTask] = []
            task_id = self._next_task_id
            self._next_task_id += 1
            self._tasks[task_id] = _ReadTask(object_rails, native_tasks)
            try:
                for rail_index, (local_offsets, remote_offsets, sizes) in enumerate(
                    batches
                ):
                    if not sizes:
                        continue
                    native_task_id = self._natives[rail_index].submit_read(
                        local_offsets, remote_offsets, sizes
                    )
                    native_tasks.append(
                        _NativeTask(
                            rail_index,
                            int(native_task_id),
                            len(sizes),
                        )
                    )
                    self._outstanding_chunks[rail_index] += len(sizes)
            except Exception as error:
                close_error = _close_all(self._natives)
                self._needs_reconnect = True
                if close_error is not None:
                    self._teardown_pending = True
                    logger.error(
                        "Native RDMA submit failed (%s) and teardown remains "
                        "pending: %s",
                        error,
                        close_error,
                        exc_info=True,
                    )
                    return task_id
                self._fail_all_tasks()
                del self._tasks[task_id]
                raise

            return task_id

    def query_read_status(self, task_id: int) -> TransferChannelReadResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("RDMA transfer client is closed")
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"Unknown RDMA read task id: {task_id}")
            if self._teardown_pending:
                close_error = self._retry_pending_teardown()
                if close_error is not None:
                    return TransferChannelReadResult(finished=False)
            for native_task in task.native_tasks:
                if native_task.finished:
                    continue
                finished, succeeded, count = self._natives[
                    native_task.rail_index
                ].query_read_status(native_task.task_id)
                if finished:
                    native_task.finished = True
                    self._outstanding_chunks[native_task.rail_index] -= (
                        native_task.chunk_count
                    )
                    native_task.succeeded = bool(succeeded) and (
                        int(count) == native_task.chunk_count
                    )
                    self._needs_reconnect |= not native_task.succeeded
            if any(not native_task.finished for native_task in task.native_tasks):
                return TransferChannelReadResult(finished=False)

            rail_success = {
                native_task.rail_index: native_task.succeeded
                for native_task in task.native_tasks
            }
            succeeded_mask = [
                all(rail_success.get(rail, False) for rail in rails)
                for rails in task.object_rails
            ]
            del self._tasks[task_id]
            return TransferChannelReadResult(
                finished=True,
                succeeded_mask=succeeded_mask,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            error = _close_all(self._natives)
            if error is not None:
                self._needs_reconnect = True
                self._teardown_pending = True
                raise error
            self._tasks.clear()
            self._outstanding_chunks = [0] * len(self._natives)
            self._teardown_pending = False
            self._closed = True


class VerbsTransferChannelContext(TransferChannelContext):
    """Register the same host L1 region once per configured HCA."""

    def __init__(
        self,
        l1_memory_desc: L1MemoryDesc,
        listen_url: str,
        advertise_url: str,
        device_name: str,
        port_num: int = 1,
        gid_index: int = -1,
        gid_indices: Any = None,
        queue_depth: int = 4096,
        handshake_timeout_ms: int = 10_000,
    ) -> None:
        devices = _parse_devices(device_name)
        indices = _parse_gid_indices(gid_indices, gid_index, len(devices))
        if (
            isinstance(port_num, bool)
            or not isinstance(port_num, int)
            or not 1 <= port_num <= 255
        ):
            raise ValueError("RDMA port must be an integer in [1, 255]")
        if (
            isinstance(queue_depth, bool)
            or not isinstance(queue_depth, int)
            or not 1 <= queue_depth <= _MAX_QUEUE_DEPTH
        ):
            raise ValueError("RDMA queue depth must be an integer in [1, 2^32-1]")
        if (
            isinstance(handshake_timeout_ms, bool)
            or not isinstance(handshake_timeout_ms, int)
            or not 1 <= handshake_timeout_ms <= _MAX_HANDSHAKE_TIMEOUT_MS
        ):
            raise ValueError(
                "RDMA handshake timeout must be an integer in [1, INT_MAX] ms"
            )

        self._l1_memory_desc = l1_memory_desc
        self._queue_depth = queue_depth
        self._listen_urls = [
            _offset_url_port(listen_url, rail) for rail in range(len(devices))
        ]
        self._advertise_urls = [
            _offset_url_port(advertise_url, rail) for rail in range(len(devices))
        ]
        native = _load_native()
        contexts: list[Any] = []
        try:
            for rail, device in enumerate(devices):
                contexts.append(
                    native.RdmaContext(
                        base_address=l1_memory_desc.ptr,
                        length=l1_memory_desc.size,
                        listen_url=self._listen_urls[rail],
                        advertise_url=self._advertise_urls[rail],
                        device_name=device,
                        port_num=port_num,
                        gid_index=indices[rail],
                        queue_depth=queue_depth,
                        handshake_timeout_ms=handshake_timeout_ms,
                    )
                )
        except Exception:
            _close_all(contexts)
            raise
        self._natives = contexts
        self._clients: dict[str, VerbsTransferChannelClient] = {}
        self._server = VerbsTransferChannelServer(
            listen_url, advertise_url, l1_memory_desc
        )
        self._connect_lock = threading.Lock()
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False

    def get_transfer_channel_server(self) -> VerbsTransferChannelServer:
        return self._server

    def _connect(self, peer_url: str) -> list[Any]:
        clients: list[Any] = []
        try:
            for rail, native in enumerate(self._natives):
                clients.append(native.connect(_offset_url_port(peer_url, rail)))
        except Exception:
            _close_all(clients)
            raise
        return clients

    def _reconnect(self, peer_url: str) -> list[Any]:
        with self._connect_lock:
            with self._lock:
                if self._closed or self._closing:
                    raise RuntimeError("RDMA transfer context is closed")
            clients = self._connect(peer_url)
            with self._lock:
                context_closed = self._closed or self._closing
            if not context_closed:
                return clients
            close_error = _close_all(clients)
            if close_error is not None:
                raise close_error
            raise RuntimeError("RDMA transfer context is closed")

    def get_transfer_channel_client(
        self,
        peer_advertise_url: str,
    ) -> VerbsTransferChannelClient:
        with self._connect_lock:
            with self._lock:
                if self._closed or self._closing:
                    raise RuntimeError("RDMA transfer context is closed")
                client = self._clients.get(peer_advertise_url)
                if client is not None:
                    return client

            native_clients = self._connect(peer_advertise_url)
            try:
                client = VerbsTransferChannelClient(
                    native_clients,
                    reconnect=lambda: self._reconnect(peer_advertise_url),
                    queue_depth=self._queue_depth,
                )
            except Exception:
                _close_all(native_clients)
                raise

            with self._lock:
                if not self._closed and not self._closing:
                    self._clients[peer_advertise_url] = client
                    return client
            close_error = _close_all(native_clients)
            if close_error is not None:
                raise close_error
            raise RuntimeError("RDMA transfer context is closed")

    def remove_transfer_channel_client(self, peer_advertise_url: str) -> None:
        with self._lock:
            client = self._clients.get(peer_advertise_url)
        if client is None:
            return
        try:
            client.close()
        except Exception:  # noqa: BLE001 - retain for a later retry
            logger.exception(
                "Error closing native RDMA client for %s",
                peer_advertise_url,
            )
            return
        with self._lock:
            if self._clients.get(peer_advertise_url) is client:
                del self._clients[peer_advertise_url]

    def get_transfer_channel_address(
        self,
        lmcache_addresses: list[tuple[int, int]],
    ) -> list[TransferChannelAddress]:
        result: list[TransferChannelAddress] = []
        for offset, size in lmcache_addresses:
            if offset < 0 or size <= 0 or offset > self._l1_memory_desc.size - size:
                raise ValueError("RDMA address is outside the registered L1 region")
            result.append(TransferChannelAddress(offset, size))
        return result

    def get_num_connected_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closing = True
            # Closing native contexts first cancels accepted and outbound
            # handshakes, including a reconnect blocked outside ``_lock``.
            native_error = _close_all(self._natives)
            with self._lock:
                clients = list(self._clients.values())
            client_error = _close_all(clients)
            if client_error is not None or native_error is not None:
                # Keep rejecting new work, but leave teardown retryable.
                if native_error is not None:
                    raise native_error
                assert client_error is not None
                raise client_error
            with self._lock:
                self._clients.clear()
                self._closed = True
                self._closing = False


def create_verbs_transfer_channel_context(
    l1_memory_desc: L1MemoryDesc,
    listen_url: str,
    advertise_url: str,
    **kwargs: Any,
) -> VerbsTransferChannelContext:
    return VerbsTransferChannelContext(
        l1_memory_desc=l1_memory_desc,
        listen_url=listen_url,
        advertise_url=advertise_url,
        device_name=kwargs.get("device_name", ""),
        port_num=kwargs.get("port_num", 1),
        gid_index=kwargs.get("gid_index", -1),
        gid_indices=kwargs.get("gid_indices"),
        queue_depth=kwargs.get("queue_depth", 4096),
        handshake_timeout_ms=kwargs.get("handshake_timeout_ms", 10_000),
    )


register_transfer_channel_factory("verbs", create_verbs_transfer_channel_context)

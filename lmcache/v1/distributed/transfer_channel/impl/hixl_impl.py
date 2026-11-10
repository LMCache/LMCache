# SPDX-License-Identifier: Apache-2.0
"""HIXL Client and local Context lifecycle for registered Host L1 pools.

The Context remains abstract until peer management is implemented. Server,
Context transfer operations and factory wiring are not available yet. Importing
this module does not load the optional Ascend extension or initialize the NPU.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4
import ctypes
import json
import os
import re
import socket

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel.abstract import (
    TransferChannelClient,
    TransferChannelContext,
)
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)

if TYPE_CHECKING:
    # Third Party
    from lmcache_ascend.hixl_npu_comms import (
        Hixl,
        TransferOp,
        TransferOpDesc,
        TransferStatus,
    )

logger = init_logger(__name__)

_ENV_PREFIX = "LMCACHE_HIXL_"
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


def _parse_channel_url(url: str) -> tuple[str, int]:
    """Parse a handshake address without resolving its host.

    Args:
        url: host:port or tcp://host:port, with brackets for IPv6 literals.

    Returns:
        Host and positive port, preserving IPv6 brackets.

    Raises:
        ValueError: The URL or port is invalid.
    """
    match = re.fullmatch(r"(?:tcp://)?(\[[^\]]+\]|[^:/\s?#@]+):([0-9]+)", url)
    try:
        if match is None:
            raise ValueError("expected host:port or tcp://host:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid Channel address {url!r}: {exc}") from exc
    return host, number


def _validate_engine_id(engine_id: str) -> None:
    """Require a listening HIXL Engine identity for this bidirectional backend.

    Args:
        engine_id: IPv4:port or [IPv6]:port, without a URL scheme.

    Raises:
        ValueError: The IP literal or positive listening port is invalid.
    """
    match = re.fullmatch(r"(\[[^\]]+\]|[^:]+):([0-9]+)", engine_id)
    try:
        if match is None:
            raise ValueError("expected IP:port or [IPv6]:port")
        host, port = match.groups()
        if host.startswith("["):
            socket.inet_pton(socket.AF_INET6, host[1:-1])
        else:
            socket.inet_pton(socket.AF_INET, host)
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid HIXL engine_id {engine_id!r}: {exc}") from exc


def _read_int(suffix: str, default: str, minimum: int, maximum: int) -> int:
    """Read one bounded decimal integer from the HIXL environment.

    Args:
        suffix: Variable name after LMCACHE_HIXL_.
        default: String used when the variable is unset.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound.

    Returns:
        The parsed integer.

    Raises:
        ValueError: The value is not decimal or is outside the bounds.
    """
    name = _ENV_PREFIX + suffix
    value = os.environ.get(name, default)
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a decimal integer, got {value!r}")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value!r}"
        )
    return number


def _read_config() -> _HixlConfig:
    """Snapshot backend configuration before loading native dependencies.

    Returns:
        A validated configuration with an explicit listening engine ID.

    Raises:
        ValueError: ENGINE_ID is missing/invalid or an integer is invalid.
    """
    engine_id = os.environ.get(_ENV_PREFIX + "ENGINE_ID", "")
    _validate_engine_id(engine_id)
    return _HixlConfig(
        engine_id=engine_id,
        device_id=_read_int("DEVICE_ID", "0", 0, _INT32_MAX),
        connect_timeout_ms=_read_int("CONNECT_TIMEOUT_MS", "10000", 1, _INT32_MAX),
        comm_listen_port=(
            _read_int("COMM_LISTEN_PORT", "", 1, 65535)
            if _ENV_PREFIX + "COMM_LISTEN_PORT" in os.environ
            else None
        ),
    )


def _validate_memory_range(ptr: int, size: int) -> None:
    """Reject a Host pool that cannot be represented by native address types.

    Args:
        ptr: Raw base address of an already registered Host allocation.
        size: Registered byte length; no alignment or rounding is applied.

    Raises:
        ValueError: Either value is invalid or the range overflows uintptr_t.
    """
    if (
        type(ptr) is not int
        or type(size) is not int
        or not 0 < ptr <= _UINTPTR_MAX
        or not 0 < size <= _SIZE_MAX
        or size > _UINTPTR_MAX - ptr
    ):
        raise ValueError(f"Invalid Host memory range: ptr={ptr!r}, size={size!r}")


def _validate_address_range(
    address: TransferChannelAddress, buffer_size: int, *, label: str
) -> None:
    """Check one object lies within a previously validated Host pool.

    Args:
        address: Object offset and byte length relative to its pool.
        buffer_size: Validated registered pool size.
        label: Address position to include in error messages.

    Raises:
        ValueError: The offset or length is invalid or outside the pool.
    """
    offset, size = address.offset, address.size
    if type(offset) is not int or type(size) is not int:
        raise ValueError(f"{label}: offset and size must be integers, not bool")
    if offset < 0 or size <= 0:
        raise ValueError(
            f"{label}: require offset >= 0 and size > 0, got {offset}, {size}"
        )
    if offset > buffer_size or size > buffer_size - offset:
        raise ValueError(
            f"{label}: range offset={offset}, size={size} "
            f"exceeds buffer_size={buffer_size}"
        )


@dataclass(frozen=True)
class _HixlConfig:
    """Engine identity, logical device and connection settings for one Context.

    comm_listen_port optionally overrides the native communication-resource
    port; it is separate from the engine ID and the Channel handshake port.
    """

    engine_id: str
    device_id: int
    connect_timeout_ms: int
    comm_listen_port: int | None

    def initialize_options(self) -> dict[str, str]:
        """Return HIXL_CS/device-RoCE options with explicit connections.

        The registered L1 is used directly, without a HIXL staging pool.
        """
        comm: dict[str, str | int] = {
            "comm_resource_config.protocol_desc": "roce:device",
        }
        if self.comm_listen_port is not None:
            comm["comm_resource_config.listen_port"] = self.comm_listen_port
        return {
            "LocalCommRes": '{"version":"1.3"}',
            "BufferPool": "0:0",
            "AutoConnect": "0",
            "GlobalResourceConfig": json.dumps(comm),
        }


class HixlChannelMetadata(msgspec.Struct, frozen=True):
    """Registered Host range and peer identity; contains no native handles.

    Context admission validates the identity fields before creating a Client.
    The base and size describe the entire registered allocation in bytes.
    """

    advertise_url: str
    engine_id: str
    buffer_base_ptr: int
    buffer_size: int
    instance_id: str


@dataclass
class _HixlConnection:
    """Context-owned connection shared by Clients, retained after a fault.

    Metadata stays fixed for the record's lifetime. Context serializes changes
    to admission and cleanup state under its engine lock; Clients never revive
    or disconnect the record themselves.
    """

    metadata: HixlChannelMetadata
    connected: bool = True
    failure: str | None = None
    disconnect_succeeded: bool = False
    disconnect_error: str | None = None
    close_attempted: bool = False


if TYPE_CHECKING:

    class _HixlClientContext(Protocol):
        """Native-call boundary of the Context, not a runtime implementation.

        Calls hold the Context's engine lock and prepare a fresh ACL scope.
        They must not acquire a Client lock: the Client holds its own lock
        while calling these methods. Context shutdown requires quiesced users.
        """

        @property
        def metadata(self) -> HixlChannelMetadata:
            """Return the immutable local registered Host pool metadata."""
            ...

        def submit_transfer(
            self,
            connection: _HixlConnection,
            operation: TransferOp,
            descriptors: list[TransferOpDesc],
        ) -> int:
            """Validate the connection and submit under the engine lock/ACL scope.

            Return the native req; propagate preparation/submission errors
            unchanged. Do not reconnect or modify the Client's task table.
            """
            ...

        def query_transfer(
            self, connection: _HixlConnection, req: int
        ) -> TransferStatus:
            """Query under the engine lock/ACL scope, handling shared faults.

            Return FAILED without a native query for an already failed record.
            Otherwise validate ownership and query once. On a native query
            HixlError, mark the connection failed, attempt its disconnect and
            re-raise the original error. ACL scope preparation failures must
            not be reclassified as HixlError or invalidate the connection.
            """
            ...


class HixlTransferChannelClient(TransferChannelClient):
    """Borrow a peer connection and manage a shared READ/WRITE task ID space.

    Context owns registration, connections and engine lifetime. The caller
    retains both data regions until native accesses have ended; this Client
    neither owns the allocations nor grants permission to overwrite a peer.
    """

    def __init__(
        self, context: _HixlClientContext, connection: _HixlConnection
    ) -> None:
        """Create a Client for an admitted peer without connecting or allocating.

        Args:
            context: Owner providing the native-call boundary and local pool.
            connection: Context-owned record with validated, fixed peer identity.

        Raises:
            ValueError: A local or remote registered memory range is invalid.
            RuntimeError: The required optional Ascend binding cannot be loaded.
        """
        self._context = context
        self._connection = connection
        self._local_metadata = context.metadata
        self._peer_metadata = connection.metadata
        for metadata in (self._local_metadata, self._peer_metadata):
            _validate_memory_range(metadata.buffer_base_ptr, metadata.buffer_size)

        try:
            # Third Party
            from lmcache_ascend.hixl_npu_comms import (
                HixlError,
                TransferOp,
                TransferOpDesc,
                TransferStatus,
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HIXL requires compatible LMCache-Ascend native bindings with "
                "HixlError, TransferOp, TransferOpDesc and TransferStatus"
            ) from exc

        self._native_error = HixlError
        self._operation = TransferOp
        self._descriptor = TransferOpDesc
        self._status = TransferStatus
        self._lock = Lock()
        self._closed = False
        self._task_counter = 0
        # Client task_id -> (native req, object count for the result mask).
        self._tasks: dict[int, tuple[int, int]] = {}

    @property
    def closed(self) -> bool:
        """Return whether close has invalidated this Client's task IDs."""
        with self._lock:
            return self._closed

    def submit_read(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Read remote objects into local destinations; return a Client task ID.

        Args:
            local_addresses: Nonempty list of local destination offsets/sizes.
            remote_addresses: Matching remote source offsets/sizes.

        Returns:
            An ID in the shared READ/WRITE space, queried with query_status.

        Raises:
            ValueError: A range, pair length or batch size is invalid.
            RuntimeError: The Client/connection is unusable or req is null.
            HixlError: Native submission failed; no task ID is published.

        Note:
            Native errors do not prove no bytes were accessed. No retry occurs.
            Source and destination must remain valid until accesses have ended.
        """
        return self._submit_transfer(
            self._operation.READ, local_addresses, remote_addresses
        )

    def submit_write(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Write local objects into remote destinations; return a Client task ID.

        Args:
            local_addresses: Nonempty list of local source offsets/sizes.
            remote_addresses: Matching remote destination offsets/sizes. The
                caller must hold permission to overwrite these ranges.

        Returns:
            An ID in the shared READ/WRITE space, queried with query_status.

        Raises:
            ValueError: A range, pair length or batch size is invalid.
            RuntimeError: The Client/connection is unusable or req is null.
            HixlError: Native submission failed; no task ID is published.

        Note:
            This does not reserve or publish remote KV objects. Failure does not
            roll back writes. Retain source bytes unchanged and destinations
            exclusively writable until native accesses have ended.
        """
        return self._submit_transfer(
            self._operation.WRITE, local_addresses, remote_addresses
        )

    def query_status(self, task_id: int) -> TransferChannelTransferResult:
        """Poll a READ or WRITE once, consuming terminal requests at most once.

        Args:
            task_id: ID returned by this Client's submit_read or submit_write.

        Returns:
            Unfinished with an empty mask while waiting; otherwise a uniform
            success/failure mask. HIXL reports request-level status, not which
            individual objects were modified before a failure.

        Raises:
            KeyError: The ID is unknown, consumed, or invalidated by close.
            HixlError: Native query failed; the task is consumed and Context
                handles the shared connection fault before propagating it.
            RuntimeError: Context preparation failed (task retained), or an
                unsupported native terminal status was returned (task consumed).
        """
        with self._lock:
            req, count = self._tasks[task_id]
            try:
                status = self._context.query_transfer(self._connection, req)
            except self._native_error:
                # A native query error consumes the HIXL req. ACL preparation
                # errors propagate separately and leave this task available.
                del self._tasks[task_id]
                raise
            if status == self._status.WAITING:
                return TransferChannelTransferResult(False, [])

            # HIXL removes every non-WAITING req from its index, even for an
            # unsupported status. Never query that native handle again.
            del self._tasks[task_id]
            if status == self._status.COMPLETED:
                return TransferChannelTransferResult(True, [True] * count)
            if status == self._status.FAILED:
                return TransferChannelTransferResult(True, [False] * count)
            raise RuntimeError(f"Unsupported HIXL transfer status: {status!r}")

    def close(self) -> None:
        """Invalidate local tasks and reject new submissions; safe to repeat.

        This does not cancel or drain native accesses, free allocations, or
        disconnect a shared peer. Context owns connection/engine cleanup.
        A returned close is not permission to reuse or free in-flight buffers.
        """
        with self._lock:
            self._closed = True
            self._tasks.clear()

    def _build_transfer_descriptors(
        self,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> list[TransferOpDesc]:
        """Validate the whole batch and describe exact local/remote ranges.

        The operation selects direction; WRITE must not swap the address fields.
        Validated pools and contained subranges ensure native address arithmetic
        fits. Descriptors live through submission, without owning either pool.
        """
        if len(local_addresses) != len(remote_addresses):
            raise ValueError("Local and remote address count must match")
        if not local_addresses:
            raise ValueError("Transfer address batch must not be empty")
        if len(local_addresses) > _UINT32_MAX:
            raise ValueError("Transfer descriptor count exceeds uint32 representation")

        descriptors = []
        for index, (local, remote) in enumerate(
            zip(local_addresses, remote_addresses, strict=True)
        ):
            _validate_address_range(
                local,
                self._local_metadata.buffer_size,
                label=f"local_addresses[{index}]",
            )
            _validate_address_range(
                remote,
                self._peer_metadata.buffer_size,
                label=f"remote_addresses[{index}]",
            )
            if local.size != remote.size:
                raise ValueError(
                    f"pair[{index}]: local size {local.size} "
                    f"!= remote size {remote.size}"
                )
            descriptors.append(
                self._descriptor(
                    local_addr=self._local_metadata.buffer_base_ptr + local.offset,
                    remote_addr=self._peer_metadata.buffer_base_ptr + remote.offset,
                    len=local.size,
                )
            )
        return descriptors

    def _submit_transfer(
        self,
        operation: TransferOp,
        local_addresses: list[TransferChannelAddress],
        remote_addresses: list[TransferChannelAddress],
    ) -> int:
        """Submit after validation, then publish a Client-local task ID.

        The Client lock serializes submission, query and close. Context takes
        its engine lock inside the call. Registration after native submission
        is not transactional: a Python allocation failure does not cancel req.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("HIXL Client is closed")
            descriptors = self._build_transfer_descriptors(
                local_addresses, remote_addresses
            )
            count = len(descriptors)
            req = self._context.submit_transfer(
                self._connection, operation, descriptors
            )
            if req == 0:
                raise RuntimeError("HIXL TransferAsync returned a null request")
            task_id = self._task_counter
            self._task_counter += 1
            self._tasks[task_id] = (req, count)
            return task_id


class HixlTransferChannelContext(TransferChannelContext):
    """Own an Engine and Host registration; borrow the L1 and ACL context.

    This lifecycle foundation retains the inherited abstract peer/server
    methods and cannot yet be instantiated as a complete Channel Context.
    It does not allocate L1 memory, create an ACL context or own the runtime.
    The L1 owner must supply a fully live, fixed ACL Host allocation and retain
    it through deregistration. Users must quiesce local and remote accesses
    before close; the engine lock alone does not provide that guarantee.
    """

    def __init__(
        self, l1_memory_desc: L1MemoryDesc, listen_url: str, advertise_url: str
    ) -> None:
        """Initialize HIXL and register the exact supplied Host L1 range.

        Args:
            l1_memory_desc: Entire fixed Host allocation, not a lazy prefix.
            listen_url: Local handshake bind address, validated but not bound
                by this lifecycle foundation.
            advertise_url: Reachable handshake address published in metadata.

        Raises:
            ValueError: Configuration, addresses or the L1 range are invalid.
            RuntimeError: Native/runtime dependencies cannot load, ACL setup
                fails, or RegisterMem returns a null handle.
            HixlError: Native initialization or registration fails.

        Note:
            The initialization thread keeps the selected logical device.
            Failed construction rolls back acquired Engine/registration
            ownership without freeing the caller's memory or ACL context.
        """
        self._config = _read_config()
        _validate_memory_range(l1_memory_desc.ptr, l1_memory_desc.size)
        _parse_channel_url(listen_url)
        _parse_channel_url(advertise_url)
        self._l1_memory_desc = l1_memory_desc
        self._engine_lock = Lock()
        # Only a successfully initialized Engine is published here.
        self._engine: Hixl | None = None
        self._mem_handle: int | None = None

        try:
            # Third Party
            from lmcache_ascend.hixl_npu_comms import MEM_HOST, Hixl
            from lmcache_ascend.v1.hixl_runtime import (
                prepare_acl_context,
                use_acl_context,
            )
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HIXL Context requires compatible LMCache-Ascend HIXL bindings "
                "and prepare_acl_context/use_acl_context runtime helpers"
            ) from exc

        self._use_acl_context: Callable[[int], AbstractContextManager[None]] = (
            use_acl_context
        )
        options = self._config.initialize_options()
        self._acl_context = prepare_acl_context(self._config.device_id)
        with self._engine_lock, self._use_acl_context(self._acl_context):
            engine = Hixl()
            try:
                engine.initialize(self._config.engine_id, options)
            except BaseException:
                # Release the local owner in the ACL scope even on interruption.
                # Initialize's internal partial cleanup remains native-owned.
                engine = None
                raise
            self._engine = engine
            del engine  # Cleanup must not leave a local owner outside its scope.

        try:
            # Initialize may change the current context: use a fresh scope.
            with self._engine_lock, self._use_acl_context(self._acl_context):
                handle = self._engine.register_mem(
                    l1_memory_desc.ptr, l1_memory_desc.size, MEM_HOST
                )
                if handle == 0:
                    raise RuntimeError("HIXL RegisterMem returned a null handle")
                self._mem_handle = handle
            self._metadata = HixlChannelMetadata(
                advertise_url=advertise_url,
                engine_id=self._config.engine_id,
                buffer_base_ptr=l1_memory_desc.ptr,
                buffer_size=l1_memory_desc.size,
                instance_id=str(uuid4()),
            )
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self) -> HixlChannelMetadata:
        """Return immutable identity and bounds of the registered local L1.

        This snapshot is not a liveness check and remains readable after close.
        """
        return self._metadata

    def get_transfer_channel_address(
        self, lmcache_addresses: list[tuple[int, int]]
    ) -> list[TransferChannelAddress]:
        """Validate local byte ranges without alignment or address adjustment.

        Args:
            lmcache_addresses: Local (offset, size) pairs within the L1 pool.

        Returns:
            Public byte-offset addresses in input order; empty for empty input.
            These values do not keep the underlying memory alive.

        Raises:
            ValueError: An offset/length is invalid or outside the local pool.
        """
        addresses = []
        for index, (offset, size) in enumerate(lmcache_addresses):
            address = TransferChannelAddress(offset=offset, size=size)
            _validate_address_range(
                address, self.metadata.buffer_size, label=f"local_addresses[{index}]"
            )
            addresses.append(address)
        return addresses

    def close(self) -> None:
        """Deregister the L1 and finalize the Engine without freeing its memory.

        Call only after local and remote accesses have stopped. Sequential
        calls are safe; no cancellation, draining or automatic retry is added.
        ACL/native RuntimeErrors are logged so rollback preserves the original
        construction error. A cleanup attempt is not proof of successful free.

        Each native operation gets a fresh ACL scope. After attempting Finalize,
        release the Engine reference inside that scope. If scope entry fails,
        retain unprocessed ownership rather than call native cleanup in the
        wrong context. The caller must keep the borrowed runtime alive.
        """
        with self._engine_lock:
            if self._engine is None:
                return
            if self._mem_handle is not None:
                try:
                    with self._use_acl_context(self._acl_context):
                        try:
                            self._engine.deregister_mem(self._mem_handle)
                        finally:
                            # A native failure has uncertain consumption; do not
                            # replay the handle. Finalize owns remaining cleanup.
                            self._mem_handle = None
                except RuntimeError:
                    logger.exception(
                        "Failed to deregister HIXL L1 range at %s",
                        self._l1_memory_desc.ptr,
                    )
            try:
                with self._use_acl_context(self._acl_context):
                    try:
                        self._engine.finalize()
                    finally:
                        self._mem_handle = None
                        self._engine = None
            except RuntimeError:
                logger.exception(
                    "Failed to finalize HIXL engine %s", self._config.engine_id
                )

# SPDX-License-Identifier: Apache-2.0
"""HIXL Client for asynchronous transfers between registered Host L1 pools.

Context, Server and factory wiring are added separately. Importing this module
does not load the optional Ascend native extension or initialize the NPU.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Protocol
import ctypes

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.transfer_channel.abstract import TransferChannelClient
from lmcache.v1.distributed.transfer_channel.api import (
    TransferChannelAddress,
    TransferChannelTransferResult,
)

if TYPE_CHECKING:
    # Third Party
    from lmcache_ascend.hixl_npu_comms import (
        TransferOp,
        TransferOpDesc,
        TransferStatus,
    )

_UINT32_MAX = (1 << 32) - 1
_UINTPTR_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_SIZE_MAX = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1


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

# SPDX-License-Identifier: Apache-2.0
"""In-memory state machine of one shared region.

``RegionState`` holds the whole of the orchestrator's logic; the gRPC servicer
only converts messages and maps errors.
"""

# Standard
from collections import OrderedDict
from dataclasses import dataclass, field
import enum
import secrets
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_orchestrator.api import (
    MAX_UINT64,
    VISIBILITY_MODES,
    CloseResult,
    Envelope,
    Handle,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionUsage,
    RegisterResult,
    TokenStatus,
    WireLayout,
    WireObjectKey,
    WriteGrantResult,
    WriteRequest,
    WriteStatus,
)

logger = init_logger(__name__)

_INVALID_ARGUMENT = "INVALID_ARGUMENT"
_FAILED_PRECONDITION = "FAILED_PRECONDITION"
_RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"

_TOKEN_BYTES = 16
_GENERATION = 1
_MAX_READER_COUNT = 1024
_COMPLETED_TOKENS_PER_CLIENT = 65536


def _round_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a multiple of the power-of-two ``alignment``."""
    return (value + alignment - 1) & ~(alignment - 1)


class StateError(Exception):
    """A batch-level failure; the call that raised it changed no state.

    Attributes:
        code: Name of the gRPC status code the servicer replies with:
            ``INVALID_ARGUMENT``, ``FAILED_PRECONDITION`` or
            ``RESOURCE_EXHAUSTED``.
        message: Description sent to the client.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _ObjectState(enum.Enum):
    WRITING = enum.auto()
    VALID = enum.auto()


class _TokenOp(enum.Enum):
    """The call that completes a token; a token completes under one call."""

    FINISH_WRITE = enum.auto()
    ABORT_WRITE = enum.auto()
    FINISH_READ = enum.auto()


@dataclass
class _ObjectRecord:
    """A WRITING or VALID object; CONSUMED objects have no record."""

    state: _ObjectState
    handle: Handle
    payload_bytes: int
    layout: WireLayout


@dataclass
class _ClientRecord:
    """The registered incarnation of one client_id and what it holds."""

    incarnation: int
    register_result: RegisterResult
    # Live write token -> key of the WRITING object it may finish or abort.
    writes: dict[bytes, WireObjectKey] = field(default_factory=dict)
    # Live read lease -> key of the VALID object it was issued for.
    leases: dict[bytes, WireObjectKey] = field(default_factory=dict)
    # Completed tokens and the call that completed each, oldest first, so a
    # repeated call is answered OK again. Bounded to the newest 65536.
    completed: OrderedDict[bytes, _TokenOp] = field(default_factory=OrderedDict)


class RegionState:
    """Allocation and readiness authority for one shared region.

    Tracks the region's objects, extents, write tokens and read leases, and the
    registered incarnation of every client. Every method is atomic under one
    lock. A method that raises ``StateError`` changed no state; per-entry and
    per-token outcomes are returned in request order and never fail a batch.

    A region constructed with ``reset_required=True`` refuses every client
    call; only ``contract()``, ``usage()`` and ``registered_clients()`` work.

    Args:
        region_id: Name of the region; requests naming another are refused.
        capacity_bytes: Bytes that may be allocated; a positive multiple of
            ``alignment_bytes``.
        alignment_bytes: Extent alignment; a power of two.
        layout_fingerprint: Layout every registering client must present.
        visibility_mode: One of ``VISIBILITY_MODES``; every client must use it.
        max_batch_entries: Most entries or tokens one call may carry.
        region_epoch: Nonzero 64-bit epoch clients present after registering.
        reset_required: Whether the orchestrator restarted uncleanly and must
            refuse every client call.

    Raises:
        ValueError: An argument violates the constraints above.
    """

    def __init__(
        self,
        region_id: str,
        capacity_bytes: int,
        alignment_bytes: int,
        layout_fingerprint: bytes,
        visibility_mode: str,
        max_batch_entries: int,
        *,
        region_epoch: int,
        reset_required: bool,
    ) -> None:
        if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
            raise ValueError(
                f"alignment_bytes must be a power of two: {alignment_bytes}"
            )
        if capacity_bytes <= 0 or capacity_bytes % alignment_bytes:
            raise ValueError(
                f"capacity_bytes must be a positive multiple of {alignment_bytes}: "
                f"{capacity_bytes}"
            )
        if visibility_mode not in VISIBILITY_MODES:
            raise ValueError(f"visibility_mode must be one of {VISIBILITY_MODES}")
        if max_batch_entries < 1:
            raise ValueError(f"max_batch_entries must be >= 1: {max_batch_entries}")
        if not 0 < region_epoch <= MAX_UINT64:
            raise ValueError(f"region_epoch must be a nonzero uint64: {region_epoch}")
        self._contract = RegionContract(
            region_id=region_id,
            region_epoch=region_epoch,
            capacity_bytes=capacity_bytes,
            alignment_bytes=alignment_bytes,
            layout_fingerprint=layout_fingerprint,
            visibility_mode=visibility_mode,
            max_batch_entries=max_batch_entries,
            reset_required=reset_required,
        )
        self._lock = threading.Lock()
        # Guarded by self._lock: WRITING and VALID objects (a CONSUMED object
        # leaves the index), client_id -> registered incarnation, the bump
        # pointer that never decreases, the VALID bytes and CONSUMED count.
        self._objects: dict[WireObjectKey, _ObjectRecord] = {}
        self._clients: dict[str, _ClientRecord] = {}
        self._allocated_bytes = 0
        self._valid_bytes = 0
        self._consumed_count = 0

    def contract(self) -> RegionContract:
        """Return the immutable region contract, including ``reset_required``."""
        return self._contract

    def usage(self) -> RegionUsage:
        """Return a consistent snapshot of the region's counters."""
        with self._lock:
            clients = self._clients.values()
            writing = sum(len(client.writes) for client in clients)
            return RegionUsage(
                capacity_bytes=self._contract.capacity_bytes,
                allocated_bytes=self._allocated_bytes,
                valid_bytes=self._valid_bytes,
                writing=writing,
                valid=len(self._objects) - writing,
                consumed=self._consumed_count,
                read_leases=sum(len(client.leases) for client in clients),
                clients=len(self._clients),
            )

    def registered_clients(self) -> int:
        """Return the number of registered client incarnations."""
        with self._lock:
            return len(self._clients)

    def check_client(self, env: Envelope) -> None:
        """Validate ``env`` the way every client call does.

        Raises:
            StateError: ``FAILED_PRECONDITION`` for a wrong region, a region
                that needs a reset, a stale epoch or an unregistered client
                incarnation.
        """
        with self._lock:
            self._registered_client(env)

    def register_client(
        self,
        env: Envelope,
        layout_fingerprint: bytes,
        mapped_bytes: int,
        visibility_mode: str,
    ) -> RegisterResult:
        """Register a client incarnation, retiring an older one.

        Registering the registered incarnation again returns its original
        result and changes nothing. A new incarnation of a known client_id
        first retires the old one: its WRITING objects become CONSUMED, its
        leases are released and all its tokens become stale.

        Args:
            env: Envelope naming the region, its epoch and the incarnation.
            layout_fingerprint: The client's layout; must equal the region's.
            mapped_bytes: Bytes of the region the client maps; at least the
                region capacity.
            visibility_mode: The client's mode; must equal the region's.

        Returns:
            The region epoch for later envelopes and what was retired.

        Raises:
            StateError: ``FAILED_PRECONDITION`` as in ``check_client``, or for
                a client that does not match the region.
        """
        with self._lock:
            self._check_region(env)
            contract = self._contract
            for what, client_value, region_value in (
                ("layout fingerprint", layout_fingerprint, contract.layout_fingerprint),
                ("visibility mode", visibility_mode, contract.visibility_mode),
            ):
                if client_value != region_value:
                    raise StateError(
                        _FAILED_PRECONDITION,
                        f"{what} mismatch: region uses {region_value!r}, client "
                        f"uses {client_value!r}",
                    )
            if mapped_bytes < contract.capacity_bytes:
                raise StateError(
                    _FAILED_PRECONDITION,
                    f"client maps {mapped_bytes} bytes, less than the region "
                    f"capacity of {contract.capacity_bytes} bytes",
                )
            previous = self._clients.get(env.client_id)
            if previous is not None and previous.incarnation == env.client_incarnation:
                return previous.register_result
            retired_writes = released_leases = 0
            if previous is not None:
                retired_writes, released_leases = self._retire(previous)
                logger.warning(
                    "Client %s registered incarnation %d; retired incarnation "
                    "%d: %d writes consumed, %d leases released",
                    env.client_id,
                    env.client_incarnation,
                    previous.incarnation,
                    retired_writes,
                    released_leases,
                )
            result = RegisterResult(
                region_epoch=contract.region_epoch,
                retired_writes=retired_writes,
                released_leases=released_leases,
            )
            self._clients[env.client_id] = _ClientRecord(env.client_incarnation, result)
            logger.info(
                "Registered client %s incarnation %d",
                env.client_id,
                env.client_incarnation,
            )
            return result

    def reserve_write(
        self, env: Envelope, entries: list[WriteRequest]
    ) -> list[WriteGrantResult]:
        """Reserve extents for new objects.

        A VALID key yields ``EXISTS_VALID`` and a WRITING key ``BUSY_WRITING``;
        neither allocates, and a competing writer never learns the handle or
        token. The absent keys are allocated as one set: if their aligned
        sizes do not all fit, every absent entry gets ``OUT_OF_SPACE`` and
        nothing is allocated. Otherwise each is ``WRITE_GRANTED`` with a fresh
        token and becomes a WRITING object owned by this client incarnation.

        Args:
            env: Envelope of a registered client incarnation.
            entries: Objects to reserve; distinct keys, ``payload_bytes > 0``,
                and a layout with at least one shape and one dtype per shape.

        Returns:
            One result per entry, in request order.

        Raises:
            StateError: ``FAILED_PRECONDITION`` as in ``check_client``,
                ``RESOURCE_EXHAUSTED`` above ``max_batch_entries``,
                ``INVALID_ARGUMENT`` for an invalid entry.
        """
        with self._lock:
            client = self._registered_client(env)
            self._check_batch_size(len(entries))
            seen: set[WireObjectKey] = set()
            for index, entry in enumerate(entries):
                shapes, dtypes = entry.layout.shapes, entry.layout.dtypes
                if entry.payload_bytes <= 0:
                    problem = f"payload_bytes must be > 0, got {entry.payload_bytes}"
                elif not shapes or len(shapes) != len(dtypes):
                    problem = "layout needs at least one shape and one dtype per shape"
                elif entry.key in seen:
                    problem = "duplicate key in batch"
                else:
                    seen.add(entry.key)
                    continue
                raise StateError(_INVALID_ARGUMENT, f"entry {index}: {problem}")

            alignment = self._contract.alignment_bytes
            grants = [WriteGrantResult(WriteStatus.OUT_OF_SPACE)] * len(entries)
            absent: list[int] = []
            needed_bytes = 0
            for index, entry in enumerate(entries):
                record = self._objects.get(entry.key)
                if record is None:
                    absent.append(index)
                    needed_bytes += _round_up(entry.payload_bytes, alignment)
                elif record.state is _ObjectState.VALID:
                    grants[index] = WriteGrantResult(WriteStatus.EXISTS_VALID)
                else:
                    grants[index] = WriteGrantResult(WriteStatus.BUSY_WRITING)
            if self._allocated_bytes + needed_bytes > self._contract.capacity_bytes:
                return grants  # every absent entry keeps OUT_OF_SPACE
            for index in absent:
                entry = entries[index]
                handle = Handle(
                    offset=self._allocated_bytes,
                    length=_round_up(entry.payload_bytes, alignment),
                    generation=_GENERATION,
                )
                token = secrets.token_bytes(_TOKEN_BYTES)
                self._allocated_bytes += handle.length
                self._objects[entry.key] = _ObjectRecord(
                    _ObjectState.WRITING, handle, entry.payload_bytes, entry.layout
                )
                client.writes[token] = entry.key
                grants[index] = WriteGrantResult(
                    WriteStatus.WRITE_GRANTED, handle=handle, token=token
                )
            return grants

    def finish_write(self, env: Envelope, tokens: list[bytes]) -> list[TokenStatus]:
        """Commit written objects: WRITING -> VALID.

        A live write token of this client incarnation commits its object and
        yields ``OK``; a token this incarnation already finished yields ``OK``
        again. Anything else (unknown, another client's, aborted or retired
        tokens, or a token used with another call) yields ``STALE_TOKEN``.

        Args:
            env: Envelope of a registered client incarnation.
            tokens: Write tokens from ``reserve_write``.

        Returns:
            One status per token, in request order.

        Raises:
            StateError: ``FAILED_PRECONDITION`` as in ``check_client``,
                ``RESOURCE_EXHAUSTED`` above ``max_batch_entries``.
        """
        return self._complete_tokens(env, tokens, _TokenOp.FINISH_WRITE)

    def abort_write(self, env: Envelope, tokens: list[bytes]) -> list[TokenStatus]:
        """Give up written objects: WRITING -> CONSUMED.

        The extent of an aborted object is never reused; its key leaves the
        index and may be reserved again on a new extent. Idempotency and
        staleness follow ``finish_write``; aborting a finished token yields
        ``STALE_TOKEN``.

        Args:
            env: Envelope of a registered client incarnation.
            tokens: Write tokens from ``reserve_write``.

        Returns:
            One status per token, in request order.

        Raises:
            StateError: As ``finish_write``.
        """
        return self._complete_tokens(env, tokens, _TokenOp.ABORT_WRITE)

    def reserve_read(
        self, env: Envelope, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        """Lease committed objects for reading.

        A VALID object yields ``READ_GRANTED`` with its handle, layout, payload
        size and ``reader_count`` fresh leases; a WRITING object yields
        ``BUSY_WRITING`` and an unknown key ``MISS``. The batch is partial by
        design: a miss never rolls back the leases granted beside it.

        Args:
            env: Envelope of a registered client incarnation.
            entries: Objects to lease, each with ``reader_count`` in [1, 1024].

        Returns:
            One result per entry, in request order.

        Raises:
            StateError: ``FAILED_PRECONDITION`` as in ``check_client``,
                ``RESOURCE_EXHAUSTED`` above ``max_batch_entries``,
                ``INVALID_ARGUMENT`` for a reader_count out of range.
        """
        with self._lock:
            client = self._registered_client(env)
            self._check_batch_size(len(entries))
            for index, entry in enumerate(entries):
                if not 1 <= entry.reader_count <= _MAX_READER_COUNT:
                    raise StateError(
                        _INVALID_ARGUMENT,
                        f"entry {index}: reader_count must be in "
                        f"[1, {_MAX_READER_COUNT}], got {entry.reader_count}",
                    )
            grants: list[ReadGrantResult] = []
            for entry in entries:
                record = self._objects.get(entry.key)
                if record is None:
                    grants.append(ReadGrantResult(ReadStatus.MISS))
                elif record.state is _ObjectState.WRITING:
                    grants.append(ReadGrantResult(ReadStatus.BUSY_WRITING))
                else:
                    leases = tuple(
                        secrets.token_bytes(_TOKEN_BYTES)
                        for _ in range(entry.reader_count)
                    )
                    for lease in leases:
                        client.leases[lease] = entry.key
                    grants.append(
                        ReadGrantResult(
                            ReadStatus.READ_GRANTED,
                            handle=record.handle,
                            leases=leases,
                            layout=record.layout,
                            payload_bytes=record.payload_bytes,
                        )
                    )
            return grants

    def finish_read(self, env: Envelope, tokens: list[bytes]) -> list[TokenStatus]:
        """Release read leases.

        A live lease of this client incarnation is released and yields ``OK``;
        idempotency and staleness follow ``finish_write``.

        Args:
            env: Envelope of a registered client incarnation.
            tokens: Leases from ``reserve_read``.

        Returns:
            One status per lease, in request order.

        Raises:
            StateError: As ``finish_write``.
        """
        return self._complete_tokens(env, tokens, _TokenOp.FINISH_READ)

    def close_client(self, env: Envelope) -> CloseResult:
        """Retire and unregister a client incarnation.

        Its WRITING objects become CONSUMED, its leases are released and all
        its tokens become stale, exactly as when a new incarnation registers.

        Args:
            env: Envelope of a registered client incarnation.

        Returns:
            The number of writes consumed and leases released.

        Raises:
            StateError: ``FAILED_PRECONDITION`` as in ``check_client``.
        """
        with self._lock:
            client = self._registered_client(env)
            aborted_writes, released_leases = self._retire(client)
            del self._clients[env.client_id]
            logger.info(
                "Closed client %s incarnation %d: %d writes consumed, %d leases "
                "released",
                env.client_id,
                env.client_incarnation,
                aborted_writes,
                released_leases,
            )
            return CloseResult(aborted_writes, released_leases)

    def _complete_tokens(
        self, env: Envelope, tokens: list[bytes], op: _TokenOp
    ) -> list[TokenStatus]:
        """Complete live tokens under ``op``; see ``finish_write``."""
        with self._lock:
            client = self._registered_client(env)
            self._check_batch_size(len(tokens))
            live = client.leases if op is _TokenOp.FINISH_READ else client.writes
            statuses: list[TokenStatus] = []
            for token in tokens:
                key = live.pop(token, None)
                if key is None:
                    # Completed by this same call before: OK again, else stale.
                    repeated = client.completed.get(token) is op
                    statuses.append(
                        TokenStatus.OK if repeated else TokenStatus.STALE_TOKEN
                    )
                    continue
                if op is _TokenOp.FINISH_WRITE:
                    record = self._objects[key]
                    record.state = _ObjectState.VALID
                    self._valid_bytes += record.handle.length
                elif op is _TokenOp.ABORT_WRITE:
                    del self._objects[key]
                    self._consumed_count += 1
                client.completed[token] = op
                if len(client.completed) > _COMPLETED_TOKENS_PER_CLIENT:
                    client.completed.popitem(last=False)
                statuses.append(TokenStatus.OK)
            return statuses

    def _check_region(self, env: Envelope) -> None:
        """Refuse a call for another region, a reset region or a stale epoch."""
        contract = self._contract
        if env.region_id != contract.region_id:
            raise StateError(
                _FAILED_PRECONDITION,
                f"region id mismatch: this orchestrator serves "
                f"{contract.region_id!r}, the request names {env.region_id!r}",
            )
        if contract.reset_required:
            raise StateError(
                _FAILED_PRECONDITION,
                f"RESET_REQUIRED: the previous orchestrator of region "
                f"{contract.region_id!r} stopped uncleanly; stop every MP server "
                "using the region, stop this orchestrator, delete its startup "
                "marker and start it again",
            )
        if env.expected_region_epoch != contract.region_epoch:
            raise StateError(
                _FAILED_PRECONDITION,
                f"stale region epoch {env.expected_region_epoch}; the region "
                f"epoch is {contract.region_epoch}",
            )

    def _registered_client(self, env: Envelope) -> _ClientRecord:
        """Validate ``env`` and return its registered client incarnation."""
        self._check_region(env)
        client = self._clients.get(env.client_id)
        if client is None or client.incarnation != env.client_incarnation:
            raise StateError(
                _FAILED_PRECONDITION,
                f"client not registered: client_id={env.client_id!r} "
                f"incarnation={env.client_incarnation}",
            )
        return client

    def _check_batch_size(self, count: int) -> None:
        """Refuse a batch larger than ``max_batch_entries``."""
        if count > self._contract.max_batch_entries:
            raise StateError(
                _RESOURCE_EXHAUSTED,
                f"batch of {count} entries exceeds max_batch_entries="
                f"{self._contract.max_batch_entries}",
            )

    def _retire(self, client: _ClientRecord) -> tuple[int, int]:
        """Consume a client's WRITING objects and count its leases as released.

        The caller drops the client record afterwards, which makes all its
        tokens stale.
        """
        for key in client.writes.values():
            del self._objects[key]
        self._consumed_count += len(client.writes)
        return len(client.writes), len(client.leases)

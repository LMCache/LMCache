# SPDX-License-Identifier: Apache-2.0
"""Client of the memory orchestrator, used by MP servers."""

# Standard
from collections.abc import Callable, Sequence
from typing import TypeVar
import secrets
import threading
import time

# Third Party
from google.protobuf.message import Message
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_orchestrator import _codec as codec
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc import (
    MemoryOrchestratorStub,
)
from lmcache.v1.memory_orchestrator.api import (
    MAX_UINT64,
    CloseResult,
    OrchestratorError,
    OrchestratorUnavailableError,
    ReadGrantResult,
    ReadRequest,
    RegionContract,
    RegionFencedError,
    RegionUsage,
    RegisterResult,
    RequestRejectedError,
    TokenStatus,
    WriteGrantResult,
    WriteRequest,
)

logger = init_logger(__name__)

_RETRYABLE_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)
_REJECTED_CODES = (
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ALREADY_EXISTS,
)
_RETRY_BACKOFF_S = 0.05

_ItemT = TypeVar("_ItemT")
_ReplyT = TypeVar("_ReplyT", bound=Message)


def _translate_rpc_error(
    rpc: str, exc: grpc.RpcError, attempts: int
) -> OrchestratorError:
    """Map a failed RPC to the client's exception hierarchy."""
    code, details = exc.code(), exc.details()
    if code in _RETRYABLE_CODES:
        return OrchestratorUnavailableError(
            f"{rpc} failed after {attempts} attempt(s): {code.name}: {details}"
        )
    if code is grpc.StatusCode.FAILED_PRECONDITION:
        return RegionFencedError(
            f"{rpc}: {details}", reset_required=details.startswith("RESET_REQUIRED")
        )
    if code in _REJECTED_CODES:
        return RequestRejectedError(f"{rpc}: {details}", code=code.name)
    return OrchestratorError(f"{rpc} failed: {code.name}: {details}")


def _check_result_count(rpc: str, received: int, sent: int) -> None:
    """Refuse a reply whose result count differs from the request's."""
    if received != sent:
        raise OrchestratorError(f"{rpc} returned {received} results for {sent} entries")


class MemoryOrchestratorClient:
    """Thread-safe client of one region's memory orchestrator.

    Call ``register()`` once, then reserve and finish objects, then ``close()``.
    Batches above the region's ``max_batch_entries`` are split into several RPCs
    and an empty batch sends none. Only ``UNAVAILABLE`` and ``DEADLINE_EXCEEDED``
    are retried, with the same request id and a short backoff.

    Every RPC method but ``close()`` raises ``OrchestratorUnavailableError`` when
    the retries run out, ``RegionFencedError`` when the orchestrator refuses the
    call (``reset_required`` if the region needs an offline reset),
    ``RequestRejectedError`` when it rejects the request, and ``OrchestratorError``
    for any other failure or a malformed reply.
    """

    def __init__(
        self,
        endpoint: str,
        region_id: str,
        client_id: str,
        *,
        rpc_timeout_s: float = 5.0,
        max_attempts: int = 3,
    ) -> None:
        """Create a client; no RPC is sent until the first call.

        Args:
            endpoint: ``HOST:PORT`` of the orchestrator.
            region_id: Region the orchestrator must serve.
            client_id: Identity of this MP server, stable across its restarts.
            rpc_timeout_s: Deadline of each attempt, in seconds; > 0.
            max_attempts: Attempts per RPC, including the first; >= 1.

        Raises:
            ValueError: If ``rpc_timeout_s`` or ``max_attempts`` is out of range.
        """
        if not rpc_timeout_s > 0:
            raise ValueError(f"rpc_timeout_s must be > 0, got {rpc_timeout_s}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._region_id = region_id
        self._client_id = client_id
        self._client_incarnation = secrets.randbelow(MAX_UINT64) + 1
        self._rpc_timeout_s = rpc_timeout_s
        self._max_attempts = max_attempts
        # A ReserveRead reply carries one lease per reader per key, so a full
        # batch can exceed gRPC's 4 MiB default receive limit.
        self._channel = grpc.insecure_channel(
            endpoint, options=[("grpc.max_receive_message_length", -1)]
        )
        self._stub = MemoryOrchestratorStub(self._channel)
        self._lock = threading.Lock()  # guards the four fields below
        self._next_request_id = 1  # ids start at 1 and never repeat
        self._region_epoch = 0  # set by register()
        self._max_batch_entries = 0  # 0 (no splitting) until a contract is read
        self._closed = False  # set by close()

    @property
    def client_id(self) -> str:
        """Identity of this MP server, stable across its restarts."""
        return self._client_id

    @property
    def client_incarnation(self) -> int:
        """Random nonzero 64-bit id of this client instance."""
        return self._client_incarnation

    @property
    def region_epoch(self) -> int:
        """Epoch this client registered under; 0 until ``register()`` succeeds."""
        with self._lock:
            return self._region_epoch

    def describe_region(self) -> RegionContract:
        """Fetch the region contract and keep its batch limit; works unregistered.

        Returns:
            The contract, including ``reset_required``.

        Raises:
            OrchestratorError: See the class docstring.
        """
        env = self._next_envelope(0)
        reply = self._invoke("DescribeRegion", self._stub.DescribeRegion, env)
        contract = codec.contract_from_proto(reply)
        with self._lock:
            self._max_batch_entries = contract.max_batch_entries
        return contract

    def register(
        self, *, layout_fingerprint: bytes, mapped_bytes: int, visibility_mode: str
    ) -> RegisterResult:
        """Register this incarnation, retiring any previous one of ``client_id``.

        Args:
            layout_fingerprint: Layout of this MP server's view of the region.
            mapped_bytes: Bytes of the region this MP server maps.
            visibility_mode: Visibility mode this MP server implements.

        Returns:
            The region epoch and what was retired from the previous incarnation.

        Raises:
            RegionFencedError: If the client mismatches the region or a reset is due.
            OrchestratorError: See the class docstring.
        """
        contract = self.describe_region()
        request = pb2.RegisterClientRequest(
            env=self._next_envelope(contract.region_epoch),
            layout_fingerprint=layout_fingerprint,
            mapped_bytes=mapped_bytes,
            visibility_mode=visibility_mode,
        )
        reply = self._invoke("RegisterClient", self._stub.RegisterClient, request)
        with self._lock:
            self._region_epoch = reply.region_epoch
        return RegisterResult(
            reply.region_epoch, reply.retired_writes, reply.released_leases
        )

    def reserve_write(self, entries: Sequence[WriteRequest]) -> list[WriteGrantResult]:
        """Reserve extents for new objects, all or nothing per RPC for absent keys.

        A failed split batch first aborts, best effort, its earlier RPCs' grants.

        Args:
            entries: Objects to reserve; distinct keys.

        Returns:
            One grant per entry, in request order.

        Raises:
            OrchestratorError: See the class docstring.
        """
        grants: list[WriteGrantResult] = []
        try:
            for chunk in self._chunks(entries):
                request = pb2.ReserveWriteRequest(
                    env=self._next_envelope(self.region_epoch),
                    entries=[codec.write_request_to_proto(entry) for entry in chunk],
                )
                reply = self._invoke("ReserveWrite", self._stub.ReserveWrite, request)
                _check_result_count("ReserveWrite", len(reply.grants), len(chunk))
                grants += [codec.write_grant_from_proto(g) for g in reply.grants]
        except OrchestratorError:
            tokens = [grant.token for grant in grants if grant.token is not None]
            self._roll_back("ReserveWrite", self.abort_write, tokens)
            raise
        return grants

    def finish_write(self, tokens: Sequence[bytes]) -> list[TokenStatus]:
        """Commit objects (WRITING -> VALID); call only once their copies are visible.

        Args:
            tokens: Write tokens from ``reserve_write``.

        Returns:
            One status per token, in request order; a retried token gets ``OK``.

        Raises:
            OrchestratorError: See the class docstring.
        """
        return self._token_batch("FinishWrite", self._stub.FinishWrite, tokens)

    def abort_write(self, tokens: Sequence[bytes]) -> list[TokenStatus]:
        """Give up written objects (WRITING -> CONSUMED).

        Args:
            tokens: Write tokens from ``reserve_write``.

        Returns:
            One status per token, in request order.

        Raises:
            OrchestratorError: See the class docstring.
        """
        return self._token_batch("AbortWrite", self._stub.AbortWrite, tokens)

    def reserve_read(self, entries: Sequence[ReadRequest]) -> list[ReadGrantResult]:
        """Lease committed objects for reading; a miss never undoes other leases.

        A failed split batch first releases, best effort, its earlier RPCs' leases.

        Args:
            entries: Objects to lease with their reader counts.

        Returns:
            One grant per entry, in request order.

        Raises:
            OrchestratorError: See the class docstring.
        """
        grants: list[ReadGrantResult] = []
        try:
            for chunk in self._chunks(entries):
                request = pb2.ReserveReadRequest(
                    env=self._next_envelope(self.region_epoch),
                    entries=[codec.read_request_to_proto(entry) for entry in chunk],
                )
                reply = self._invoke("ReserveRead", self._stub.ReserveRead, request)
                _check_result_count("ReserveRead", len(reply.grants), len(chunk))
                grants += [codec.read_grant_from_proto(g) for g in reply.grants]
        except OrchestratorError:
            leases = [lease for grant in grants for lease in grant.leases]
            self._roll_back("ReserveRead", self.finish_read, leases)
            raise
        return grants

    def finish_read(self, tokens: Sequence[bytes]) -> list[TokenStatus]:
        """Release leases; call only once every copy out is complete or cancelled.

        Args:
            tokens: Leases from ``reserve_read``.

        Returns:
            One status per lease, in request order; a released lease gets ``OK``.

        Raises:
            OrchestratorError: See the class docstring.
        """
        return self._token_batch("FinishRead", self._stub.FinishRead, tokens)

    def usage(self) -> RegionUsage:
        """Fetch the region's counters; requires a registered client.

        Returns:
            Allocation, object, lease and client counters.

        Raises:
            OrchestratorError: See the class docstring.
        """
        env = self._next_envelope(self.region_epoch)
        return codec.usage_from_proto(self._invoke("Usage", self._stub.Usage, env))

    def close(self) -> CloseResult | None:
        """Unregister if registered and close the channel; idempotent, never raises.

        Returns:
            What was retired, or None if the client never registered, was
            already closed, or could not unregister (logged).
        """
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            registered_epoch = self._region_epoch
        result: CloseResult | None = None
        if registered_epoch:
            env = self._next_envelope(registered_epoch)
            try:
                reply = self._invoke("CloseClient", self._stub.CloseClient, env)
                result = CloseResult(reply.aborted_writes, reply.released_leases)
            except OrchestratorError:
                logger.exception(
                    "Could not unregister client %s incarnation %d; the orchestrator "
                    "keeps its writes and leases until the client registers again",
                    self._client_id,
                    self._client_incarnation,
                )
        self._channel.close()
        return result

    def _next_envelope(self, expected_region_epoch: int) -> pb2.Envelope:
        """Build an envelope carrying a fresh request id."""
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
        return pb2.Envelope(
            region_id=self._region_id,
            expected_region_epoch=expected_region_epoch,
            client_id=self._client_id,
            client_incarnation=self._client_incarnation,
            request_id=request_id,
        )

    def _chunks(self, items: Sequence[_ItemT]) -> list[Sequence[_ItemT]]:
        """Split ``items`` into batches of at most ``max_batch_entries``."""
        with self._lock:
            limit = self._max_batch_entries
        size = limit if limit > 0 else max(len(items), 1)
        return [items[start : start + size] for start in range(0, len(items), size)]

    def _invoke(
        self, rpc: str, method: Callable[..., _ReplyT], request: Message
    ) -> _ReplyT:
        """Send one request, retrying transport failures with the same request id."""
        attempt = 1
        while True:
            try:
                return method(request, timeout=self._rpc_timeout_s)
            except grpc.RpcError as exc:
                if exc.code() not in _RETRYABLE_CODES or attempt >= self._max_attempts:
                    raise _translate_rpc_error(rpc, exc, attempt) from exc
                logger.debug(
                    "%s attempt %d of %d failed with %s; retrying",
                    rpc,
                    attempt,
                    self._max_attempts,
                    exc.code().name,
                )
            time.sleep(_RETRY_BACKOFF_S * 2 ** (attempt - 1))
            attempt += 1

    def _token_batch(
        self, rpc: str, method: Callable[..., pb2.ResultBatch], tokens: Sequence[bytes]
    ) -> list[TokenStatus]:
        """Send a finish or abort call for ``tokens``."""
        statuses: list[TokenStatus] = []
        for chunk in self._chunks(tokens):
            env = self._next_envelope(self.region_epoch)
            reply = self._invoke(rpc, method, pb2.TokenBatch(env=env, tokens=chunk))
            _check_result_count(rpc, len(reply.results), len(chunk))
            statuses.extend(codec.token_status_from_proto(s) for s in reply.results)
        return statuses

    def _roll_back(
        self, rpc: str, undo: Callable[[Sequence[bytes]], object], tokens: list[bytes]
    ) -> None:
        """Undo, best effort, what the earlier RPCs of a failed batch were granted."""
        try:
            undo(tokens)
        except OrchestratorError:
            logger.exception(
                "Could not roll back %d tokens of a failed %s batch; they stay held "
                "until client %s registers again or closes",
                len(tokens),
                rpc,
                self._client_id,
            )

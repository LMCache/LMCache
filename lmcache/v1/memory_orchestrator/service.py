# SPDX-License-Identifier: Apache-2.0
"""gRPC servicer of the memory orchestrator; ``RegionState`` holds the logic."""

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn, TypeVar
import hashlib
import threading

# Third Party
from google.protobuf.message import Message
import grpc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_orchestrator import _codec as codec
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc import (
    MemoryOrchestratorServicer,
)
from lmcache.v1.memory_orchestrator.api import Envelope, TokenStatus
from lmcache.v1.memory_orchestrator.state import RegionState, StateError

logger = init_logger(__name__)

_REPLY_WINDOW_SIZE = 1024
_ReplyT = TypeVar("_ReplyT", bound=Message)


def _abort(
    context: grpc.ServicerContext, code: grpc.StatusCode, message: str
) -> NoReturn:
    """Fail the RPC with ``code``; ``context.abort`` raises."""
    context.abort(code, message)
    raise RuntimeError("gRPC context abort unexpectedly returned")


def _abort_state_error(context: grpc.ServicerContext, exc: StateError) -> NoReturn:
    """Fail the RPC with the status named by a ``StateError``."""
    logger.debug("Refused request: %s", exc)
    _abort(context, grpc.StatusCode[exc.code], exc.message)


@dataclass
class _ReplyWindow:
    """Replies of one incarnation: request_id -> (request sha256, reply bytes)."""

    incarnation: int
    replies: OrderedDict[int, tuple[bytes, bytes]] = field(default_factory=OrderedDict)


class OrchestratorServicer(MemoryOrchestratorServicer):
    """Serves one ``RegionState`` over the ``MemoryOrchestrator`` service.

    A mutating RPC repeating a request id of its client incarnation gets the
    recorded reply if the request is byte-identical and ``ALREADY_EXISTS``
    otherwise. Failed requests are not recorded, so their retry runs again.

    Thread safety: mutating RPCs run one at a time under ``self._lock``, so a
    retry racing its original waits for the recorded reply; ``RegionState``
    serializes every call anyway.
    """

    def __init__(self, state: RegionState) -> None:
        """Create the servicer.

        Args:
            state: The region state this servicer exposes.
        """
        self._state = state
        self._lock = threading.Lock()
        # client_id -> window of its registered incarnation; holds exactly the
        # client_ids registered in self._state. Guarded by self._lock.
        self._windows: dict[str, _ReplyWindow] = {}

    def DescribeRegion(
        self, request: pb2.Envelope, context: grpc.ServicerContext
    ) -> pb2.RegionContract:
        """Return the region contract in any state; checks only the region id."""
        contract = self._state.contract()
        if request.region_id != contract.region_id:
            _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                f"region id mismatch: this orchestrator serves "
                f"{contract.region_id!r}, the request names {request.region_id!r}",
            )
        return codec.contract_to_proto(contract)

    def RegisterClient(
        self, request: pb2.RegisterClientRequest, context: grpc.ServicerContext
    ) -> pb2.RegisterClientReply:
        """Register a client incarnation; see ``RegionState.register_client``."""
        env = codec.envelope_from_proto(request.env)

        def execute() -> pb2.RegisterClientReply:
            result = self._state.register_client(
                env,
                request.layout_fingerprint,
                request.mapped_bytes,
                request.visibility_mode,
            )
            window = self._windows.get(env.client_id)
            if window is None or window.incarnation != env.client_incarnation:
                self._windows[env.client_id] = _ReplyWindow(env.client_incarnation)
            return pb2.RegisterClientReply(
                region_epoch=result.region_epoch,
                retired_writes=result.retired_writes,
                released_leases=result.released_leases,
            )

        return self._run_mutation(
            env, request, context, pb2.RegisterClientReply, execute
        )

    def ReserveWrite(
        self, request: pb2.ReserveWriteRequest, context: grpc.ServicerContext
    ) -> pb2.ReserveWriteReply:
        """Reserve extents; see ``RegionState.reserve_write``."""
        env = codec.envelope_from_proto(request.env)
        entries = [codec.write_request_from_proto(entry) for entry in request.entries]

        def execute() -> pb2.ReserveWriteReply:
            grants = self._state.reserve_write(env, entries)
            return pb2.ReserveWriteReply(grants=map(codec.write_grant_to_proto, grants))

        return self._run_mutation(env, request, context, pb2.ReserveWriteReply, execute)

    def FinishWrite(
        self, request: pb2.TokenBatch, context: grpc.ServicerContext
    ) -> pb2.ResultBatch:
        """Commit written objects; see ``RegionState.finish_write``."""
        return self._token_batch(request, context, self._state.finish_write)

    def AbortWrite(
        self, request: pb2.TokenBatch, context: grpc.ServicerContext
    ) -> pb2.ResultBatch:
        """Give up written objects; see ``RegionState.abort_write``."""
        return self._token_batch(request, context, self._state.abort_write)

    def ReserveRead(
        self, request: pb2.ReserveReadRequest, context: grpc.ServicerContext
    ) -> pb2.ReserveReadReply:
        """Lease committed objects; see ``RegionState.reserve_read``."""
        env = codec.envelope_from_proto(request.env)
        entries = [codec.read_request_from_proto(entry) for entry in request.entries]

        def execute() -> pb2.ReserveReadReply:
            grants = self._state.reserve_read(env, entries)
            return pb2.ReserveReadReply(grants=map(codec.read_grant_to_proto, grants))

        return self._run_mutation(env, request, context, pb2.ReserveReadReply, execute)

    def FinishRead(
        self, request: pb2.TokenBatch, context: grpc.ServicerContext
    ) -> pb2.ResultBatch:
        """Release read leases; see ``RegionState.finish_read``."""
        return self._token_batch(request, context, self._state.finish_read)

    def Usage(
        self, request: pb2.Envelope, context: grpc.ServicerContext
    ) -> pb2.RegionUsage:
        """Return the region's counters to a registered client."""
        try:
            self._state.check_client(codec.envelope_from_proto(request))
        except StateError as exc:
            _abort_state_error(context, exc)
        return codec.usage_to_proto(self._state.usage())

    def CloseClient(
        self, request: pb2.Envelope, context: grpc.ServicerContext
    ) -> pb2.CloseClientReply:
        """Retire and unregister a client; see ``RegionState.close_client``."""
        env = codec.envelope_from_proto(request)

        def execute() -> pb2.CloseClientReply:
            result = self._state.close_client(env)
            del self._windows[env.client_id]
            return pb2.CloseClientReply(
                aborted_writes=result.aborted_writes,
                released_leases=result.released_leases,
            )

        return self._run_mutation(env, request, context, pb2.CloseClientReply, execute)

    def _token_batch(
        self,
        request: pb2.TokenBatch,
        context: grpc.ServicerContext,
        apply: Callable[[Envelope, list[bytes]], list[TokenStatus]],
    ) -> pb2.ResultBatch:
        """Run a finish or abort call over ``request.tokens``."""
        env = codec.envelope_from_proto(request.env)
        tokens = list(request.tokens)

        def execute() -> pb2.ResultBatch:
            statuses = apply(env, tokens)
            return pb2.ResultBatch(results=map(codec.token_status_to_proto, statuses))

        return self._run_mutation(env, request, context, pb2.ResultBatch, execute)

    def _run_mutation(
        self,
        env: Envelope,
        request: Message,
        context: grpc.ServicerContext,
        reply_class: type[_ReplyT],
        execute: Callable[[], _ReplyT],
    ) -> _ReplyT:
        """Run ``execute`` with ``self._lock`` held, once per request id."""
        digest = hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
        with self._lock:
            window = self._windows.get(env.client_id)
            if window is not None and window.incarnation == env.client_incarnation:
                recorded = window.replies.get(env.request_id)
                if recorded is not None:
                    recorded_digest, recorded_reply = recorded
                    if recorded_digest != digest:
                        _abort(
                            context,
                            grpc.StatusCode.ALREADY_EXISTS,
                            f"request_id {env.request_id} of client {env.client_id!r} "
                            "was already used for a different request",
                        )
                    return reply_class.FromString(recorded_reply)
            try:
                reply = execute()
            except StateError as exc:
                _abort_state_error(context, exc)
            # Re-read: execute registers or retires the window of env's client.
            window = self._windows.get(env.client_id)
            if window is not None and window.incarnation == env.client_incarnation:
                window.replies[env.request_id] = (digest, reply.SerializeToString())
                if len(window.replies) > _REPLY_WINDOW_SIZE:
                    window.replies.popitem(last=False)
            return reply

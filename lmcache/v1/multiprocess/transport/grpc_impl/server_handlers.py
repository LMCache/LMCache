# SPDX-License-Identifier: Apache-2.0
"""Handler binding shared by the sync and aio gRPC request servers."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

# Third Party
import grpc

# First Party
from lmcache.v1.multiprocess.request_handler import (
    BoundRequestHandler,
    HandlerType,
    iter_request_handlers,
)
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import ServiceBinding
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)
from lmcache.v1.multiprocess.transport.grpc_impl.proto_codec import (
    RequestDecoder,
    ResponseEncoder,
)

GRPC_SERVER_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
    # gRPC binds with SO_REUSEPORT by default, so a second server on the same
    # port would start and split the clients instead of failing like ZMQ does.
    ("grpc.so_reuseport", 0),
)
CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"
# Numbers a client's calls to ordered methods, in the order it sent them.
CALL_SEQUENCE_METADATA_KEY = "lmcache-call-sequence"
# How long a call waits for an earlier one, which may have been lost.
AFFINITY_TURN_TIMEOUT_S = 1.0


@dataclass
class GrpcRequestHandler:
    """A module handler bound to one generated gRPC method."""

    operation: str
    handler: Callable[..., Any] | None
    handler_type: HandlerType
    requires_client_affinity: bool
    request_decoder: RequestDecoder
    response_encoder: ResponseEncoder


def bind_module_handlers(
    modules: Sequence[object],
) -> dict[str, BoundRequestHandler]:
    """Collect the decorated handlers of server modules by operation.

    Args:
        modules: Ordered business modules. A later module overrides an
            earlier handler for the same operation.

    Returns:
        The bound handlers, keyed by operation name.
    """
    handlers_by_operation: dict[str, BoundRequestHandler] = {}
    for module in modules:
        for registered in iter_request_handlers(module):
            handlers_by_operation[registered.operation] = registered
    return handlers_by_operation


def build_service_handlers(
    binding: ServiceBinding,
    handlers_by_operation: dict[str, BoundRequestHandler],
) -> dict[str, GrpcRequestHandler]:
    """Match every method of a generated service with its module handler.

    Methods without a module handler are kept with ``handler=None`` so the
    server can answer them with ``UNIMPLEMENTED``.

    Args:
        binding: The generated service to serve.
        handlers_by_operation: Handlers from ``bind_module_handlers``.

    Returns:
        The handler for every method, keyed by the method's full name.

    Raises:
        TypeError: If a module handler does not match its protocol types.
    """
    service_handlers: dict[str, GrpcRequestHandler] = {}
    codec_registry = get_method_codec_registry()
    for method in binding.descriptor.methods:
        method_codec = codec_registry.by_full_name[method.full_name]
        bound_handler = handlers_by_operation.get(method_codec.operation)
        if bound_handler is not None:
            method_codec.validate_handler(bound_handler.handler)
        service_handlers[method.full_name] = GrpcRequestHandler(
            operation=method_codec.operation,
            handler=(bound_handler.handler if bound_handler is not None else None),
            handler_type=(
                bound_handler.options.handler_type
                if bound_handler is not None
                else HandlerType.SERIALIZED
            ),
            requires_client_affinity=(
                bound_handler.options.requires_client_affinity
                if bound_handler is not None
                else False
            ),
            request_decoder=method_codec.request_decoder,
            response_encoder=method_codec.response_encoder,
        )
    return service_handlers


def client_turn(
    context: grpc.ServicerContext | grpc.aio.ServicerContext,
) -> tuple[int, int | None]:
    """Return who sent a call, and its place in that client's order.

    Clients send a random ID in their metadata; callers without one fall back
    to their peer address.

    Args:
        context: The call's servicer context, sync or aio.

    Returns:
        A hash identifying the calling client, which keeps its affinity
        handlers on one worker, and the call's sequence number, or None if
        the client didn't send one.
    """
    client_key = None
    sequence = None
    for key, value in context.invocation_metadata() or ():
        if key == CLIENT_ID_METADATA_KEY:
            client_key = hash(value)
        elif key == CALL_SEQUENCE_METADATA_KEY:
            sequence = int(value)
    return (hash(context.peer()) if client_key is None else client_key), sequence


class AffinityTurns:
    """Which call of each client may go to the affinity pool next.

    gRPC can hand a client's calls to the server out of order, so they are put
    back in the order the client numbered them, starting from 0. A call that
    never arrives must not stop the ones after it, so after
    ``AFFINITY_TURN_TIMEOUT_S`` the lowest waiting call skips it, and the calls
    behind it stay in order. If the missing call does arrive later, it runs late.
    A server restart looks the same, once per client, since the client's numbers
    no longer start from 0.

    Not thread-safe: the server serializes access.
    """

    def __init__(self) -> None:
        self._next: dict[int, int] = {}
        self._waiting: dict[int, set[int]] = {}

    def is_turn(self, client_key: int, sequence: int) -> bool:
        """Return whether no earlier call of this client is still expected.

        Args:
            client_key: The client, as ``client_turn`` returns it.
            sequence: The call's number.

        Returns:
            True if the call may go to the affinity pool now.
        """
        return sequence <= self._next.get(client_key, 0)

    def finish(self, client_key: int, sequence: int) -> None:
        """Let the call after ``sequence`` go next.

        Args:
            client_key: The client, as ``client_turn`` returns it.
            sequence: The number of the call that is done waiting.
        """
        if sequence >= self._next.get(client_key, 0):
            self._next[client_key] = sequence + 1

    def add_waiter(self, client_key: int, sequence: int) -> None:
        """Record that a call is waiting for its turn.

        Args:
            client_key: The client, as ``client_turn`` returns it.
            sequence: The waiting call's number.
        """
        self._waiting.setdefault(client_key, set()).add(sequence)

    def remove_waiter(self, client_key: int, sequence: int) -> None:
        """Record that a call stopped waiting.

        Args:
            client_key: The client, as ``client_turn`` returns it.
            sequence: The call's number.
        """
        waiting = self._waiting[client_key]
        waiting.discard(sequence)
        if not waiting:
            del self._waiting[client_key]

    def first_waiter(self, client_key: int) -> int | None:
        """Return the lowest waiting number of a client, if any.

        Args:
            client_key: The client, as ``client_turn`` returns it.

        Returns:
            The number, or None if none of the client's calls is waiting.
        """
        waiting = self._waiting.get(client_key)
        return min(waiting) if waiting else None

    def skip_to(self, client_key: int, sequence: int) -> bool:
        """Give up on the calls before ``sequence``, if it is the first waiting.

        Args:
            client_key: The client, as ``client_turn`` returns it.
            sequence: The number of the call that waited too long.

        Returns:
            True if the call may go now. False if a lower number is still
            waiting, which has to go first.
        """
        if self.first_waiter(client_key) != sequence:
            return False
        self._next[client_key] = sequence
        return True

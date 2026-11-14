# SPDX-License-Identifier: Apache-2.0
"""gRPC-only control RPCs between cache servers, separate from worker RPCs."""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, TypeVar, cast

# Third Party
import grpc

# First Party
from lmcache.v1.client.transport.errors import MPRequestError, MPServerUnavailableError
from lmcache.v1.client.transport.grpc_impl.auth import (
    GrpcClientAuthenticationInterceptor,
)
from lmcache.v1.common.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.transport.auth import (
    load_request_client_api_key,
)
from lmcache.v1.transport.futures import MessagingFuture
from lmcache.v1.transport.grpc._proto_gen import p2p_service_pb2 as pb
from lmcache.v1.transport.grpc._proto_gen.p2p_service_pb2_grpc import (
    P2PServiceStub,
)
from lmcache.v1.transport.grpc.endpoint import parse_grpc_target
from lmcache.v1.transport.grpc.proto_codec import (
    compile_request_codec_for_types,
    compile_response_decoder_for_type,
)
from lmcache.v1.transport.rpc import rpc_method

T = TypeVar("T")
_lookup_request, _ = compile_request_codec_for_types(
    pb.P2pLookupAndLockRequest, (list[ObjectKey], dict[int, MemoryLayoutDesc])
)
_unlock_request, _ = compile_request_codec_for_types(
    pb.P2pUnlockObjectsRequest, (list[ObjectKey],)
)
_reserve_write_request, _ = compile_request_codec_for_types(
    pb.P2pReserveWriteRequest, (list[ObjectKey], MemoryLayoutDesc)
)
_finish_write_request, _ = compile_request_codec_for_types(
    pb.P2pFinishWriteRequest, (list[ObjectKey], list[bool])
)
_query_result = compile_response_decoder_for_type(
    pb.P2pQueryLookupResultsResponse, list[TransferChannelAddress] | None
)
_reserve_write_result = compile_response_decoder_for_type(
    pb.P2pReserveWriteResponse, list[TransferChannelAddress]
)
_finish_write_result = compile_response_decoder_for_type(
    pb.P2pFinishWriteResponse, list[bool]
)

_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
)


def _future(call: grpc.Future[Any], decode: Callable[[Any], T]) -> MessagingFuture[T]:
    """Bridge a peer call to the adapter's existing timeout/error contract."""
    future: MessagingFuture[T] = MessagingFuture()

    def completed(call: grpc.Future[Any]) -> None:
        try:
            result = decode(call.result())
        except grpc.RpcError as exc:
            status = cast(grpc.Call, exc)
            error = (
                MPServerUnavailableError
                if status.code()
                in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED)
                else MPRequestError
            )
            future.set_exception(error(status.code().name, status.details() or ""))
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    call.add_done_callback(completed)
    return future


class P2PClient:
    """Call only the peer service; no worker methods or ZMQ support.

    Args:
        server_url: Peer endpoint using grpc:// or grpc+unix://.
        api_key: Optional credential; shared environment sources apply if omitted.

    Raises:
        ValueError: The endpoint is not gRPC or the credential is invalid.
    """

    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        key = load_request_client_api_key(api_key)
        channel: grpc.Channel = grpc.insecure_channel(
            parse_grpc_target(server_url),
            options=_GRPC_OPTIONS + (("grpc.enable_http_proxy", 0),),
        )
        if key is not None:
            channel = grpc.intercept_channel(
                channel, GrpcClientAuthenticationInterceptor(key)
            )
        self._channel = channel
        self._stub = P2PServiceStub(channel)

    @rpc_method
    def p2p_lookup_and_lock(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> MessagingFuture[int]:
        """Submit peer keys and group layouts; return the lookup task ID."""
        return _future(
            self._stub.P2PLookupAndLock.future(
                _lookup_request((keys, group_layout_descs), {}), wait_for_ready=True
            ),
            lambda reply: int(reply.task_id),
        )

    @rpc_method
    def p2p_query_lookup_results(
        self, task_id: int
    ) -> MessagingFuture[list[TransferChannelAddress] | None]:
        """Poll the task; None means pending, a list contains peer addresses."""
        return _future(
            self._stub.P2PQueryLookupResults.future(
                pb.P2pQueryLookupResultsRequest(task_id=task_id), wait_for_ready=True
            ),
            _query_result,
        )

    @rpc_method
    def p2p_unlock_objects(self, keys: list[ObjectKey]) -> MessagingFuture[None]:
        """Release the peer read locks for the supplied keys."""
        return _future(
            self._stub.P2PUnlockObjects.future(
                _unlock_request((keys,), {}), wait_for_ready=True
            ),
            lambda _reply: None,
        )

    @rpc_method
    def p2p_reserve_write(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> MessagingFuture[list[TransferChannelAddress]]:
        """Reserve one peer write buffer per key and return its address."""
        return _future(
            self._stub.P2PReserveWrite.future(
                _reserve_write_request((keys, layout_desc), {}), wait_for_ready=True
            ),
            _reserve_write_result,
        )

    @rpc_method
    def p2p_finish_write(
        self, keys: list[ObjectKey], succeeded: list[bool]
    ) -> MessagingFuture[list[bool]]:
        """Commit successful peer writes, discard failures, and report residency."""
        return _future(
            self._stub.P2PFinishWrite.future(
                _finish_write_request((keys, succeeded), {}), wait_for_ready=True
            ),
            _finish_write_result,
        )

    def close(self) -> None:
        """Close the peer channel and cancel outstanding calls."""
        self._channel.close()

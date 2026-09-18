# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    NO_SESSION_END_INFO,
    DeviceIPCWrapper,
    SessionEndInfo,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import common_pb2
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.base import (
    RegisteredMessageCodec,
)


def _write_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper, value: DeviceIPCWrapper
) -> None:
    message.pickled_payload = DeviceIPCWrapper.Serialize(value)


def _read_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper,
) -> DeviceIPCWrapper:
    return DeviceIPCWrapper.Deserialize(message.pickled_payload)


def _write_session_end_info(
    message: common_pb2.SessionEndInfo, value: SessionEndInfo
) -> None:
    message.finish_reason = value.finish_reason
    message.stop_token_id = value.stop_token_id


def _read_session_end_info(
    message: common_pb2.SessionEndInfo,
) -> SessionEndInfo:
    """Read a SessionEndInfo, mapping an absent stop token to "unknown".

    An EndSessionRequest without end_info yields the default submessage,
    whose stop_token_id is unset; the structural codec would read proto3's
    0 there, which is a valid token id. Presence tells the two apart.
    """
    if not message.HasField("stop_token_id"):
        if not message.finish_reason:
            return NO_SESSION_END_INFO
        return SessionEndInfo(finish_reason=message.finish_reason)
    return SessionEndInfo(
        finish_reason=message.finish_reason,
        stop_token_id=message.stop_token_id,
    )


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper]
    | RegisteredMessageCodec[common_pb2.SessionEndInfo, SessionEndInfo],
    ...,
]:
    """Return custom codecs for protobuf messages shared across services.

    Returns:
        Explicit registrations for shared non-structural message types.
    """
    return (
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.DeviceIpcWrapper",
            python_type=DeviceIPCWrapper,
            writer=_write_device_ipc_wrapper,
            reader=_read_device_ipc_wrapper,
            include_subclasses=True,
        ),
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.SessionEndInfo",
            python_type=SessionEndInfo,
            writer=_write_session_end_info,
            reader=_read_session_end_info,
        ),
    )

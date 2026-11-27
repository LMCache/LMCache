# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# Third Party
import msgspec

# First Party
from lmcache.v1.common.custom_types import RetrieveCoverage, RetrieveReply
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.transport.grpc._proto_gen import common_pb2
from lmcache.v1.transport.grpc.codecs.base import (
    RegisteredMessageCodec,
)


def _write_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper, value: DeviceIPCWrapper
) -> None:
    message.descriptor_payload = DeviceIPCWrapper.Serialize(value)


def _read_device_ipc_wrapper(
    message: common_pb2.DeviceIpcWrapper,
) -> DeviceIPCWrapper:
    return DeviceIPCWrapper.Deserialize(message.descriptor_payload)


def _write_retrieve_reply(message, value: RetrieveReply) -> None:
    message.event_ipc_handle = value[0]
    message.success = bool(value[1])
    if isinstance(value[1], RetrieveCoverage):
        value[1].validate()
        message.encoded_coverage = msgspec.msgpack.encode(value[1])


def _read_retrieve_reply(message) -> RetrieveReply:
    result = message.success
    if message.encoded_coverage:
        result = msgspec.msgpack.decode(message.encoded_coverage, type=RetrieveCoverage)
        result.validate()
    return message.event_ipc_handle, result


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper], ...
]:
    """Return custom codecs for protobuf messages shared across services.

    Returns:
        Explicit registrations for shared non-structural message types.
    """
    return (
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.RetrieveEventIpcHandleResult",
            python_type=RetrieveReply,
            writer=_write_retrieve_reply,
            reader=_read_retrieve_reply,
        ),
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.DeviceIpcWrapper",
            python_type=DeviceIPCWrapper,
            writer=_write_device_ipc_wrapper,
            reader=_read_device_ipc_wrapper,
            include_subclasses=True,
        ),
    )

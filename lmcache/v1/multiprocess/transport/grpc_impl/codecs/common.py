# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# First Party
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
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


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper], ...
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
    )

# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# First Party
from lmcache.v1.common.custom_types import EventBinding, KVEventProfile
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


def _write_hash(message: common_pb2.NativeBlockHash, value: bytes | int) -> None:
    if isinstance(value, bytes):
        message.raw = value
    elif value < 0:
        message.signed_value = value
    else:
        message.unsigned_value = value


def _read_hash(message: common_pb2.NativeBlockHash) -> bytes | int:
    field = message.WhichOneof("value")
    if field == "raw":
        return message.raw
    if field == "signed_value":
        return message.signed_value
    if field == "unsigned_value":
        return message.unsigned_value
    raise ValueError("Missing native block hash")


def _write_binding(message: common_pb2.EventBinding, value: EventBinding) -> None:
    message.profile.profile_id = value.profile.profile_id
    message.profile.block_size = value.profile.block_size
    for block_hash in value.block_hashes:
        _write_hash(message.block_hashes.add(), block_hash)
    if value.parent_block_hash is not None:
        _write_hash(message.parent_block_hash, value.parent_block_hash)


def _read_binding(message: common_pb2.EventBinding) -> EventBinding:
    return EventBinding(
        KVEventProfile(message.profile.profile_id, message.profile.block_size),
        tuple(_read_hash(h) for h in message.block_hashes),
        _read_hash(message.parent_block_hash)
        if message.HasField("parent_block_hash")
        else None,
    )


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper],
    RegisteredMessageCodec[common_pb2.EventBinding, EventBinding],
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
            protobuf_type="lmcache.mp.EventBinding",
            python_type=EventBinding,
            writer=_write_binding,
            reader=_read_binding,
        ),
    )

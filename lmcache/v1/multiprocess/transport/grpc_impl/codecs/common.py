# SPDX-License-Identifier: Apache-2.0
"""Codecs for protobuf types shared by multiple gRPC services."""

# First Party
from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
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


def _write_engine_group_info(
    message: common_pb2.EngineGroupInfo, value: EngineGroupInfo
) -> None:
    message.engine_group_id = value.engine_group_id
    message.layer_indices.extend(value.layer_indices)
    message.tokens_per_block = value.tokens_per_block
    message.sw_size_tokens = value.sw_size_tokens
    message.extra_object_group_tag = value.extra_object_group_tag
    message.recurrent_state = value.recurrent_state
    if value.null_block_id is None:
        message.no_null_block = True
    else:
        message.null_block_id = value.null_block_id


def _read_engine_group_info(message: common_pb2.EngineGroupInfo) -> EngineGroupInfo:
    null_policy = message.WhichOneof("null_block_policy")
    if null_policy == "no_null_block":
        null_block_id = None
    elif null_policy == "null_block_id":
        null_block_id = message.null_block_id
    else:
        null_block_id = 0
    return EngineGroupInfo(
        engine_group_id=message.engine_group_id,
        layer_indices=tuple(message.layer_indices),
        tokens_per_block=message.tokens_per_block,
        sw_size_tokens=message.sw_size_tokens,
        extra_object_group_tag=message.extra_object_group_tag,
        recurrent_state=message.recurrent_state,
        null_block_id=null_block_id,
    )


def get_message_codecs() -> tuple[
    RegisteredMessageCodec[common_pb2.DeviceIpcWrapper, DeviceIPCWrapper],
    RegisteredMessageCodec[common_pb2.EngineGroupInfo, EngineGroupInfo],
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
            protobuf_type="lmcache.mp.EngineGroupInfo",
            python_type=EngineGroupInfo,
            writer=_write_engine_group_info,
            reader=_read_engine_group_info,
        ),
    )

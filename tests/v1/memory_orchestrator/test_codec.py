# SPDX-License-Identifier: Apache-2.0
"""Check the orchestrator's shared wire types and lossless conversions."""

# Standard
from dataclasses import fields

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_orchestrator import _codec as codec
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator.api import (
    ReadRequest,
    WireLayout,
    WireObjectKey,
    WriteRequest,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import p2p_service_pb2


def test_shared_object_key_and_layout_round_trip() -> None:
    assert WireObjectKey is ObjectKey
    key = ObjectKey(
        b"\x00\xff",
        "model",
        37,
        11,
        "tenant-a",
        extra_attributes={"encryption_key": "key-1"},
    )
    names = {field.name for field in fields(ObjectKey)}
    proto_names = {
        name.removeprefix("encoded_")
        for name in p2p_service_pb2.ObjectKey.DESCRIPTOR.fields_by_name
    }
    assert names == proto_names
    entry = WriteRequest(
        key,
        257,
        WireLayout(((2, 3), (4,)), ("float16", "int8")),
    )
    encoded = codec.write_request_to_proto(entry)
    assert encoded.key.DESCRIPTOR is p2p_service_pb2.ObjectKey.DESCRIPTOR
    assert encoded.layout.DESCRIPTOR is p2p_service_pb2.MemoryLayoutDesc.DESCRIPTOR
    assert encoded.layout.shapes[0].DESCRIPTOR is p2p_service_pb2.TensorShape.DESCRIPTOR
    restored = pb2.WriteEntry.FromString(encoded.SerializeToString())
    restored_write = codec.write_request_from_proto(restored)
    assert restored_write == entry
    assert restored.key.encoded_extra_attributes
    assert restored_write.key.extra_attributes == {"encryption_key": "key-1"}
    read = ReadRequest(entry.key, 2)
    restored_read = codec.read_request_from_proto(codec.read_request_to_proto(read))
    assert restored_read == read
    assert restored_read.key.extra_attributes == {"encryption_key": "key-1"}

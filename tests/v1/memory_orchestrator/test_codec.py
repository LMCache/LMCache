# SPDX-License-Identifier: Apache-2.0
"""Check the orchestrator's shared wire types and lossless conversions."""

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
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
        identity=ObjectKeyIdentity(
            chunk_hash=b"\x00\xff",
            model_name="model",
            kv_rank=37,
            object_group_id=11,
            cache_salt="tenant-a",
            tags={"z": "last", "a": "first"},
        ),
        xattrs={"owner": "worker", "priority": 2},
    )
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
    decoded = codec.write_request_from_proto(restored)
    assert decoded == entry
    assert decoded.key.xattrs == key.xattrs
    read = ReadRequest(entry.key, 2)
    decoded_read = codec.read_request_from_proto(codec.read_request_to_proto(read))
    assert decoded_read == read
    assert decoded_read.key.xattrs == key.xattrs

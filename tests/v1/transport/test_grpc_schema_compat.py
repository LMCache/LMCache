# SPDX-License-Identifier: Apache-2.0
"""Compatibility and wire tests for canonical shared gRPC schemas."""

# Standard
import importlib

# First Party
from lmcache.v1.transport.grpc._proto_gen import common_pb2, p2p_service_pb2


def test_legacy_generated_module_imports_are_canonical_objects() -> None:
    canonical = importlib.import_module(
        "lmcache.v1.transport.grpc._proto_gen.common_pb2"
    )
    legacy = importlib.import_module(
        "lmcache.v1.multiprocess.transport.grpc_impl._proto_gen.common_pb2"
    )

    assert legacy is canonical
    assert legacy.ObjectKeyIdentity is canonical.ObjectKeyIdentity


def test_schema_names_field_numbers_and_wire_bytes_are_frozen() -> None:
    identity = common_pb2.ObjectKeyIdentity(
        chunk_hash=b"hash",
        kv_rank=2,
        model_name="model",
        object_group_id=3,
        cache_salt="salt",
        tags={"revision": "v1"},
    )
    key = p2p_service_pb2.ObjectKey(
        identity=identity,
        encoded_extra_attributes=b"attrs",
    )

    assert identity.DESCRIPTOR.full_name == "lmcache.mp.ObjectKeyIdentity"
    assert key.DESCRIPTOR.full_name == "lmcache.mp.ObjectKey"
    assert {
        field.name: field.number for field in identity.DESCRIPTOR.fields
    } == {
        "chunk_hash": 1,
        "kv_rank": 2,
        "model_name": 3,
        "object_group_id": 4,
        "cache_salt": 5,
        "tags": 6,
    }
    assert key.SerializeToString(deterministic=True).hex() == (
        "0a270a046861736810021a056d6f64656c20032a0473616c74320e0a0872657669"
        "73696f6e1202763112056174747273"
    )

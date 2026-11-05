# SPDX-License-Identifier: Apache-2.0
"""Shared structural protobuf codec tests."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl import proto_codec as legacy
from lmcache.v1.transport.grpc import proto_codec
from lmcache.v1.transport.grpc._proto_gen import common_pb2, p2p_service_pb2
from lmcache.v1.transport.grpc.codec_registry import (
    MessageCodecRegistry,
    RegisteredMessageCodec,
)


@dataclass(frozen=True)
class Identity:
    chunk_hash: bytes
    kv_rank: int
    model_name: str
    object_group_id: int
    cache_salt: str
    tags: dict[str, str]


@dataclass(frozen=True)
class Dims:
    values: tuple[int, ...]


def test_structural_codec_round_trips_without_endpoint_registry() -> None:
    value = Identity(b"hash", 2, "model", 3, "salt", {"revision": "v1"})
    encode = proto_codec.compile_response_encoder_for_type(
        common_pb2.ObjectKeyIdentity, Identity
    )
    decode = proto_codec.compile_response_decoder_for_type(
        common_pb2.ObjectKeyIdentity, Identity
    )

    message = encode(value)
    assert decode(message) == value
    assert message.SerializeToString(deterministic=True).hex() == (
        "0a046861736810021a056d6f64656c20032a0473616c74320e0a08726576697369"
        "6f6e12027631"
    )


def test_custom_message_codec_is_explicitly_injected() -> None:
    def write(message: p2p_service_pb2.TensorShape, value: Dims) -> None:
        message.dims.extend(value.values)

    def read(message: p2p_service_pb2.TensorShape) -> Dims:
        return Dims(tuple(message.dims))

    registry = MessageCodecRegistry(
        (
            RegisteredMessageCodec(
                protobuf_type="lmcache.mp.TensorShape",
                python_type=Dims,
                writer=write,
                reader=read,
            ),
        )
    )
    writer, reader = proto_codec._compile_message_codec(
        p2p_service_pb2.TensorShape.DESCRIPTOR, Dims, registry
    )
    message = p2p_service_pb2.TensorShape()
    writer(message, Dims((2, 4)))

    assert message.SerializeToString() == b"\n\x02\x02\x04"
    assert reader(message) == Dims((2, 4))


def test_legacy_facade_reexports_shared_codec_types() -> None:
    assert legacy.RequestEncoder is proto_codec.RequestEncoder
    assert legacy.ResponseDecoder is proto_codec.ResponseDecoder

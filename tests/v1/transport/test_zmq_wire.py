# SPDX-License-Identifier: Apache-2.0
"""Compatibility and byte-level tests for the shared ZMQ wire protocol."""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.multiprocess.transport.zmq_impl import wire as legacy_wire
from lmcache.v1.transport import zmq_wire


def test_legacy_zmq_wire_path_reexports_canonical_objects() -> None:
    names = (
        "LEGACY_OPERATION_IDS",
        "DEFAULT_ZMQ_PAYLOAD_CODEC",
        "RequestUID",
        "ZMQ_ERROR_OPERATION",
        "ZmqErrorResponse",
        "ZmqPayloadCodec",
        "decode_error_response",
        "decode_operation",
        "decode_request_uid",
        "encode_error_response",
        "encode_operation",
        "encode_request_uid",
    )

    for name in names:
        assert getattr(legacy_wire, name) is getattr(zmq_wire, name)


def test_zmq_operation_wire_bytes_are_frozen() -> None:
    assert zmq_wire.encode_operation("lookup") == b"\x08"
    assert zmq_wire.decode_operation(b"\x08") == "lookup"
    assert zmq_wire.encode_operation("future_extension_rpc") == bytes.fromhex(
        "b46675747572655f657874656e73696f6e5f727063"
    )


def test_request_uid_and_scalar_payload_wire_bytes_are_frozen() -> None:
    assert zmq_wire.encode_request_uid(128) == bytes.fromhex("cc80")
    assert zmq_wire.decode_request_uid(bytes.fromhex("cc80")) == 128
    assert zmq_wire.DEFAULT_ZMQ_PAYLOAD_CODEC.encode(True, int) == b"\x01"
    assert zmq_wire.DEFAULT_ZMQ_PAYLOAD_CODEC.decode(b"\x01", int) == 1


def test_payload_codec_requires_explicit_special_codec_injection() -> None:
    class Special:
        pass

    encoder = msgspec.msgpack.Encoder(enc_hook=lambda _obj: "special")
    decoder = msgspec.msgpack.Decoder(type=str)
    codec = zmq_wire.ZmqPayloadCodec({Special: (encoder, decoder)})

    assert codec.encode(Special(), Special) == msgspec.msgpack.encode("special")
    assert codec.decode(msgspec.msgpack.encode("special"), Special) == "special"
    with pytest.raises(TypeError):
        zmq_wire.DEFAULT_ZMQ_PAYLOAD_CODEC.encode(Special(), Special)


def test_payload_count_mismatch_is_rejected_before_decode() -> None:
    with pytest.raises(ValueError, match="Payload count"):
        zmq_wire.DEFAULT_ZMQ_PAYLOAD_CODEC.decode_payloads([b"\x01"], [])


def test_zmq_error_envelope_wire_bytes_are_frozen() -> None:
    response = zmq_wire.ZmqErrorResponse(
        code="BAD_REQUEST",
        operation="lookup",
        message="invalid key",
    )
    encoded = zmq_wire.encode_error_response(response)

    assert encoded == bytes.fromhex(
        "83a4636f6465ab4241445f52455155455354a96f7065726174696f6ea66c6f6f6b"
        "7570a76d657373616765ab696e76616c6964206b6579"
    )
    assert zmq_wire.decode_error_response(encoded) == response


def test_zmq_wire_rejects_unknown_and_malformed_operations() -> None:
    with pytest.raises(ValueError, match="Unknown legacy ZMQ operation id"):
        zmq_wire.decode_operation(msgspec.msgpack.encode(999))
    with pytest.raises(TypeError, match="Invalid ZMQ operation identifier"):
        zmq_wire.decode_operation(msgspec.msgpack.encode(["lookup"]))

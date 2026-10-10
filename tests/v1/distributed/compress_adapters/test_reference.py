# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for the portable reference record codec."""

# Standard
from collections.abc import Callable
from pathlib import Path
import hashlib
import zlib

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.compress_adapters import (
    RECORD_PAYLOAD_ALIGNMENT,
    CompressedChunkDescriptor,
    CompressedRecordFormatError,
    CompressedRecordHeader,
    CompressionCodec,
    CompressionFraming,
    StoredCompressionFormat,
    align_record_payload_offset,
    crc32_ieee,
    encode_record_header,
    parse_record_header,
    record_header_size,
)
from lmcache.v1.distributed.compress_adapters.reference import (
    decode_reference_record,
    encode_reference_record,
)
import lmcache.v1.distributed.compress_adapters as compress_adapters

_V1_RAW_DEFLATE_HELLO_RECORD = bytes.fromhex(
    "4c4d435201010100"
    "44000000"
    "5700000000000000"
    "0500000000000000"
    "01000000"
    "5c9845f7"
    "f13ce63c"
    "00000000"
    "5000000000000000"
    "07000000"
    "05000000"
    "86a61036"
    "00000000"
    "000000000000000000000000"
    "cb48cdc9c90700"
)

_V1_GZIP_HELLO_RECORD = bytes.fromhex(
    "4c4d435201010200"
    "44000000"
    "6900000000000000"
    "0500000000000000"
    "01000000"
    "37c8471e"
    "ae032f78"
    "00000000"
    "5000000000000000"
    "19000000"
    "05000000"
    "86a61036"
    "00000000"
    "000000000000000000000000"
    "1f8b08000000000000ffcb48cdc9c9070086a6103605000000"
)

_VENDOR_FIXTURE_DIRECTORY = Path(__file__).with_name("vendor_compatibility") / (
    "fixtures"
)


def _stored_format(framing: CompressionFraming) -> StoredCompressionFormat:
    """Return the reference codec's supported Deflate format."""
    return StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=framing,
    )


def _raw_deflate(data: bytes) -> bytes:
    """Produce one raw Deflate stream for malformed-record tests."""
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _one_chunk_record(
    compressed: bytes,
    *,
    uncompressed_size: int,
    uncompressed_crc32: int,
) -> bytes:
    """Wrap an explicit raw Deflate payload in public record metadata."""
    header_size = record_header_size(1)
    payload_offset = align_record_payload_offset(header_size)
    payload = b"\x00" * (payload_offset - header_size) + compressed
    header = CompressedRecordHeader(
        stored_format=_stored_format(CompressionFraming.RAW),
        chunks=(
            CompressedChunkDescriptor(
                payload_offset=payload_offset,
                compressed_size=len(compressed),
                uncompressed_size=uncompressed_size,
                uncompressed_crc32=uncompressed_crc32,
            ),
        ),
        record_size=payload_offset + len(compressed),
        compressed_payload_crc32=crc32_ieee(payload),
    )
    return encode_record_header(header) + payload


def test_reference_helpers_are_not_package_root_api() -> None:
    """Fixture helpers remain visibly separate from production contracts."""
    assert not hasattr(compress_adapters, "encode_record")
    assert not hasattr(compress_adapters, "decode_record")
    assert not hasattr(compress_adapters, "encode_reference_record")
    assert not hasattr(compress_adapters, "decode_reference_record")


@pytest.mark.parametrize(
    "framing",
    [CompressionFraming.RAW, CompressionFraming.GZIP],
)
def test_reference_record_round_trip(framing: CompressionFraming) -> None:
    """Both advertised framings preserve bytes across independent chunks."""
    original = bytes(range(256)) * 3 + b"partial-final-chunk"

    encoded = encode_reference_record(
        original,
        stored_format=_stored_format(framing),
        chunk_size=128,
    )

    assert (
        decode_reference_record(
            encoded,
            expected_uncompressed_size=len(original),
        )
        == original
    )


@pytest.mark.parametrize(
    ("framing", "wbits"),
    [
        (CompressionFraming.RAW, -zlib.MAX_WBITS),
        (CompressionFraming.GZIP, zlib.MAX_WBITS | 16),
    ],
)
def test_encoder_produces_independent_standard_streams(
    framing: CompressionFraming,
    wbits: int,
) -> None:
    """Every descriptor addresses one independently decodable stream."""
    original = b"abcdefghijklmnopqrstuvwxyz0123456789"
    encoded = encode_reference_record(
        original,
        stored_format=_stored_format(framing),
        chunk_size=16,
    )
    header = parse_record_header(encoded)

    assert [chunk.uncompressed_size for chunk in header.chunks] == [16, 16, 4]
    assert header.compressed_payload_crc32 == crc32_ieee(encoded[header.header_size :])
    decoded_chunks = [
        zlib.decompress(
            encoded[
                chunk.payload_offset : chunk.payload_offset + chunk.compressed_size
            ],
            wbits=wbits,
        )
        for chunk in header.chunks
    ]
    assert decoded_chunks == [b"abcdefghijklmnop", b"qrstuvwxyz012345", b"6789"]
    assert header.chunks[0].payload_offset == align_record_payload_offset(
        header.header_size
    )
    assert encoded[header.header_size : header.chunks[0].payload_offset] == b"\x00" * (
        header.chunks[0].payload_offset - header.header_size
    )
    assert all(
        chunk.payload_offset % RECORD_PAYLOAD_ALIGNMENT == 0 for chunk in header.chunks
    )
    assert all(
        chunk.uncompressed_size % RECORD_PAYLOAD_ALIGNMENT == 0
        for chunk in header.chunks[:-1]
    )
    assert all(
        encoded[
            current.payload_offset + current.compressed_size : following.payload_offset
        ]
        == b"\x00"
        * (following.payload_offset - current.payload_offset - current.compressed_size)
        for current, following in zip(
            header.chunks,
            header.chunks[1:],
            strict=False,
        )
    )


def test_empty_input_uses_a_descriptor_free_record() -> None:
    """Empty input needs no artificial compressed chunk."""
    encoded = encode_reference_record(
        b"",
        stored_format=_stored_format(CompressionFraming.GZIP),
        chunk_size=1024,
    )

    header = parse_record_header(encoded)
    assert header.chunks == ()
    assert header.record_size == header.header_size == len(encoded) == 44
    assert decode_reference_record(encoded, expected_uncompressed_size=0) == b""


@pytest.mark.parametrize(
    ("record", "framing"),
    [
        (_V1_RAW_DEFLATE_HELLO_RECORD, CompressionFraming.RAW),
        (_V1_GZIP_HELLO_RECORD, CompressionFraming.GZIP),
    ],
)
def test_decoder_accepts_frozen_version_1_full_records(
    record: bytes,
    framing: CompressionFraming,
) -> None:
    """Independent raw and Gzip vectors remain reference-decodable."""
    header = parse_record_header(record)

    assert header.stored_format == _stored_format(framing)
    assert header.record_size == len(record)
    assert decode_reference_record(record, expected_uncompressed_size=5) == b"hello"


@pytest.mark.parametrize(
    ("filename", "record"),
    [
        ("raw-deflate-v1.hex", _V1_RAW_DEFLATE_HELLO_RECORD),
        ("gzip-v1.hex", _V1_GZIP_HELLO_RECORD),
    ],
)
def test_vendor_probe_fixtures_match_frozen_version_1_records(
    filename: str,
    record: bytes,
) -> None:
    """Native probes consume the exact records frozen by the Python contract."""
    fixture_text = (_VENDOR_FIXTURE_DIRECTORY / filename).read_text()

    assert bytes.fromhex(fixture_text) == record


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record[:-1], "truncated record"),
        (lambda record: record + b"extra", "trailing bytes after record"),
    ],
)
def test_decoder_requires_exact_record_bytes(
    change: Callable[[bytes], bytes],
    message: str,
) -> None:
    """A reference decode consumes exactly one complete record."""
    record = _V1_RAW_DEFLATE_HELLO_RECORD

    with pytest.raises(CompressedRecordFormatError, match=message):
        decode_reference_record(
            change(record),
            expected_uncompressed_size=5,
        )


def test_decoder_requires_exact_caller_output_size_before_decompression() -> None:
    """The logical layout must exactly match metadata before payload processing."""
    with pytest.raises(CompressedRecordFormatError, match="caller expects 4"):
        decode_reference_record(
            _V1_RAW_DEFLATE_HELLO_RECORD,
            expected_uncompressed_size=4,
        )


def test_decoder_requires_caller_output_size_argument() -> None:
    """A reference decode cannot derive its allocation policy from the record."""
    with pytest.raises(TypeError, match="expected_uncompressed_size"):
        decode_reference_record(  # type: ignore[call-arg]
            _V1_RAW_DEFLATE_HELLO_RECORD,
        )


@pytest.mark.parametrize(
    "record",
    [
        bytearray(_V1_RAW_DEFLATE_HELLO_RECORD),
        memoryview(_V1_RAW_DEFLATE_HELLO_RECORD),
        memoryview(_V1_RAW_DEFLATE_HELLO_RECORD)[::2],
    ],
)
def test_decoder_rejects_mutable_or_aliased_record_input(
    record: bytearray | memoryview,
) -> None:
    """Reference decoding inherits the immutable validated-record boundary."""
    with pytest.raises(TypeError, match="data must be immutable bytes"):
        decode_reference_record(  # type: ignore[arg-type]
            record,
            expected_uncompressed_size=5,
        )


def test_decoder_handles_compressed_input_across_bounded_windows() -> None:
    """A payload larger than one input window remains completely decodable."""
    original = hashlib.shake_256(b"lmcache-window-test").digest(200_000)
    encoded = encode_reference_record(
        original,
        stored_format=_stored_format(CompressionFraming.RAW),
        chunk_size=len(original),
    )
    chunk = parse_record_header(encoded).chunks[0]

    assert chunk.compressed_size > 64 * 1024
    assert (
        decode_reference_record(
            encoded,
            expected_uncompressed_size=len(original),
        )
        == original
    )


def test_decoder_caps_large_compressed_input_with_tiny_advertised_output() -> None:
    """A large hostile payload stops at the advertised output boundary."""
    original = hashlib.shake_256(b"lmcache-window-bomb-test").digest(200_000)
    compressed = _raw_deflate(original)
    assert len(compressed) > 64 * 1024
    record = _one_chunk_record(
        compressed,
        uncompressed_size=1,
        uncompressed_crc32=crc32_ieee(original[:1]),
    )

    with pytest.raises(CompressedRecordFormatError, match="more than its advertised"):
        decode_reference_record(record, expected_uncompressed_size=1)


def test_decoder_rejects_self_consistent_invalid_deflate_payload() -> None:
    """Invalid Deflate reaches the payload decoder with coherent metadata."""
    record = _one_chunk_record(
        b"\x07",
        uncompressed_size=5,
        uncompressed_crc32=crc32_ieee(b"hello"),
    )

    with pytest.raises(CompressedRecordFormatError, match="not a valid Deflate"):
        decode_reference_record(record, expected_uncompressed_size=5)


def test_decoder_rejects_self_consistent_truncated_deflate_stream() -> None:
    """A shortened stream is rejected after record-length validation succeeds."""
    record = _one_chunk_record(
        _raw_deflate(b"hello")[:-1],
        uncompressed_size=5,
        uncompressed_crc32=crc32_ieee(b"hello"),
    )

    with pytest.raises(CompressedRecordFormatError, match="truncated Deflate stream"):
        decode_reference_record(record, expected_uncompressed_size=5)


def test_decoder_rejects_self_consistent_underproducing_stream() -> None:
    """A complete stream must produce the descriptor's exact output size."""
    record = _one_chunk_record(
        _raw_deflate(b"hell"),
        uncompressed_size=5,
        uncompressed_crc32=crc32_ieee(b"hello"),
    )

    with pytest.raises(
        CompressedRecordFormatError, match="produced 4 bytes; expected 5"
    ):
        decode_reference_record(record, expected_uncompressed_size=5)


def test_decoder_rejects_output_larger_than_descriptor() -> None:
    """A hostile stream cannot expand past its advertised chunk size."""
    record = _one_chunk_record(
        _raw_deflate(b"too much output"),
        uncompressed_size=3,
        uncompressed_crc32=crc32_ieee(b"too"),
    )

    with pytest.raises(CompressedRecordFormatError, match="more than its advertised"):
        decode_reference_record(record, expected_uncompressed_size=3)


def test_decoder_rejects_trailing_stream_inside_descriptor() -> None:
    """One descriptor cannot silently contain multiple Deflate streams."""
    record = _one_chunk_record(
        _raw_deflate(b"hello") + _raw_deflate(b"ignored"),
        uncompressed_size=5,
        uncompressed_crc32=crc32_ieee(b"hello"),
    )

    with pytest.raises(CompressedRecordFormatError, match="trailing compressed bytes"):
        decode_reference_record(record, expected_uncompressed_size=5)


def test_decoder_rejects_chunk_crc_mismatch() -> None:
    """Valid Deflate bytes must still match the descriptor integrity check."""
    record = _one_chunk_record(
        _raw_deflate(b"hello"),
        uncompressed_size=5,
        uncompressed_crc32=crc32_ieee(b"jello"),
    )

    with pytest.raises(CompressedRecordFormatError, match="CRC-32/IEEE mismatch"):
        decode_reference_record(record, expected_uncompressed_size=5)


@pytest.mark.parametrize("chunk_size", [True, 0, -1, 15, 17, 1 << 32])
def test_encoder_rejects_invalid_chunk_size(chunk_size: int) -> None:
    """Chunk sizing rejects booleans and values outside descriptor widths."""
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        encode_reference_record(
            b"data",
            stored_format=_stored_format(CompressionFraming.RAW),
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize("expected_size", [True, -1])
def test_decoder_rejects_invalid_expected_output_size(expected_size: int) -> None:
    """Expected sizes use explicit non-boolean integer semantics."""
    with pytest.raises((TypeError, ValueError), match="expected_uncompressed_size"):
        decode_reference_record(
            _V1_RAW_DEFLATE_HELLO_RECORD,
            expected_uncompressed_size=expected_size,
        )


def test_encoder_rejects_untyped_stored_format() -> None:
    """Configuration must construct a portable format before encoding."""
    with pytest.raises(TypeError, match="StoredCompressionFormat"):
        encode_reference_record(
            b"data",
            stored_format=object(),  # type: ignore[arg-type]
            chunk_size=4,
        )


def test_reference_encoder_rejects_noncontiguous_input() -> None:
    """Reference encoding requires one contiguous source byte range."""
    noncontiguous = memoryview(b"abcdef")[::2]

    with pytest.raises(TypeError, match="contiguous byte buffer"):
        encode_reference_record(
            noncontiguous,
            stored_format=_stored_format(CompressionFraming.RAW),
            chunk_size=16,
        )

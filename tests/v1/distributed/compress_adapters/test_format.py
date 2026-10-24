# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for portable compression-format identity."""

# Standard
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.compress_adapters import (
    MAX_RECORD_HEADER_SIZE,
    RECORD_FORMAT_VERSION,
    RECORD_MAGIC,
    RECORD_PAYLOAD_ALIGNMENT,
    CompressedChunkDescriptor,
    CompressedRecordFormatError,
    CompressedRecordHeader,
    CompressionCodec,
    CompressionFraming,
    PostDecompressTransform,
    StoredCompressionFormat,
    ValidatedCompressedRecord,
    align_record_payload_offset,
    crc32_ieee,
    encode_record_header,
    parse_record_header,
    record_header_size,
    validate_complete_record,
)

_V1_ONE_CHUNK_HEADER = bytes.fromhex(
    "4c4d435201010100"
    "44000000"
    "5300000000000000"
    "0500000000000000"
    "01000000"
    "a2b8006f"
    "379bb61d"
    "00000000"
    "5000000000000000"
    "03000000"
    "05000000"
    "86a61036"
    "00000000"
)

_V1_ONE_CHUNK_PAYLOAD = b"\x00" * 12 + b"abc"


def _rewrite_header_crc(encoded: bytearray) -> None:
    """Recompute the draft header checksum after a deliberate mutation."""
    struct.pack_into("<I", encoded, 32, 0)
    struct.pack_into("<I", encoded, 32, crc32_ieee(encoded))


def _sample_payload() -> bytes:
    """Return canonical padding and payload bytes for ``_sample_header``."""
    return b"\x00" * 4 + b"abcd" + b"\x00" * 12 + b"xyz"


def _sample_header() -> CompressedRecordHeader:
    """Return a valid two-chunk header with an alignment gap."""
    return CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(
            CompressedChunkDescriptor(
                payload_offset=96,
                compressed_size=4,
                uncompressed_size=16,
                uncompressed_crc32=0x12345678,
            ),
            CompressedChunkDescriptor(
                payload_offset=112,
                compressed_size=3,
                uncompressed_size=5,
                uncompressed_crc32=0x90ABCDEF,
            ),
        ),
        record_size=115,
        compressed_payload_crc32=crc32_ieee(_sample_payload()),
    )


def test_format_requires_explicit_codec_and_framing() -> None:
    """A stored format names its codec and framing without a backend."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    assert stored_format.codec is CompressionCodec.DEFLATE
    assert stored_format.framing is CompressionFraming.RAW
    assert stored_format.post_decompress_transform is PostDecompressTransform.NONE


def test_format_exposes_only_a_fully_specified_transform() -> None:
    """Version 1 does not advertise an underspecified byte-reorder mode."""
    assert list(PostDecompressTransform) == [PostDecompressTransform.NONE]


def test_format_is_immutable() -> None:
    """Stored-format identity cannot change after construction."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    with pytest.raises(FrozenInstanceError):
        stored_format.framing = CompressionFraming.GZIP  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("codec", "deflate", "codec must be a CompressionCodec"),
        ("framing", "raw", "framing must be a CompressionFraming"),
        (
            "post_decompress_transform",
            "none",
            "post_decompress_transform must be a PostDecompressTransform",
        ),
    ],
)
def test_format_rejects_untyped_values(
    field: str,
    value: str,
    message: str,
) -> None:
    """Configuration strings must be parsed before format construction."""
    kwargs: dict[str, object] = {
        "codec": CompressionCodec.DEFLATE,
        "framing": CompressionFraming.RAW,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match=message):
        StoredCompressionFormat(**kwargs)  # type: ignore[arg-type]


def test_record_header_round_trip_is_deterministic() -> None:
    """Encoding is stable and parsing recovers the public header value."""
    header = _sample_header()

    encoded_once = encode_record_header(header)
    encoded_twice = encode_record_header(header)

    assert encoded_once == encoded_twice
    assert len(encoded_once) == header.header_size == 92
    assert encoded_once[:4] == RECORD_MAGIC
    assert encoded_once[4] == RECORD_FORMAT_VERSION
    assert parse_record_header(encoded_once) == header


def test_v1_wire_format_matches_frozen_vector() -> None:
    """Version 1 field order, IDs, widths, and CRC remain byte-for-byte stable."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(
            CompressedChunkDescriptor(
                payload_offset=80,
                compressed_size=3,
                uncompressed_size=5,
                uncompressed_crc32=0x3610A686,
            ),
        ),
        record_size=83,
        compressed_payload_crc32=crc32_ieee(_V1_ONE_CHUNK_PAYLOAD),
    )

    assert encode_record_header(header) == _V1_ONE_CHUNK_HEADER
    assert parse_record_header(_V1_ONE_CHUNK_HEADER) == header


def test_complete_record_validation_retains_immutable_bytes() -> None:
    """Full validation promotes exact bytes beyond header-only metadata."""
    record = _V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD

    validated = validate_complete_record(record)

    assert isinstance(validated, ValidatedCompressedRecord)
    assert validated.header == parse_record_header(record)
    assert type(validated.record_bytes) is bytes
    assert validated.record_bytes is record


def test_complete_record_validation_rejects_payload_checksum_mismatch() -> None:
    """Corrupt stored bytes fail before a caller may submit native decoding."""
    record = bytearray(_V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD)
    record[-1] ^= 1

    with pytest.raises(
        CompressedRecordFormatError,
        match="compressed payload CRC-32/IEEE mismatch",
    ):
        validate_complete_record(bytes(record))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: record[:-1], "truncated record"),
        (lambda record: record + b"extra", "trailing bytes after record"),
    ],
)
def test_complete_record_validation_requires_exact_length(
    change: Callable[[bytes], bytes],
    message: str,
) -> None:
    """The public validator consumes exactly one complete record."""
    record = _V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD

    with pytest.raises(CompressedRecordFormatError, match=message):
        validate_complete_record(change(record))


@pytest.mark.parametrize(
    "data",
    [
        bytearray(_V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD),
        memoryview(_V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD),
        memoryview(_V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD)[::2],
    ],
)
def test_complete_record_validation_rejects_mutable_or_aliased_input(
    data: bytearray | memoryview,
) -> None:
    """Public validation rejects aliased buffers before parsing or copying."""
    with pytest.raises(TypeError, match="data must be immutable bytes"):
        validate_complete_record(data)  # type: ignore[arg-type]


def test_validated_record_bytes_remain_stable_for_repeated_consumers() -> None:
    """The stored public bytes cannot be released between later consumers."""
    record = _V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD
    validated = validate_complete_record(record)

    first_consumer = memoryview(validated.record_bytes)
    first_consumer.release()

    assert memoryview(validated.record_bytes).tobytes() == record


def test_validated_record_is_immutable() -> None:
    """Callers cannot replace metadata after full validation."""
    record = _V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD
    validated = validate_complete_record(record)

    with pytest.raises(FrozenInstanceError):
        validated.header = _sample_header()  # type: ignore[misc]


def test_header_crc_protects_compressed_payload_checksum() -> None:
    """Changing the expected payload checksum invalidates the header first."""
    record = bytearray(_V1_ONE_CHUNK_HEADER + _V1_ONE_CHUNK_PAYLOAD)
    record[36] ^= 1

    with pytest.raises(CompressedRecordFormatError, match="header CRC-32/IEEE"):
        validate_complete_record(bytes(record))


def test_complete_record_validation_rejects_nonzero_padding() -> None:
    """Checksum-valid alignment gaps still require deterministic zero bytes."""
    payload = bytearray(_sample_payload())
    payload[0] = 1
    header = replace(
        _sample_header(),
        compressed_payload_crc32=crc32_ieee(payload),
    )
    record = encode_record_header(header) + bytes(payload)

    with pytest.raises(CompressedRecordFormatError, match="non-zero alignment"):
        validate_complete_record(record)


def test_header_rejects_invalid_chunk_element_with_type_error() -> None:
    """Invalid descriptor elements fail the documented type contract first."""
    stored_format = StoredCompressionFormat(
        codec=CompressionCodec.DEFLATE,
        framing=CompressionFraming.RAW,
    )

    with pytest.raises(
        TypeError,
        match=r"chunks\[0\] must be a CompressedChunkDescriptor",
    ):
        CompressedRecordHeader(
            stored_format=stored_format,
            chunks=(object(),),  # type: ignore[arg-type]
            record_size=68,
            compressed_payload_crc32=0,
        )


@pytest.mark.parametrize("checksum", [True, 1 << 32])
def test_header_rejects_invalid_compressed_payload_checksum(checksum: int) -> None:
    """Compressed-payload checksums use unsigned non-boolean wire integers."""
    with pytest.raises((TypeError, ValueError), match="compressed_payload_crc32"):
        replace(_sample_header(), compressed_payload_crc32=checksum)


def test_crc32_ieee_matches_the_standard_check_value() -> None:
    """CRC semantics match CRC-32/IEEE as implemented by ``zlib``."""
    assert crc32_ieee(b"123456789") == 0xCBF43926


def test_encoded_header_contains_crc32_ieee() -> None:
    """The stored header checksum covers fixed metadata and descriptors."""
    encoded = bytearray(encode_record_header(_sample_header()))
    stored_crc32 = struct.unpack_from("<I", encoded, 32)[0]
    struct.pack_into("<I", encoded, 32, 0)

    assert stored_crc32 == crc32_ieee(encoded)


def test_parser_rejects_header_checksum_mismatch() -> None:
    """A valid-looking metadata bit flip fails header integrity validation."""
    encoded = bytearray(encode_record_header(_sample_header()))
    encoded[48] ^= 1

    with pytest.raises(CompressedRecordFormatError, match="CRC-32/IEEE mismatch"):
        parse_record_header(encoded)


def test_empty_record_header_round_trip() -> None:
    """An empty record contains only the fixed header and no descriptors."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.GZIP,
        ),
        chunks=(),
        record_size=44,
        compressed_payload_crc32=crc32_ieee(b""),
    )

    encoded = encode_record_header(header)

    assert len(encoded) == 44
    assert header.uncompressed_size == 0
    assert parse_record_header(encoded) == header
    assert validate_complete_record(encoded).header == header


def test_record_header_size_reports_version_1_layout() -> None:
    """Callers can size a record before constructing its descriptors."""
    assert record_header_size(0) == 44
    assert record_header_size(2) == 92


@pytest.mark.parametrize(
    ("offset", "aligned"),
    [(0, 0), (68, 80), (80, 80), (81, 96)],
)
def test_record_payload_offset_alignment(offset: int, aligned: int) -> None:
    """Payload producers share the format's canonical alignment calculation."""
    assert align_record_payload_offset(offset) == aligned


@pytest.mark.parametrize("offset", [True, -1, (1 << 64) - 1])
def test_record_payload_offset_alignment_rejects_invalid_values(offset: int) -> None:
    """Alignment rejects invalid types, negative offsets, and overflow."""
    with pytest.raises((TypeError, ValueError), match="offset"):
        align_record_payload_offset(offset)


def test_record_header_size_enforces_version_1_chunk_boundary() -> None:
    """The largest sub-1-MiB table is accepted and one more is rejected."""
    assert record_header_size(43_688) == MAX_RECORD_HEADER_SIZE - 20
    with pytest.raises(ValueError, match="exceeds version-1 maximum 43688"):
        record_header_size(43_689)


@pytest.mark.parametrize("chunk_count", [True, -1, 1 << 32])
def test_record_header_size_rejects_invalid_counts(chunk_count: int) -> None:
    """Header sizing applies the same integer contract as record parsing."""
    with pytest.raises((TypeError, ValueError), match="chunk_count"):
        record_header_size(chunk_count)


@pytest.mark.parametrize(
    ("byte_offset", "replacement", "message"),
    [
        (0, ord("X"), "invalid record magic"),
        (4, RECORD_FORMAT_VERSION + 1, "unsupported record version"),
        (5, 0xFF, "unknown compression codec id"),
        (6, 0xFF, "unknown compression framing id"),
        (7, 0xFF, "unknown post-decompression transform id"),
    ],
)
def test_parser_rejects_unknown_fixed_header_values(
    byte_offset: int,
    replacement: int,
    message: str,
) -> None:
    """Magic, version, and enum identifiers fail closed when unknown."""
    encoded = bytearray(encode_record_header(_sample_header()))
    encoded[byte_offset] = replacement
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match=message):
        parse_record_header(encoded)


@pytest.mark.parametrize("size", [0, 1, 43, 91])
def test_parser_rejects_truncated_headers(size: int) -> None:
    """Both the fixed header and descriptor table must be complete."""
    encoded = encode_record_header(_sample_header())

    with pytest.raises(CompressedRecordFormatError, match="truncated"):
        parse_record_header(encoded[:size])


def test_parser_rejects_inconsistent_header_size() -> None:
    """The declared header size must match the descriptor count exactly."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 8, 32)

    with pytest.raises(CompressedRecordFormatError, match="header_size"):
        parse_record_header(encoded)


def test_parser_rejects_chunk_beyond_record_size() -> None:
    """A descriptor cannot address bytes beyond the declared record."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 12, 114)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="beyond record_size"):
        parse_record_header(encoded)


def test_parser_rejects_noncanonical_payload_offset() -> None:
    """Descriptor offsets must use the next minimal aligned address."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 68, 111)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="canonical 16-byte"):
        parse_record_header(encoded)


def test_header_rejects_unaligned_nonfinal_output_size() -> None:
    """Prefix-sum output ranges remain aligned after every non-final chunk."""
    header = _sample_header()
    first = replace(header.chunks[0], uncompressed_size=15)

    with pytest.raises(ValueError, match="not divisible by 16"):
        replace(header, chunks=(first, header.chunks[1]))


def test_parser_rejects_inconsistent_uncompressed_total() -> None:
    """The fixed-header output total must equal the descriptor sum."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<Q", encoded, 20, 14)
    _rewrite_header_crc(encoded)

    with pytest.raises(
        CompressedRecordFormatError,
        match="declared uncompressed size",
    ):
        parse_record_header(encoded)


def test_parser_rejects_reserved_chunk_flags() -> None:
    """Unknown per-chunk flags are rejected instead of silently ignored."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 64, 1)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="unsupported flags"):
        parse_record_header(encoded)


def test_parser_rejects_reserved_header_flags() -> None:
    """Unknown fixed-header flags are rejected instead of silently ignored."""
    encoded = bytearray(encode_record_header(_sample_header()))
    struct.pack_into("<I", encoded, 40, 1)
    _rewrite_header_crc(encoded)

    with pytest.raises(CompressedRecordFormatError, match="header uses unsupported"):
        parse_record_header(encoded)


def test_parser_rejects_impractical_chunk_table() -> None:
    """Wire-valid counts cannot force allocation or iteration past the limit."""
    header = CompressedRecordHeader(
        stored_format=StoredCompressionFormat(
            codec=CompressionCodec.DEFLATE,
            framing=CompressionFraming.RAW,
        ),
        chunks=(),
        record_size=44,
        compressed_payload_crc32=0,
    )
    encoded = bytearray(encode_record_header(header))
    excessive_chunk_count = (MAX_RECORD_HEADER_SIZE - 44) // 24 + 1
    struct.pack_into("<I", encoded, 8, 44 + excessive_chunk_count * 24)
    struct.pack_into("<I", encoded, 28, excessive_chunk_count)

    with pytest.raises(CompressedRecordFormatError, match="exceeds version-1"):
        parse_record_header(encoded)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("payload_offset", True, "payload_offset must be an int"),
        ("payload_offset", -1, "payload_offset must be in"),
        ("compressed_size", 0, "compressed_size must be greater than zero"),
        ("uncompressed_size", 0, "uncompressed_size must be greater than zero"),
        ("uncompressed_crc32", 1 << 32, "uncompressed_crc32 must be in"),
    ],
)
def test_chunk_descriptor_validates_wire_ranges(
    field: str,
    value: int,
    message: str,
) -> None:
    """Descriptors reject values that cannot be encoded or decoded safely."""
    kwargs = {
        "payload_offset": 32,
        "compressed_size": 1,
        "uncompressed_size": 1,
        "uncompressed_crc32": 0,
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        CompressedChunkDescriptor(**kwargs)


def test_header_uses_canonical_alignment_gaps_between_chunks() -> None:
    """Payload offsets use minimal alignment without including padding."""
    header = _sample_header()

    assert header.header_size == 92
    assert header.chunks[0].payload_offset == 96
    assert header.chunks[0].payload_offset % RECORD_PAYLOAD_ALIGNMENT == 0
    assert header.chunks[0].payload_offset + header.chunks[0].compressed_size == 100
    assert header.chunks[1].payload_offset == 112
    assert header.chunks[1].payload_offset % RECORD_PAYLOAD_ALIGNMENT == 0
    assert parse_record_header(encode_record_header(header)) == header

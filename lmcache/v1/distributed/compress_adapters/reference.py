# SPDX-License-Identifier: Apache-2.0
"""Pure-Python reference codec for portable compressed records.

This module turns byte buffers into complete LMCache compressed records and
decodes them on the CPU. It is a correctness oracle and fixture producer, not
the production serde or an accelerator backend. Each input chunk becomes an
independent Deflate stream so a future nvCOMP or hipCOMP backend can submit the
record as a batch without depending on Python or zlib-specific metadata.
"""

# Standard
import zlib

# First Party
from lmcache.v1.distributed.compress_adapters.format import (
    RECORD_PAYLOAD_ALIGNMENT,
    CompressedChunkDescriptor,
    CompressedRecordFormatError,
    CompressedRecordHeader,
    CompressionCodec,
    CompressionFraming,
    PostDecompressTransform,
    StoredCompressionFormat,
    align_record_payload_offset,
    crc32_ieee,
    encode_record_header,
    record_header_size,
    validate_complete_record,
)

_UINT32_MAX = (1 << 32) - 1
_DECOMPRESS_INPUT_WINDOW_SIZE = 64 * 1024


def _as_byte_view(
    name: str,
    data: bytes | bytearray | memoryview,
) -> memoryview:
    try:
        return memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose a contiguous byte buffer") from exc


def _require_chunk_size(chunk_size: int) -> None:
    if type(chunk_size) is not int:
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size <= 0 or chunk_size > _UINT32_MAX:
        raise ValueError(f"chunk_size must be in [1, {_UINT32_MAX}], got {chunk_size}")
    if chunk_size % RECORD_PAYLOAD_ALIGNMENT != 0:
        raise ValueError(
            f"chunk_size must be divisible by {RECORD_PAYLOAD_ALIGNMENT}, "
            f"got {chunk_size}"
        )


def _require_expected_uncompressed_size(expected_uncompressed_size: int) -> None:
    if type(expected_uncompressed_size) is not int:
        raise TypeError(
            "expected_uncompressed_size must be an int, got "
            f"{type(expected_uncompressed_size).__name__}"
        )
    if expected_uncompressed_size < 0:
        raise ValueError(
            "expected_uncompressed_size must be greater than or equal to zero, got "
            f"{expected_uncompressed_size}"
        )


def _deflate_wbits(stored_format: StoredCompressionFormat) -> int:
    if stored_format.codec is not CompressionCodec.DEFLATE:
        raise ValueError(
            "the reference codec supports only Deflate, got "
            f"{stored_format.codec.value}"
        )
    if stored_format.post_decompress_transform is not PostDecompressTransform.NONE:
        raise ValueError(
            "the reference codec does not support post-decompression "
            f"transform {stored_format.post_decompress_transform.value}"
        )
    if stored_format.framing is CompressionFraming.RAW:
        return -zlib.MAX_WBITS
    if stored_format.framing is CompressionFraming.GZIP:
        return zlib.MAX_WBITS | 16
    raise ValueError(
        f"the reference codec does not support framing {stored_format.framing.value}"
    )


def _compress_chunk(data: memoryview, wbits: int) -> bytes:
    compressor = zlib.compressobj(wbits=wbits)
    return compressor.compress(data) + compressor.flush()


def _decompress_chunk(
    data: memoryview,
    *,
    expected_size: int,
    wbits: int,
    chunk_index: int,
) -> bytes:
    output_limit = expected_size + 1
    output = bytearray()
    decompressor = zlib.decompressobj(wbits)
    input_offset = 0

    try:
        while input_offset < data.nbytes:
            window_end = min(
                input_offset + _DECOMPRESS_INPUT_WINDOW_SIZE,
                data.nbytes,
            )
            pending: bytes | memoryview = data[input_offset:window_end]
            input_offset = window_end

            while pending:
                decoded = decompressor.decompress(
                    pending,
                    output_limit - len(output),
                )
                output.extend(decoded)
                if len(output) > expected_size:
                    raise CompressedRecordFormatError(
                        f"chunks[{chunk_index}] produced more than its advertised "
                        f"{expected_size} bytes"
                    )
                pending = decompressor.unconsumed_tail

                if decompressor.eof:
                    if decompressor.unused_data or input_offset < data.nbytes:
                        raise CompressedRecordFormatError(
                            f"chunks[{chunk_index}] contains trailing compressed bytes"
                        )
                    break

        output.extend(decompressor.flush(output_limit - len(output)))
    except zlib.error as exc:
        raise CompressedRecordFormatError(
            f"chunks[{chunk_index}] is not a valid Deflate stream: {exc}"
        ) from exc

    if len(output) > expected_size:
        raise CompressedRecordFormatError(
            f"chunks[{chunk_index}] produced more than its advertised "
            f"{expected_size} bytes"
        )
    if not decompressor.eof:
        raise CompressedRecordFormatError(
            f"chunks[{chunk_index}] contains a truncated Deflate stream"
        )
    if decompressor.unused_data:
        raise CompressedRecordFormatError(
            f"chunks[{chunk_index}] contains trailing compressed bytes"
        )
    if len(output) != expected_size:
        raise CompressedRecordFormatError(
            f"chunks[{chunk_index}] produced {len(output)} bytes; expected "
            f"{expected_size}"
        )
    return bytes(output)


def encode_reference_record(
    data: bytes | bytearray | memoryview,
    *,
    stored_format: StoredCompressionFormat,
    chunk_size: int,
) -> bytes:
    """Encode bytes as an independently chunked portable record.

    Args:
        data: Contiguous uncompressed input bytes.
        stored_format: Portable codec, framing, and transform identity. The
            reference implementation currently supports Deflate with raw or
            Gzip framing and no post-decompression transform.
        chunk_size: Maximum uncompressed bytes in each independent stream.
            This value is required so the reference layer does not establish a
            production chunk-size default prematurely. It must be divisible by
            :data:`RECORD_PAYLOAD_ALIGNMENT`; only the final input chunk may
            contain fewer bytes.

    Returns:
        A complete record containing its header, descriptor table, and
        canonically aligned compressed payload chunks.

    Raises:
        TypeError: If an argument has the wrong type or ``data`` is not a
            contiguous byte buffer.
        ValueError: If ``chunk_size`` is outside the wire-format range, the
            format is unsupported, or the record exceeds version-1 bounds.

    Notes:
        Empty input is represented by a header with no chunk descriptors.
        Deflate output is standards-compliant but not canonical: compatible
        encoder versions may produce different compressed bytes.
    """
    view = _as_byte_view("data", data)
    if not isinstance(stored_format, StoredCompressionFormat):
        raise TypeError(
            "stored_format must be a StoredCompressionFormat, got "
            f"{type(stored_format).__name__}"
        )
    _require_chunk_size(chunk_size)
    wbits = _deflate_wbits(stored_format)

    chunk_count = (view.nbytes + chunk_size - 1) // chunk_size
    header_size = record_header_size(chunk_count)
    compressed_chunks = [
        _compress_chunk(view[offset : offset + chunk_size], wbits)
        for offset in range(0, view.nbytes, chunk_size)
    ]

    descriptors: list[CompressedChunkDescriptor] = []
    payload_parts: list[bytes] = []
    payload_offset = header_size
    for index, compressed in enumerate(compressed_chunks):
        aligned_payload_offset = align_record_payload_offset(payload_offset)
        payload_parts.append(b"\x00" * (aligned_payload_offset - payload_offset))
        payload_offset = aligned_payload_offset
        uncompressed = view[index * chunk_size : (index + 1) * chunk_size]
        descriptors.append(
            CompressedChunkDescriptor(
                payload_offset=payload_offset,
                compressed_size=len(compressed),
                uncompressed_size=uncompressed.nbytes,
                uncompressed_crc32=crc32_ieee(uncompressed),
            )
        )
        payload_parts.append(compressed)
        payload_offset += len(compressed)

    payload = b"".join(payload_parts)
    header = CompressedRecordHeader(
        stored_format=stored_format,
        chunks=tuple(descriptors),
        record_size=payload_offset,
        compressed_payload_crc32=crc32_ieee(payload),
    )
    return encode_record_header(header) + payload


def decode_reference_record(
    data: bytes,
    *,
    expected_uncompressed_size: int,
) -> bytes:
    """Validate and CPU-decode one complete portable compressed record.

    Args:
        data: Immutable bytes containing exactly one complete portable record.
        expected_uncompressed_size: Exact decoded byte count required by the
            caller's logical KV layout. This mandatory value prevents record
            metadata from choosing an unbounded allocation size.

    Returns:
        The concatenated uncompressed chunk bytes.

    Raises:
        TypeError: If an argument has the wrong type or ``data`` is not
            immutable :class:`bytes`.
        ValueError: If ``expected_uncompressed_size`` is negative.
        CompressedRecordFormatError: If metadata or payload bytes are invalid,
            the buffer is not exactly one record, its advertised output size
            differs from the caller's expectation, or a chunk fails size or
            CRC validation.

    Notes:
        Decompression of each chunk is capped one byte beyond its advertised
        output size and compressed input is supplied to zlib in bounded
        windows. This function intentionally does not apply a post-transform.
    """
    _require_expected_uncompressed_size(expected_uncompressed_size)
    validated_record = validate_complete_record(data)
    header = validated_record.header
    view = memoryview(validated_record.record_bytes)
    if header.uncompressed_size != expected_uncompressed_size:
        raise CompressedRecordFormatError(
            f"record advertises {header.uncompressed_size} uncompressed bytes, "
            f"but caller expects {expected_uncompressed_size}"
        )

    wbits = _deflate_wbits(header.stored_format)
    output_chunks: list[bytes] = []
    for index, chunk in enumerate(header.chunks):
        payload_end = chunk.payload_offset + chunk.compressed_size
        uncompressed = _decompress_chunk(
            view[chunk.payload_offset : payload_end],
            expected_size=chunk.uncompressed_size,
            wbits=wbits,
            chunk_index=index,
        )
        computed_crc32 = crc32_ieee(uncompressed)
        if computed_crc32 != chunk.uncompressed_crc32:
            raise CompressedRecordFormatError(
                f"chunks[{index}] CRC-32/IEEE mismatch: stored "
                f"0x{chunk.uncompressed_crc32:08x}, computed "
                f"0x{computed_crc32:08x}"
            )
        output_chunks.append(uncompressed)

    return b"".join(output_chunks)

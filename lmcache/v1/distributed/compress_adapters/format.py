# SPDX-License-Identifier: Apache-2.0
"""Portable compression-record vocabulary and draft header encoding.

The types in this module identify the bytes stored by LMCache independently of
the library that produced or consumes them. Vendor backends such as nvCOMP and
hipCOMP advertise which combinations they support; their names and native
descriptors do not become part of the stored-format identity.

The draft binary header is little-endian and contains no vendor/runtime data.
It describes independently decompressible payload chunks by absolute record
offset, compressed size, expected output size, and CRC-32/IEEE of the
uncompressed bytes. The header also carries CRC-32/IEEE of the complete stored
payload region so callers can detect accidental corruption before native
decompression. The header has its own CRC-32/IEEE covering the fixed header and
complete chunk table, with the header-checksum field zeroed while that checksum
is computed. Payload compression is intentionally outside this module.

Version 1 readers require the declared header size to exactly equal the fixed
header plus its chunk descriptors. Adding, removing, or resizing fields
therefore requires a new ``RECORD_FORMAT_VERSION``; readers do not skip unknown
header extensions.
"""

# Standard
from dataclasses import dataclass
import enum
import struct
import zlib

RECORD_MAGIC = b"LMCR"
"""Magic prefix for a portable LMCache compressed record."""

RECORD_FORMAT_VERSION = 1
"""Current draft portable-record version."""

MAX_RECORD_HEADER_SIZE = 1 << 20
"""Maximum accepted version-1 header size (1 MiB)."""

RECORD_PAYLOAD_ALIGNMENT = 16
"""Required byte alignment for every version-1 compressed payload."""

_UINT8_MAX = (1 << 8) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

# 44 bytes:
# magic, version, codec, framing, transform, header size, record size,
# total uncompressed size, chunk count, header CRC32, compressed-payload CRC32,
# reserved flags.
_FIXED_HEADER = struct.Struct("<4sBBBBIQQIIII")

# 24 bytes per chunk:
# payload offset, compressed size, uncompressed size, uncompressed CRC32,
# reserved flags.
_CHUNK_DESCRIPTOR = struct.Struct("<QIIII")

assert _FIXED_HEADER.size == 44
assert _CHUNK_DESCRIPTOR.size == 24

_HEADER_CRC32_OFFSET = 32
_MAX_RECORD_CHUNKS = (
    MAX_RECORD_HEADER_SIZE - _FIXED_HEADER.size
) // _CHUNK_DESCRIPTOR.size


def _require_uint(name: str, value: int, maximum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")


def crc32_ieee(data: bytes | bytearray | memoryview) -> int:
    """Return the unsigned CRC-32/IEEE checksum of ``data``.

    Args:
        data: Contiguous bytes to checksum.

    Returns:
        ``zlib.crc32(data) & 0xFFFFFFFF``.

    Raises:
        TypeError: If ``data`` does not expose a contiguous byte buffer.
    """
    try:
        view = memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("data must expose a contiguous byte buffer") from exc
    return zlib.crc32(view) & _UINT32_MAX


def align_record_payload_offset(offset: int) -> int:
    """Round ``offset`` up to the version-1 payload alignment.

    Args:
        offset: Nonnegative record-relative byte offset.

    Returns:
        The smallest aligned offset greater than or equal to ``offset``.

    Raises:
        TypeError: If ``offset`` is not an integer.
        ValueError: If ``offset`` or its aligned result is outside the
            unsigned 64-bit record-offset range.
    """
    _require_uint("offset", offset, _UINT64_MAX)
    aligned = (
        (offset + RECORD_PAYLOAD_ALIGNMENT - 1) // RECORD_PAYLOAD_ALIGNMENT
    ) * RECORD_PAYLOAD_ALIGNMENT
    if aligned > _UINT64_MAX:
        raise ValueError(
            f"aligned offset {aligned} exceeds unsigned 64-bit maximum {_UINT64_MAX}"
        )
    return aligned


def record_header_size(chunk_count: int) -> int:
    """Return the exact version-1 header size for ``chunk_count`` chunks.

    Args:
        chunk_count: Number of chunk descriptors in the record.

    Returns:
        Fixed-header bytes plus the complete descriptor-table size.

    Raises:
        TypeError: If ``chunk_count`` is not an integer.
        ValueError: If ``chunk_count`` is outside the version-1 resource
            bounds.
    """
    _require_uint("chunk_count", chunk_count, _UINT32_MAX)
    if chunk_count > _MAX_RECORD_CHUNKS:
        raise ValueError(
            f"chunk_count {chunk_count} exceeds version-1 maximum {_MAX_RECORD_CHUNKS}"
        )
    return _FIXED_HEADER.size + chunk_count * _CHUNK_DESCRIPTOR.size


def _header_crc32(data: bytes | bytearray | memoryview) -> int:
    header_bytes = bytearray(data)
    struct.pack_into("<I", header_bytes, _HEADER_CRC32_OFFSET, 0)
    return crc32_ieee(header_bytes)


class CompressedRecordFormatError(ValueError):
    """Raised when portable compressed-record metadata is malformed."""


class CompressionCodec(str, enum.Enum):
    """Logical lossless codec carried by a stored record.

    ``DEFLATE`` is the initial interoperability candidate from the RFC. Adding
    an enum member does not by itself make a codec supported; construction-time
    capability validation will make that decision.
    """

    DEFLATE = "deflate"


class CompressionFraming(str, enum.Enum):
    """Portable framing applied around a codec bitstream."""

    RAW = "raw"
    """An unwrapped codec bitstream, such as RFC 1951 Deflate bytes."""

    GZIP = "gzip"
    """The standard Gzip container around a Deflate bitstream."""


class PostDecompressTransform(str, enum.Enum):
    """Transform required after lossless device decompression."""

    NONE = "none"
    """Decompressed bytes are already in LMCache's contiguous KV layout."""


_CODEC_TO_ID = {CompressionCodec.DEFLATE: 1}
_ID_TO_CODEC = {value: key for key, value in _CODEC_TO_ID.items()}

_FRAMING_TO_ID = {
    CompressionFraming.RAW: 1,
    CompressionFraming.GZIP: 2,
}
_ID_TO_FRAMING = {value: key for key, value in _FRAMING_TO_ID.items()}

_TRANSFORM_TO_ID = {PostDecompressTransform.NONE: 0}
_ID_TO_TRANSFORM = {value: key for key, value in _TRANSFORM_TO_ID.items()}


@dataclass(frozen=True, kw_only=True, slots=True)
class StoredCompressionFormat:
    """Backend-independent identity of compressed bytes stored by LMCache.

    Args:
        codec: Logical lossless compression algorithm.
        framing: Container or bitstream framing around ``codec``.
        post_decompress_transform: Reversible transform required after
            decompression. Defaults to no transform.

    Raises:
        TypeError: If any argument is not a member of its declared enum.

    Notes:
        This type deliberately excludes backend names, device identities,
        streams, events, pointers, workspace requirements, and native library
        descriptors. Those belong to runtime capability and execution objects.
    """

    codec: CompressionCodec
    framing: CompressionFraming
    post_decompress_transform: PostDecompressTransform = PostDecompressTransform.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.codec, CompressionCodec):
            raise TypeError(
                f"codec must be a CompressionCodec, got {type(self.codec).__name__}"
            )
        if not isinstance(self.framing, CompressionFraming):
            raise TypeError(
                "framing must be a CompressionFraming, got "
                f"{type(self.framing).__name__}"
            )
        if not isinstance(self.post_decompress_transform, PostDecompressTransform):
            raise TypeError(
                "post_decompress_transform must be a PostDecompressTransform, "
                f"got {type(self.post_decompress_transform).__name__}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class CompressedChunkDescriptor:
    """Location and validation metadata for one compressed payload chunk.

    Args:
        payload_offset: Absolute byte offset from the beginning of the record.
        compressed_size: Number of stored payload bytes for this chunk.
        uncompressed_size: Exact number of bytes the decoder must produce.
        uncompressed_crc32: Unsigned CRC-32/IEEE of the uncompressed chunk
            bytes, calculated by :func:`crc32_ieee`.

    Raises:
        TypeError: If a field is not an integer.
        ValueError: If a field is outside its wire-format range or either size
            is zero.

    Notes:
        ``compressed_size`` covers exactly one complete codec stream. A
        producer must not include bytes after that stream's end marker. This
        semantic property is checked by a decoder rather than by structural
        header parsing.
    """

    payload_offset: int
    compressed_size: int
    uncompressed_size: int
    uncompressed_crc32: int

    def __post_init__(self) -> None:
        _require_uint("payload_offset", self.payload_offset, _UINT64_MAX)
        _require_uint("compressed_size", self.compressed_size, _UINT32_MAX)
        _require_uint("uncompressed_size", self.uncompressed_size, _UINT32_MAX)
        _require_uint("uncompressed_crc32", self.uncompressed_crc32, _UINT32_MAX)
        if self.compressed_size == 0:
            raise ValueError("compressed_size must be greater than zero")
        if self.uncompressed_size == 0:
            raise ValueError("uncompressed_size must be greater than zero")


@dataclass(frozen=True, kw_only=True, slots=True)
class CompressedRecordHeader:
    """Parsed metadata for one portable compressed record.

    Args:
        stored_format: Backend-independent codec, framing, and transform.
        chunks: Ordered compressed-payload descriptors.
        record_size: Exact total bytes in the header and payload record.
        compressed_payload_crc32: Unsigned CRC-32/IEEE of every stored byte in
            ``[header_size, record_size)``. This includes alignment gaps.
        version: Portable-record version. Defaults to the current draft
            version.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If sizes, offsets, ordering, or the version are invalid.

    Notes:
        Payloads use canonical minimal gaps required by
        :data:`RECORD_PAYLOAD_ALIGNMENT`; complete-record validation requires
        those gaps to contain only zero bytes. Every non-final chunk's expected
        output size is also alignment-sized so prefix-sum output ranges stay
        aligned. The final chunk must end at ``record_size``. An empty record
        has no chunks and consists only of the fixed header. Version 1 headers
        cannot exceed :data:`MAX_RECORD_HEADER_SIZE`.
    """

    stored_format: StoredCompressionFormat
    chunks: tuple[CompressedChunkDescriptor, ...]
    record_size: int
    compressed_payload_crc32: int
    version: int = RECORD_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.stored_format, StoredCompressionFormat):
            raise TypeError(
                "stored_format must be a StoredCompressionFormat, got "
                f"{type(self.stored_format).__name__}"
            )
        if not isinstance(self.chunks, tuple):
            raise TypeError(f"chunks must be a tuple, got {type(self.chunks).__name__}")
        for index, chunk in enumerate(self.chunks):
            if not isinstance(chunk, CompressedChunkDescriptor):
                raise TypeError(
                    f"chunks[{index}] must be a CompressedChunkDescriptor, got "
                    f"{type(chunk).__name__}"
                )
        _require_uint("version", self.version, _UINT8_MAX)
        if self.version != RECORD_FORMAT_VERSION:
            raise ValueError(
                f"unsupported record version {self.version}; "
                f"expected {RECORD_FORMAT_VERSION}"
            )
        _require_uint("record_size", self.record_size, _UINT64_MAX)
        _require_uint(
            "compressed_payload_crc32",
            self.compressed_payload_crc32,
            _UINT32_MAX,
        )
        _require_uint("chunk_count", len(self.chunks), _UINT32_MAX)
        _require_uint("header_size", self.header_size, _UINT32_MAX)
        _require_uint("uncompressed_size", self.uncompressed_size, _UINT64_MAX)
        if len(self.chunks) > _MAX_RECORD_CHUNKS:
            raise ValueError(
                f"chunk_count {len(self.chunks)} exceeds version-1 maximum "
                f"{_MAX_RECORD_CHUNKS}"
            )
        if self.header_size > MAX_RECORD_HEADER_SIZE:
            raise ValueError(
                f"header_size {self.header_size} exceeds version-1 maximum "
                f"{MAX_RECORD_HEADER_SIZE}"
            )

        if self.record_size < self.header_size:
            raise ValueError(
                f"record_size {self.record_size} is smaller than "
                f"header_size {self.header_size}"
            )

        previous_end = self.header_size
        for index, chunk in enumerate(self.chunks):
            expected_payload_offset = align_record_payload_offset(previous_end)
            if chunk.payload_offset != expected_payload_offset:
                raise ValueError(
                    f"chunks[{index}] payload_offset {chunk.payload_offset} "
                    f"does not match canonical {RECORD_PAYLOAD_ALIGNMENT}-byte "
                    f"aligned offset {expected_payload_offset}"
                )
            if (
                index < len(self.chunks) - 1
                and chunk.uncompressed_size % RECORD_PAYLOAD_ALIGNMENT != 0
            ):
                raise ValueError(
                    f"chunks[{index}] uncompressed_size "
                    f"{chunk.uncompressed_size} is not divisible by "
                    f"{RECORD_PAYLOAD_ALIGNMENT} for a non-final chunk"
                )
            chunk_end = chunk.payload_offset + chunk.compressed_size
            if chunk_end > self.record_size:
                raise ValueError(
                    f"chunks[{index}] ends at {chunk_end}, beyond record_size "
                    f"{self.record_size}"
                )
            previous_end = chunk_end

        if self.chunks:
            if previous_end != self.record_size:
                raise ValueError(
                    f"final chunk ends at {previous_end}, expected record_size "
                    f"{self.record_size}"
                )
        elif self.record_size != self.header_size:
            raise ValueError(
                "an empty record must contain only its fixed header: "
                f"record_size={self.record_size}, header_size={self.header_size}"
            )

    @property
    def header_size(self) -> int:
        """Return the encoded header size in bytes."""
        return record_header_size(len(self.chunks))

    @property
    def uncompressed_size(self) -> int:
        """Return the sum of all expected chunk output sizes."""
        return sum(chunk.uncompressed_size for chunk in self.chunks)


def encode_record_header(header: CompressedRecordHeader) -> bytes:
    """Encode a validated portable compressed-record header.

    Args:
        header: Header metadata to encode.

    Returns:
        Deterministic little-endian header bytes. Payload bytes are not
        included.

    Raises:
        TypeError: If ``header`` is not a :class:`CompressedRecordHeader`.
    """
    if not isinstance(header, CompressedRecordHeader):
        raise TypeError(
            f"header must be a CompressedRecordHeader, got {type(header).__name__}"
        )

    stored_format = header.stored_format
    descriptor_bytes = b"".join(
        _CHUNK_DESCRIPTOR.pack(
            chunk.payload_offset,
            chunk.compressed_size,
            chunk.uncompressed_size,
            chunk.uncompressed_crc32,
            0,
        )
        for chunk in header.chunks
    )
    fixed_header = _FIXED_HEADER.pack(
        RECORD_MAGIC,
        header.version,
        _CODEC_TO_ID[stored_format.codec],
        _FRAMING_TO_ID[stored_format.framing],
        _TRANSFORM_TO_ID[stored_format.post_decompress_transform],
        header.header_size,
        header.record_size,
        header.uncompressed_size,
        len(header.chunks),
        0,
        header.compressed_payload_crc32,
        0,
    )
    header_crc32 = crc32_ieee(fixed_header + descriptor_bytes)
    fixed_header = _FIXED_HEADER.pack(
        RECORD_MAGIC,
        header.version,
        _CODEC_TO_ID[stored_format.codec],
        _FRAMING_TO_ID[stored_format.framing],
        _TRANSFORM_TO_ID[stored_format.post_decompress_transform],
        header.header_size,
        header.record_size,
        header.uncompressed_size,
        len(header.chunks),
        header_crc32,
        header.compressed_payload_crc32,
        0,
    )
    return fixed_header + descriptor_bytes


def parse_record_header(
    data: bytes | bytearray | memoryview,
) -> CompressedRecordHeader:
    """Parse and validate a draft portable compressed-record header.

    Args:
        data: Bytes beginning at the start of a compressed record. The buffer
            must include the complete header and chunk table; payload bytes are
            optional.

    Returns:
        Validated immutable header metadata.

    Raises:
        TypeError: If ``data`` does not expose a contiguous byte buffer.
        CompressedRecordFormatError: If the header is truncated, contains an
            unknown identifier, uses reserved flags, or violates record bounds.

    Notes:
        The parser validates descriptor bounds against the declared
        ``record_size``. It does not require the payload itself to be present,
        allowing callers to inspect a header before allocating or loading the
        complete record.
    """
    try:
        view = memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("data must expose a contiguous byte buffer") from exc

    if view.nbytes < _FIXED_HEADER.size:
        raise CompressedRecordFormatError(
            f"truncated fixed header: got {view.nbytes} bytes, "
            f"need {_FIXED_HEADER.size}"
        )

    (
        magic,
        version,
        codec_id,
        framing_id,
        transform_id,
        declared_header_size,
        record_size,
        declared_uncompressed_size,
        chunk_count,
        declared_header_crc32,
        compressed_payload_crc32,
        header_flags,
    ) = _FIXED_HEADER.unpack_from(view)

    if magic != RECORD_MAGIC:
        raise CompressedRecordFormatError(f"invalid record magic {magic!r}")
    if version != RECORD_FORMAT_VERSION:
        raise CompressedRecordFormatError(
            f"unsupported record version {version}; expected {RECORD_FORMAT_VERSION}"
        )

    expected_header_size = _FIXED_HEADER.size + chunk_count * _CHUNK_DESCRIPTOR.size
    if chunk_count > _MAX_RECORD_CHUNKS:
        raise CompressedRecordFormatError(
            f"chunk_count {chunk_count} exceeds version-1 maximum {_MAX_RECORD_CHUNKS}"
        )
    if expected_header_size > MAX_RECORD_HEADER_SIZE:
        raise CompressedRecordFormatError(
            f"header_size {expected_header_size} exceeds version-1 maximum "
            f"{MAX_RECORD_HEADER_SIZE}"
        )
    if declared_header_size != expected_header_size:
        raise CompressedRecordFormatError(
            f"declared header_size {declared_header_size} does not match "
            f"chunk table size {expected_header_size}"
        )
    if view.nbytes < declared_header_size:
        raise CompressedRecordFormatError(
            f"truncated chunk table: got {view.nbytes} bytes, "
            f"need {declared_header_size}"
        )

    computed_header_crc32 = _header_crc32(view[:declared_header_size])
    if declared_header_crc32 != computed_header_crc32:
        raise CompressedRecordFormatError(
            f"header CRC-32/IEEE mismatch: stored 0x{declared_header_crc32:08x}, "
            f"computed 0x{computed_header_crc32:08x}"
        )
    if header_flags != 0:
        raise CompressedRecordFormatError(
            f"header uses unsupported flags 0x{header_flags:08x}"
        )

    stored_codec = _ID_TO_CODEC.get(codec_id)
    if stored_codec is None:
        raise CompressedRecordFormatError(f"unknown compression codec id {codec_id}")
    stored_framing = _ID_TO_FRAMING.get(framing_id)
    if stored_framing is None:
        raise CompressedRecordFormatError(
            f"unknown compression framing id {framing_id}"
        )
    stored_transform = _ID_TO_TRANSFORM.get(transform_id)
    if stored_transform is None:
        raise CompressedRecordFormatError(
            f"unknown post-decompression transform id {transform_id}"
        )

    chunks: list[CompressedChunkDescriptor] = []
    offset = _FIXED_HEADER.size
    try:
        for index in range(chunk_count):
            (
                payload_offset,
                compressed_size,
                uncompressed_size,
                uncompressed_crc32,
                flags,
            ) = _CHUNK_DESCRIPTOR.unpack_from(view, offset)
            if flags != 0:
                raise CompressedRecordFormatError(
                    f"chunks[{index}] uses unsupported flags 0x{flags:08x}"
                )
            chunks.append(
                CompressedChunkDescriptor(
                    payload_offset=payload_offset,
                    compressed_size=compressed_size,
                    uncompressed_size=uncompressed_size,
                    uncompressed_crc32=uncompressed_crc32,
                )
            )
            offset += _CHUNK_DESCRIPTOR.size

        header = CompressedRecordHeader(
            stored_format=StoredCompressionFormat(
                codec=stored_codec,
                framing=stored_framing,
                post_decompress_transform=stored_transform,
            ),
            chunks=tuple(chunks),
            record_size=record_size,
            compressed_payload_crc32=compressed_payload_crc32,
            version=version,
        )
    except CompressedRecordFormatError:
        raise
    except (TypeError, ValueError) as exc:
        raise CompressedRecordFormatError(str(exc)) from exc

    if header.uncompressed_size != declared_uncompressed_size:
        raise CompressedRecordFormatError(
            f"declared uncompressed size {declared_uncompressed_size} does not "
            f"match chunk total {header.uncompressed_size}"
        )
    return header


@dataclass(frozen=True, slots=True, init=False)
class ValidatedCompressedRecord:
    """Complete portable record whose stored payload checksum is valid.

    Args:
        data: Exact bytes of one complete portable record.

    Raises:
        TypeError: If ``data`` is not immutable :class:`bytes`.
        CompressedRecordFormatError: If metadata is invalid, the supplied
            length differs from ``record_size``, or the compressed-payload
            checksum does not match.

    Notes:
        The immutable input is retained without copying so the validated value
        cannot change or be invalidated after validation. A future zero-copy
        L1 path requires a separate guard that retains the source object's read
        lock for the complete native-use lifetime.

        CRC-32 detects accidental corruption; it does not authenticate a
        record from an attacker-controlled writer.
    """

    header: CompressedRecordHeader
    record_bytes: bytes

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError(f"data must be immutable bytes, got {type(data).__name__}")
        view = memoryview(data)

        header = parse_record_header(view)
        if view.nbytes < header.record_size:
            raise CompressedRecordFormatError(
                f"truncated record: got {view.nbytes} bytes, expected "
                f"{header.record_size}"
            )
        if view.nbytes > header.record_size:
            raise CompressedRecordFormatError(
                f"trailing bytes after record: got {view.nbytes} bytes, expected "
                f"{header.record_size}"
            )

        computed_payload_crc32 = crc32_ieee(view[header.header_size :])
        if computed_payload_crc32 != header.compressed_payload_crc32:
            raise CompressedRecordFormatError(
                "compressed payload CRC-32/IEEE mismatch: stored "
                f"0x{header.compressed_payload_crc32:08x}, computed "
                f"0x{computed_payload_crc32:08x}"
            )

        previous_end = header.header_size
        for index, chunk in enumerate(header.chunks):
            padding = view[previous_end : chunk.payload_offset]
            if any(padding):
                raise CompressedRecordFormatError(
                    f"non-zero alignment padding before chunks[{index}]"
                )
            previous_end = chunk.payload_offset + chunk.compressed_size

        object.__setattr__(self, "header", header)
        object.__setattr__(self, "record_bytes", data)


def validate_complete_record(data: bytes) -> ValidatedCompressedRecord:
    """Validate one complete record before native decompression.

    Args:
        data: Exact header, descriptor table, and stored payload bytes.

    Returns:
        A validated record retaining the supplied immutable bytes without a
        copy.

    Raises:
        TypeError: If ``data`` is not immutable :class:`bytes`.
        CompressedRecordFormatError: If the header, exact record length, or
            compressed-payload checksum is invalid.

    Notes:
        CRC-32 detects accidental corruption but does not authenticate
        attacker-controlled records. Native decompression requires a trusted
        or independently authenticated source.
    """
    return ValidatedCompressedRecord(data)

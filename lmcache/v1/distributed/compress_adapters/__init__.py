# SPDX-License-Identifier: Apache-2.0
"""Hardware-compression contracts for distributed KV storage."""

# First Party
from lmcache.v1.distributed.compress_adapters.format import (
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

__all__ = [
    "MAX_RECORD_HEADER_SIZE",
    "RECORD_FORMAT_VERSION",
    "RECORD_MAGIC",
    "RECORD_PAYLOAD_ALIGNMENT",
    "CompressedChunkDescriptor",
    "CompressedRecordFormatError",
    "CompressedRecordHeader",
    "CompressionCodec",
    "CompressionFraming",
    "PostDecompressTransform",
    "StoredCompressionFormat",
    "ValidatedCompressedRecord",
    "align_record_payload_offset",
    "crc32_ieee",
    "encode_record_header",
    "parse_record_header",
    "record_header_size",
    "validate_complete_record",
]

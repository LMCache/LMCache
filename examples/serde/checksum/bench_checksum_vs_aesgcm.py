# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

"""Focused in-memory benchmark for the checksum serde.

The benchmark compares a byte-copy baseline, the shipped xxh3_64 checksum
serde, and AES-GCM through the same Serializer/Deserializer interfaces used by
the production path. It measures serialization, deserialization, and combined
round-trip latency at representative KV chunk sizes.

Usage:
    python examples/serde/checksum/bench_checksum_vs_aesgcm.py
    python examples/serde/checksum/bench_checksum_vs_aesgcm.py \
        --chunk-sizes 131072 1048576
"""

# Standard
from typing import Any
import argparse
import ctypes
import os
import time

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.serde.aesgcm import (
    AesGcmDeserializer,
    AesGcmSerializer,
)
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.checksum import (
    ChecksumDeserializer,
    ChecksumSerializer,
)
from lmcache.v1.distributed.serde.key_provider import HkdfKeyProvider

_CHECKSUM_FRAME_OVERHEAD = 1 + 8 + 8
_AESGCM_FRAME_OVERHEAD = 1 + 12 + 16


class _ByteBuf:
    """Minimal MemoryObj stand-in exposing a mutable byte_array."""

    def __init__(self, data: bytes) -> None:
        self._arr = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)

    @property
    def byte_array(self) -> memoryview:
        """Return the mutable byte view consumed by serde implementations."""
        return memoryview(self._arr)

    @property
    def buf(self) -> bytes:
        """Return the current buffer contents."""
        return bytes(self._arr)


class _CopySerializer(Serializer):
    """Measure the fixed cost of the Serializer interface without hashing."""

    def serialize(self, src: Any, dst: Any, key: ObjectKey) -> int:
        payload = bytes(src.byte_array)
        out = memoryview(dst.byte_array).cast("B")
        out[: len(payload)] = payload
        return len(payload)

    def estimate_serialized_size(self, layout_desc: Any) -> int:
        raise NotImplementedError


class _CopyDeserializer(Deserializer):
    """Copy bytes without framing or integrity verification."""

    def deserialize(self, src: Any, dst: Any, key: ObjectKey) -> None:
        out = memoryview(dst.byte_array).cast("B")
        payload = bytes(src.byte_array)[: len(out)]
        out[: len(payload)] = payload


def _key() -> ObjectKey:
    """Return a stable object key for all benchmark cases."""
    return ObjectKey(chunk_hash=b"\x11" * 32, model_name="bench", kv_rank=0)


def _build_serde(name: str, master_key: bytes) -> tuple[Any, Any, int]:
    """Build a serde pair and return its fixed frame overhead in bytes."""
    if name == "copy":
        return _CopySerializer(), _CopyDeserializer(), 0
    if name == "checksum_xxh3_64":
        return ChecksumSerializer(), ChecksumDeserializer(), _CHECKSUM_FRAME_OVERHEAD
    if name == "aesgcm":
        provider = HkdfKeyProvider(master_key, key_len=16, info_prefix=b"bench")
        return (
            AesGcmSerializer(provider),
            AesGcmDeserializer(provider),
            _AESGCM_FRAME_OVERHEAD,
        )
    raise ValueError(f"unknown serde: {name}")


def _human_size(size: int) -> str:
    """Format a byte count using the units used in the benchmark output."""
    if size >= 1024 * 1024:
        return f"{size // (1024 * 1024)}MB"
    return f"{size // 1024}KB"


def benchmark_serde(
    name: str,
    chunk_size: int,
    master_key: bytes,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure one serde at one payload size.

    Args:
        name: Benchmark case name.
        chunk_size: Plaintext payload size in bytes.
        master_key: AES-GCM master key used by the encryption case.
        warmup: Number of unmeasured iterations.
        iterations: Number of measured iterations.

    Returns:
        A row containing average latency and throughput metrics.
    """
    plaintext = os.urandom(chunk_size)
    serializer, deserializer, overhead = _build_serde(name, master_key)
    encoded_size = chunk_size + overhead
    key = _key()

    def new_encoded_buffer() -> _ByteBuf:
        return _ByteBuf(bytes(encoded_size))

    def new_decoded_buffer() -> _ByteBuf:
        return _ByteBuf(bytes(chunk_size))

    for _ in range(warmup):
        src = _ByteBuf(plaintext)
        dst = new_encoded_buffer()
        size = serializer.serialize(src, dst, key)
        deserializer.deserialize(_ByteBuf(dst.buf[:size]), new_decoded_buffer(), key)

    serialize_times: list[float] = []
    deserialize_times: list[float] = []
    for _ in range(iterations):
        src = _ByteBuf(plaintext)
        dst = new_encoded_buffer()
        start = time.perf_counter()
        size = serializer.serialize(src, dst, key)
        serialize_times.append((time.perf_counter() - start) * 1000)

        frame = _ByteBuf(dst.buf[:size])
        decoded = new_decoded_buffer()
        start = time.perf_counter()
        deserializer.deserialize(frame, decoded, key)
        deserialize_times.append((time.perf_counter() - start) * 1000)

    serialize_ms = sum(serialize_times) / iterations
    deserialize_ms = sum(deserialize_times) / iterations
    return {
        "config": name,
        "chunk_size": _human_size(chunk_size),
        "chunk_bytes": chunk_size,
        "overhead_bytes": overhead,
        "serialize_ms": serialize_ms,
        "deserialize_ms": deserialize_ms,
        "roundtrip_ms": serialize_ms + deserialize_ms,
        "serialize_GBps": (chunk_size / (1024**3)) / (serialize_ms / 1000),
        "deserialize_GBps": (chunk_size / (1024**3)) / (deserialize_ms / 1000),
    }


def main() -> None:
    """Run the focused serde benchmark and print a Markdown table."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[128 * 1024, 1024 * 1024, 16 * 1024 * 1024],
        help="Chunk sizes in bytes (default: 128KB, 1MB, 16MB)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    master_key = os.urandom(16)
    configs = ("copy", "checksum_xxh3_64", "aesgcm")
    rows = [
        benchmark_serde(
            name,
            chunk_size,
            master_key,
            args.warmup,
            args.iters,
        )
        for chunk_size in args.chunk_sizes
        for name in configs
    ]

    headers = [
        "config",
        "chunk_size",
        "overhead_bytes",
        "serialize_ms",
        "deserialize_ms",
        "roundtrip_ms",
        "serialize_GBps",
        "deserialize_GBps",
    ]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["config"]),
                    str(row["chunk_size"]),
                    str(row["overhead_bytes"]),
                    f"{row['serialize_ms']:.4f}",
                    f"{row['deserialize_ms']:.4f}",
                    f"{row['roundtrip_ms']:.4f}",
                    f"{row['serialize_GBps']:.2f}",
                    f"{row['deserialize_GBps']:.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""
QAT hardware integration test: QatBackend + AccelCompressSerde + DramL2Adapter.

Requires:
  - KVCLIP_QZIP_LIB_PATH pointing to libkvclip_qzip.so
  - QAT hardware or SW fallback available
  - No GPU needed (uses CPU tensors as mock KV data)

Run:
  KVCLIP_QZIP_LIB_PATH=/path/to/libkvclip_qzip.so \
    python -m pytest tests/v1/distributed/test_qat_dram_l2_flow.py -v
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.compress_adapters.qat_backend import QatBackend
from lmcache.v1.distributed.compress_adapters.serde import (
    AccelCompressDeserializer,
    AccelCompressSerializer,
)
from lmcache.v1.distributed.l2_adapters.dram_l2_adapter import (
    DramL2Adapter,
    DramL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

# Skip entire module if QAT library is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("KVCLIP_QZIP_LIB_PATH"),
    reason="KVCLIP_QZIP_LIB_PATH not set (QAT library not available)",
)


# =============================================================================
# Helpers
# =============================================================================


def create_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def create_memory_obj(size: int, fill_value: int = 0) -> TensorMemoryObj:
    raw_data = torch.full((size,), fill_value, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.uint8,
        address=0,
        phy_size=size,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def qat_backend():
    """Create a QatBackend (one per module to amortize qzInit cost)."""
    backend = QatBackend()
    yield backend
    backend.close()


@pytest.fixture
def adapter():
    config = DramL2AdapterConfig(max_size_gb=0.01)
    a = DramL2Adapter(config)
    yield a
    a.close()


# =============================================================================
# Tests: QatBackend directly
# =============================================================================


class TestQatBackendDirect:
    """Validate QatBackend compress/decompress with raw buffers."""

    def test_roundtrip_constant_data(self, qat_backend):
        """Constant data should compress well and roundtrip exactly."""
        src = memoryview(bytearray(b"\xab" * 8192))
        dst = memoryview(bytearray(qat_backend.max_compressed_length(8192)))
        n = qat_backend.compress(src, dst)
        assert 0 < n < 8192  # should compress significantly

        out = memoryview(bytearray(8192))
        qat_backend.decompress(memoryview(dst)[:n], out)
        assert bytes(out) == bytes(src)

    def test_roundtrip_random_data(self, qat_backend):
        """Random data roundtrips correctly (may not compress well)."""
        # Standard
        import random

        random.seed(42)
        raw = bytearray(random.getrandbits(8) for _ in range(16384))
        src = memoryview(raw)
        dst = memoryview(bytearray(qat_backend.max_compressed_length(len(raw))))
        n = qat_backend.compress(src, dst)
        assert n > 0

        out = memoryview(bytearray(len(raw)))
        qat_backend.decompress(memoryview(dst)[:n], out)
        assert bytes(out) == bytes(src)

    def test_roundtrip_bf16_shaped_data(self, qat_backend):
        """Simulate bf16 KV cache data (torch randn then view as bytes)."""
        kv = torch.randn(32, 128, dtype=torch.bfloat16)  # 32 heads x 128 dim
        raw_bytes = bytearray(kv.view(torch.uint8).numpy().tobytes())
        size = len(raw_bytes)

        src = memoryview(raw_bytes)
        dst = memoryview(bytearray(qat_backend.max_compressed_length(size)))
        n = qat_backend.compress(src, dst)
        assert n > 0

        out = memoryview(bytearray(size))
        qat_backend.decompress(memoryview(dst)[:n], out)
        assert bytes(out) == bytes(src)


# =============================================================================
# Tests: Serde + QatBackend (no DramL2)
# =============================================================================


class TestQatSerde:
    """Validate AccelCompressSerializer/Deserializer with real QAT."""

    def test_serialize_deserialize_identity(self, qat_backend):
        """No preprocessing, just QAT compress/decompress."""
        serializer = AccelCompressSerializer(
            backend=qat_backend, byte_reorder=False, truncate_bits=0
        )
        deserializer = AccelCompressDeserializer(
            backend=qat_backend, byte_reorder=False
        )

        src = create_memory_obj(size=16384)
        src.raw_data[:] = torch.randint(0, 256, (16384,), dtype=torch.uint8)
        original = bytes(src.byte_array)

        max_c = qat_backend.max_compressed_length(16384)
        c_buf = create_memory_obj(size=max_c)
        c_size = serializer.serialize(src, c_buf)
        assert 0 < c_size <= max_c

        c_exact = create_memory_obj(size=c_size)
        c_exact.raw_data[:] = c_buf.raw_data[:c_size]

        output = create_memory_obj(size=16384)
        deserializer.deserialize(c_exact, output)
        assert bytes(output.byte_array) == original

    def test_serialize_with_shuffle(self, qat_backend):
        """byte_reorder (data_shuffle) + QAT should roundtrip."""
        serializer = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=True,
            truncate_bits=0,
            element_size=2,
        )
        deserializer = AccelCompressDeserializer(
            backend=qat_backend,
            byte_reorder=True,
            element_size=2,
        )

        src = create_memory_obj(size=8192)
        src.raw_data[:] = torch.randint(0, 256, (8192,), dtype=torch.uint8)
        original = bytes(src.byte_array)

        max_c = qat_backend.max_compressed_length(8192)
        c_buf = create_memory_obj(size=max_c)
        c_size = serializer.serialize(src, c_buf)

        c_exact = create_memory_obj(size=c_size)
        c_exact.raw_data[:] = c_buf.raw_data[:c_size]

        output = create_memory_obj(size=8192)
        deserializer.deserialize(c_exact, output)
        assert bytes(output.byte_array) == original

    def test_truncate_improves_compression(self, qat_backend):
        """Truncating LSBs should yield smaller compressed output."""
        data = torch.randint(0, 256, (8192,), dtype=torch.uint8)

        # Without truncation
        ser_no_trunc = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=False,
            truncate_bits=0,
            element_size=1,
        )
        src1 = create_memory_obj(size=8192)
        src1.raw_data[:] = data.clone()
        max_c = qat_backend.max_compressed_length(8192)
        c1 = create_memory_obj(size=max_c)
        size_no_trunc = ser_no_trunc.serialize(src1, c1)

        # With 4-bit truncation
        ser_trunc = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=False,
            truncate_bits=4,
            element_size=1,
        )
        src2 = create_memory_obj(size=8192)
        src2.raw_data[:] = data.clone()
        c2 = create_memory_obj(size=max_c)
        size_trunc = ser_trunc.serialize(src2, c2)

        # Truncated version should compress better (or equal)
        assert size_trunc <= size_no_trunc


# =============================================================================
# Tests: Full pipeline — QAT + Serde + DramL2Adapter
# =============================================================================


class TestQatDramL2Flow:
    """End-to-end: QAT compress → DramL2 store → load → QAT decompress."""

    def test_full_roundtrip(self, qat_backend, adapter):
        """Single key: compress, store, load, decompress, verify."""
        serializer = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=False,
            truncate_bits=0,
        )
        deserializer = AccelCompressDeserializer(
            backend=qat_backend,
            byte_reorder=False,
        )
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        # Create mock KV data
        src = create_memory_obj(size=32768)
        src.raw_data[:] = torch.randint(0, 256, (32768,), dtype=torch.uint8)
        original = bytes(src.byte_array)

        # Compress
        max_c = qat_backend.max_compressed_length(32768)
        c_buf = create_memory_obj(size=max_c)
        c_size = serializer.serialize(src, c_buf)

        c_obj = create_memory_obj(size=c_size)
        c_obj.raw_data[:] = c_buf.raw_data[:c_size]

        # Store
        key = create_object_key(1)
        task_id = adapter.submit_store_task([key], [c_obj])
        assert wait_for_event_fd(store_fd)
        results = adapter.pop_completed_store_tasks()
        assert results[task_id].is_successful()

        # Load
        loaded = create_memory_obj(size=c_size)
        load_task = adapter.submit_load_task([key], [loaded])
        assert wait_for_event_fd(load_fd)
        bitmap = adapter.query_load_result(load_task)
        assert bitmap.test(0) is True

        # Decompress
        output = create_memory_obj(size=32768)
        deserializer.deserialize(loaded, output)
        assert bytes(output.byte_array) == original

    def test_multiple_keys_roundtrip(self, qat_backend, adapter):
        """Multiple keys with different data, all roundtrip correctly."""
        serializer = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=True,
            truncate_bits=0,
            element_size=2,
        )
        deserializer = AccelCompressDeserializer(
            backend=qat_backend,
            byte_reorder=True,
            element_size=2,
        )
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        originals = {}
        compressed_sizes = {}
        num_keys = 8

        for i in range(num_keys):
            src = create_memory_obj(size=16384)
            src.raw_data[:] = torch.randint(0, 256, (16384,), dtype=torch.uint8)
            originals[i] = bytes(src.byte_array)

            max_c = qat_backend.max_compressed_length(16384)
            c_buf = create_memory_obj(size=max_c)
            c_size = serializer.serialize(src, c_buf)
            compressed_sizes[i] = c_size

            c_obj = create_memory_obj(size=c_size)
            c_obj.raw_data[:] = c_buf.raw_data[:c_size]

            adapter.submit_store_task([create_object_key(i)], [c_obj])
            wait_for_event_fd(store_fd)

        # Load and verify each
        for i in range(num_keys):
            loaded = create_memory_obj(size=compressed_sizes[i])
            adapter.submit_load_task([create_object_key(i)], [loaded])
            wait_for_event_fd(load_fd)

            output = create_memory_obj(size=16384)
            deserializer.deserialize(loaded, output)
            assert bytes(output.byte_array) == originals[i], f"Key {i} mismatch"

    def test_compression_ratio_report(self, qat_backend, adapter):
        """Print compression stats for visibility (always passes)."""
        serializer = AccelCompressSerializer(
            backend=qat_backend,
            byte_reorder=True,
            truncate_bits=2,
            element_size=2,
        )

        # Simulate realistic bf16 KV data
        kv = torch.randn(64, 128, dtype=torch.bfloat16)
        raw_bytes = kv.view(torch.uint8).numpy().tobytes()
        size = len(raw_bytes)

        src = create_memory_obj(size=size)
        src.raw_data[:size] = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)

        max_c = qat_backend.max_compressed_length(size)
        c_buf = create_memory_obj(size=max_c)
        c_size = serializer.serialize(src, c_buf)

        ratio = size / c_size if c_size > 0 else float("inf")
        print(
            f"\n  QAT compression: {size} -> {c_size} bytes "
            f"({ratio:.2f}x, {100 * (1 - c_size / size):.1f}% saved)"
        )
        assert c_size > 0

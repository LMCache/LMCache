# SPDX-License-Identifier: Apache-2.0
"""
Integration test: AccelCompressSerializer/Deserializer + DramL2Adapter.

Uses a zlib-based mock backend (no QAT hardware required).
Validates the full compress → store → load → decompress roundtrip.
"""

# Standard
import select
import zlib

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
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

# =============================================================================
# Mock zlib backend (no QAT hardware needed)
# =============================================================================


class ZlibBackend(AccelCompressBackend):
    """zlib-based mock that satisfies AccelCompressBackend interface."""

    def compress(self, src: memoryview, dst: memoryview) -> int:
        compressed = zlib.compress(bytes(src), level=1)
        dst_cast = dst.cast("B") if dst.format != "B" else dst
        dst_cast[: len(compressed)] = compressed
        return len(compressed)

    def decompress(self, src: memoryview, dst: memoryview) -> int:
        decompressed = zlib.decompress(bytes(src))
        dst_cast = dst.cast("B") if dst.format != "B" else dst
        dst_cast[: len(decompressed)] = decompressed
        return len(decompressed)

    def max_compressed_length(self, src_size: int) -> int:
        # zlib worst case: src_size + ~0.1% + 12 bytes header
        return src_size + (src_size // 100) + 64

    def close(self) -> None:
        pass


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
    """Create a TensorMemoryObj with uint8 data."""
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


def wait_for_event_fd(event_fd: int, timeout: float = 2.0) -> bool:
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
# Tests
# =============================================================================


_DUMMY_KEY = ObjectKey(chunk_hash=b"test", model_name="m", kv_rank=0)


class TestSerdeRoundtrip:
    """Test Serializer/Deserializer with mock zlib backend directly."""

    def test_compress_decompress_identity(self):
        """Data survives compress→decompress (no preprocessing)."""
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(
            backend=backend, byte_reorder=False, truncate_bits=0
        )
        deserializer = AccelCompressDeserializer(backend=backend, byte_reorder=False)

        # Source: random-ish KV data
        src = create_memory_obj(size=4096, fill_value=0)
        src.raw_data[:] = torch.randint(0, 256, (4096,), dtype=torch.uint8)
        original = bytes(src.byte_array)

        # Compress: src → compressed_buf
        max_compressed = backend.max_compressed_length(4096)
        compressed_buf = create_memory_obj(size=max_compressed)
        compressed_size = serializer.serialize(src, compressed_buf, _DUMMY_KEY)
        assert 0 < compressed_size <= max_compressed

        # Prepare a compressed MemoryObj with exact size for decompress
        compressed_exact = create_memory_obj(size=compressed_size)
        compressed_exact.raw_data[:] = compressed_buf.raw_data[:compressed_size]

        # Decompress: compressed_exact → output
        output = create_memory_obj(size=4096)
        deserializer.deserialize(compressed_exact, output, _DUMMY_KEY)

        assert bytes(output.byte_array) == original

    def test_compress_decompress_with_shuffle(self):
        """Data survives compress→decompress with byte_reorder enabled."""
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(
            backend=backend, byte_reorder=True, truncate_bits=0, element_size=2
        )
        deserializer = AccelCompressDeserializer(
            backend=backend, byte_reorder=True, element_size=2
        )

        # Use even size for 2-byte element shuffle
        src = create_memory_obj(size=4096)
        src.raw_data[:] = torch.randint(0, 256, (4096,), dtype=torch.uint8)
        original = bytes(src.byte_array)

        max_compressed = backend.max_compressed_length(4096)
        compressed_buf = create_memory_obj(size=max_compressed)
        compressed_size = serializer.serialize(src, compressed_buf, _DUMMY_KEY)

        compressed_exact = create_memory_obj(size=compressed_size)
        compressed_exact.raw_data[:] = compressed_buf.raw_data[:compressed_size]

        output = create_memory_obj(size=4096)
        deserializer.deserialize(compressed_exact, output, _DUMMY_KEY)

        assert bytes(output.byte_array) == original

    def test_truncate_is_lossy(self):
        """quant_trunc zeros LSBs so result won't be bit-identical."""
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(
            backend=backend, byte_reorder=False, truncate_bits=4, element_size=1
        )
        deserializer = AccelCompressDeserializer(backend=backend, byte_reorder=False)

        src = create_memory_obj(size=1024)
        src.raw_data[:] = torch.randint(0, 256, (1024,), dtype=torch.uint8)
        _original = bytes(src.byte_array)  # noqa: F841

        max_compressed = backend.max_compressed_length(1024)
        compressed_buf = create_memory_obj(size=max_compressed)
        compressed_size = serializer.serialize(src, compressed_buf, _DUMMY_KEY)

        compressed_exact = create_memory_obj(size=compressed_size)
        compressed_exact.raw_data[:] = compressed_buf.raw_data[:compressed_size]

        output = create_memory_obj(size=1024)
        deserializer.deserialize(compressed_exact, output, _DUMMY_KEY)

        # Truncation is lossy — output should differ from original
        # but each byte should have lower 4 bits zeroed
        out_bytes = bytes(output.byte_array)
        for b in out_bytes:
            assert b & 0x0F == 0, f"Expected lower 4 bits zeroed, got {b:#04x}"


class TestSerdeWithDramL2:
    """End-to-end: serialize → DramL2Adapter.store → load → deserialize."""

    def test_full_flow(self):
        """Full pipeline: compress, store in DramL2, load, decompress."""
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(
            backend=backend, byte_reorder=False, truncate_bits=0
        )
        deserializer = AccelCompressDeserializer(backend=backend, byte_reorder=False)

        # Setup DramL2Adapter
        config = DramL2AdapterConfig(max_size_gb=0.001)
        adapter = DramL2Adapter(config)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        try:
            # Create source KV data
            src = create_memory_obj(size=8192)
            src.raw_data[:] = torch.randint(0, 256, (8192,), dtype=torch.uint8)
            original = bytes(src.byte_array)

            # Step 1: Serialize (compress)
            max_compressed = backend.max_compressed_length(8192)
            compressed_buf = create_memory_obj(size=max_compressed)
            compressed_size = serializer.serialize(src, compressed_buf, _DUMMY_KEY)

            # Create exact-size MemoryObj for store
            compressed_obj = create_memory_obj(size=compressed_size)
            compressed_obj.raw_data[:] = compressed_buf.raw_data[:compressed_size]

            # Step 2: Store compressed in DramL2
            key = create_object_key(42)
            task_id = adapter.submit_store_task([key], [compressed_obj])
            assert wait_for_event_fd(store_fd)
            results = adapter.pop_completed_store_tasks()
            assert results[task_id].is_successful()
            assert results[task_id].bytes_transferred() == compressed_size

            # Step 3: Load from DramL2
            loaded_buf = create_memory_obj(size=compressed_size)
            load_task = adapter.submit_load_task([key], [loaded_buf])
            assert wait_for_event_fd(load_fd)
            bitmap = adapter.query_load_result(load_task)
            assert bitmap.test(0) is True

            # Verify loaded matches what we stored
            assert bytes(loaded_buf.byte_array) == bytes(compressed_obj.byte_array)

            # Step 4: Deserialize (decompress)
            output = create_memory_obj(size=8192)
            deserializer.deserialize(loaded_buf, output, _DUMMY_KEY)

            # Verify roundtrip
            assert bytes(output.byte_array) == original

        finally:
            adapter.close()
            backend.close()

    def test_multiple_keys_flow(self):
        """Multiple keys stored/loaded independently."""
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(
            backend=backend, byte_reorder=False, truncate_bits=0
        )
        deserializer = AccelCompressDeserializer(backend=backend, byte_reorder=False)

        config = DramL2AdapterConfig(max_size_gb=0.01)
        adapter = DramL2Adapter(config)
        store_fd = adapter.get_store_event_fd()
        load_fd = adapter.get_load_event_fd()

        try:
            originals = {}
            compressed_sizes = {}

            # Store 5 different keys
            for i in range(5):
                src = create_memory_obj(size=2048)
                src.raw_data[:] = torch.randint(0, 256, (2048,), dtype=torch.uint8)
                originals[i] = bytes(src.byte_array)

                max_c = backend.max_compressed_length(2048)
                c_buf = create_memory_obj(size=max_c)
                c_size = serializer.serialize(src, c_buf, _DUMMY_KEY)
                compressed_sizes[i] = c_size

                c_obj = create_memory_obj(size=c_size)
                c_obj.raw_data[:] = c_buf.raw_data[:c_size]

                key = create_object_key(i)
                adapter.submit_store_task([key], [c_obj])
                wait_for_event_fd(store_fd)

            # Load and verify each
            for i in range(5):
                key = create_object_key(i)
                loaded = create_memory_obj(size=compressed_sizes[i])
                adapter.submit_load_task([key], [loaded])
                wait_for_event_fd(load_fd)

                output = create_memory_obj(size=2048)
                deserializer.deserialize(loaded, output, _DUMMY_KEY)
                assert bytes(output.byte_array) == originals[i], f"Key {i} mismatch"

        finally:
            adapter.close()
            backend.close()

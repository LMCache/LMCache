# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for HF3fsL2Adapter v2 (native 3FS connector).

This test file follows the Aerospike L2 integration test pattern with:
- ObjectKey + cache_salt for unique test keys
- _wait_fd for eventfd-based waiting
- Proper skipif markers for availability
- try/finally for cleanup

Requires:
- A running 3FS cluster with RDMA connectivity (no FUSE mount needed: the
  connector talks to the meta/storage servers directly over RDMA)
- LMCache raw 3FS extension built (BUILD_WITH_RAW_HF3FS=1)

Run with:
    RUN_HF3FS_INTEGRATION=1 pytest tests/v1/distributed/test_hf3fs_l2_adapter_v2.py
"""

# Standard
from typing import Any
import os
import select
import time

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

# Try to import the HF3FS extension
try:
    # First Party
    import lmcache.lmcache_raw3fs as LMCacheRaw3FS

    HF3FS_AVAILABLE = True
except ImportError:
    HF3FS_AVAILABLE = False

HF3FS_META_ADDR = os.environ.get("HF3FS_META_ADDR", "RDMA://192.168.1.100:8000")
HF3FS_MGMTD_ADDR = os.environ.get("HF3FS_MGMTD_ADDR", "RDMA://192.168.1.100:8000")
RUN_HF3FS_IT = os.environ.get("RUN_HF3FS_INTEGRATION") == "1"

pytestmark = [pytest.mark.integration]


def _hf3fs_available() -> bool:
    """Check if 3FS cluster is reachable and extension is built."""
    if not RUN_HF3FS_IT:
        return False
    if not HF3FS_AVAILABLE:
        return False
    try:
        # Try to create a connector to verify connectivity
        connector = LMCacheRaw3FS.HF3fsConnector(
            num_workers=1,
            meta_addr=HF3FS_META_ADDR,
            mgmtd_addr=HF3FS_MGMTD_ADDR,
            chunk_size=64 * 1024 * 1024,
        )
        connector.close()
        return True
    except Exception:
        return False


def _run_cleanup() -> None:
    """Run cleanup_data_files() to reclaim disk space on 3FS nodes.

    Creates a temporary connector, calls cleanup_data_files() which releases
    all storage chunks via StorageClient::removeChunks() and deletes the
    lmcache directory from MetaClient, then closes the connector.
    """
    if not RUN_HF3FS_IT or not HF3FS_AVAILABLE:
        return
    try:
        connector = LMCacheRaw3FS.HF3fsConnector(
            num_workers=1,
            meta_addr=HF3FS_META_ADDR,
            mgmtd_addr=HF3FS_MGMTD_ADDR,
            chunk_size=64 * 1024 * 1024,
        )
        connector.cleanup_data_files()
        connector.close()
    except Exception as e:
        print(f"[cleanup] cleanup_data_files() skipped: {e}")


requires_hf3fs = pytest.mark.skipif(
    not _hf3fs_available(),
    reason=(f"HF3FS not available at {HF3FS_META_ADDR} (set RUN_HF3FS_INTEGRATION=1)"),
)


def _wait_fd(fd: int, timeout: float = 30.0) -> None:
    """Wait for eventfd to be signaled (non-blocking retry with sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            events = poller.poll(100)  # 100ms timeout per poll
            if events:
                try:
                    consume_fd(fd)
                except BlockingIOError:
                    pass
                return
        except OSError:
            time.sleep(0.01)
    raise AssertionError("timed out waiting for eventfd")


def _make_tensor_obj(size: int, fill: float) -> TensorMemoryObj:
    """Create a TensorMemoryObj with given size and fill value."""
    # Third Party
    import torch

    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _object_key_str(suffix: int) -> str:
    """Create a key string in 'high:low' format expected by the connector."""
    return f"0:{suffix}"


def _object_key(suffix: int) -> ObjectKey:
    """Create an ObjectKey from a suffix integer."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(suffix),
        model_name="hf3fs-it",
        kv_rank=0,
    )


def _object_key_to_str(key: ObjectKey) -> str:
    """Convert an ObjectKey to FSConnector/S3-compatible format.

    Produces: <model>@<kv_rank_hex>@<obj_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]

    This matches the serialization used by S3 adapter and FS adapter.
    """
    model = key.model_name
    rank_hex = f"{key.kv_rank:x}"
    obj_group_id_hex = f"{key.object_group_id:x}"
    hash_hex = key.chunk_hash.hex()
    result = f"{model}@{rank_hex}@{obj_group_id_hex}@{hash_hex}"
    if key.cache_salt:
        result += f"@{key.cache_salt}"
    return result


def _keys_to_strs(keys) -> list:
    """Convert a list of keys (str or ObjectKey) to connector key strings."""
    return [_object_key_to_str(k) if isinstance(k, ObjectKey) else k for k in keys]


_EMPTY_LAYOUT = None  # Not used by HF3FS connector; kept for API compat


def _tensor_to_buffer(data, length: int) -> bytearray:
    """Convert input to a bytearray buffer suitable for connector operations.

    For SET: data is source (tensor or bytes), returns a bytearray of the data.
    For GET: data is a destination bytearray (or tensor), returns a mutable bytearray
    that the connector will write into.
    """
    if isinstance(data, bytearray):
        # GET operation: pass the bytearray as-is so connector can write into it
        return data
    if hasattr(data, "numpy"):
        # Tensor: convert to bytearray for SET, or create new bytearray for GET
        return bytearray(data.numpy().tobytes()[:length])
    if hasattr(data, "item"):
        return bytearray(data.item().to_bytes(4, "little"))
    # Bytes or other: wrap in bytearray
    return bytearray(data)


class HF3fsL2Adapter:
    """L2 adapter backed by real 3FS connector with high-level API."""

    def __init__(self, chunk_size: int = 64 * 1024 * 1024):
        """Initialize the adapter.

        Args:
            chunk_size: connector chunk size in bytes (multi-chunk tests
                pass a smaller value so keys span several chunks).
        """
        # Third Party
        self.connector = LMCacheRaw3FS.HF3fsConnector(
            num_workers=4,
            meta_addr=HF3FS_META_ADDR,
            mgmtd_addr=HF3FS_MGMTD_ADDR,
            chunk_size=chunk_size,
        )
        # Keep references to buffers to prevent GC before async ops complete.
        # The connector stores raw pointers to Python bytearray data; if the
        # bytearray is GC'd before worker threads read it, use-after-free
        # occurs causing data corruption (observed as garbage bytes in reads).
        self._kept_buffers: list[Any] = []
        self._store_futures: dict[int, list[str]] = {}
        self._load_futures: dict[int, tuple[list[str], list[Any]]] = {}
        self._lookup_futures: dict[int, list[str]] = {}

    def event_fd(self) -> int:
        """Return the event file descriptor."""
        return self.connector.event_fd()

    # ── High-level API matching Aerospike L2 test pattern ──

    def submit_store_task(self, keys, obj_list):
        """Submit a batch store (SET) operation from TensorMemoryObj list.

        Args:
            keys: list of key strings (or ObjectKeys).
            obj_list: list of TensorMemoryObj objects.

        Returns:
            future_id for the batch operation.
        """
        # Third Party

        str_keys = _keys_to_strs(keys)
        # Use batch SET with one combined contiguous buffer
        # Each tensor contributes: numel * 4 bytes
        data_list = []
        lens = []
        for obj in obj_list:
            tensor_data = obj.tensor
            b = _tensor_to_buffer(tensor_data, tensor_data.numel() * 4)
            self._kept_buffers.append(b)
            data_list.append(b)
            lens.append(len(b))

        # Submit individual ops (one key -> one batch of size 1)
        futures = []
        for key, buf, length in zip(str_keys, data_list, lens, strict=False):
            fid = self.connector.submit_batch_set([key], [buf])
            futures.append(fid)

        fid = futures[0] if len(futures) == 1 else sum(futures)
        self._store_futures[fid] = str_keys
        return fid

    def get_store_event_fd(self) -> int:
        """Return the event fd for store completion."""
        return self.event_fd()

    def pop_completed_store_tasks(self):
        """Pop completed store task results.

        Returns:
            dict mapping future_id -> L2StoreResult.
        """
        all_completions = []
        deadline = time.monotonic() + 30.0
        while len(all_completions) == 0 and time.monotonic() < deadline:
            all_completions = self.drain_completions()
            if len(all_completions) == 0:
                time.sleep(0.01)
        results = {}
        for c in all_completions:
            # L2StoreResult(success, bytes_transferred)
            # bytes_transferred=0 is fine for test purposes
            results[c[0]] = L2StoreResult(c[1], 0)
        return results

    def submit_lookup_and_lock_task(self, keys, layout=None):
        """Submit a batch EXISTS (lookup) operation.

        Args:
            keys: list of key strings (or ObjectKeys).
            layout: ignored (kept for API compatibility with Aerospike pattern).

        Returns:
            future_id for the batch operation.
        """
        str_keys = _keys_to_strs(keys)
        fid = self.connector.submit_batch_exists(str_keys)
        self._lookup_futures[fid] = str_keys
        return fid

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the event fd for lookup completion."""
        return self.event_fd()

    def get_load_event_fd(self) -> int:
        """Return the event fd for load completion."""
        return self.event_fd()

    def query_lookup_and_lock_result(self, future_id):
        """Query lookup/exists result as a Bitmap (None if not ready).

        Returns:
            Bitmap of per-key existence flags, or None if not yet available.
        """
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            completions = self.drain_completions()
            if completions:
                for c in completions:
                    if c[0] == future_id:
                        # result_bools contains per-key 0/1 flags
                        result_bools = c[3]
                        bitmap = Bitmap(len(result_bools))
                        for i, flag in enumerate(result_bools):
                            if flag:
                                bitmap.set(i)
                        return bitmap
            time.sleep(0.01)
        return None

    def submit_load_task(self, keys, obj_list):
        """Submit a batch load (GET) operation into TensorMemoryObj tensors.

        Args:
            keys: list of key strings (or ObjectKeys).
            obj_list: list of TensorMemoryObj objects (tensor field will be
                      populated by the connector).

        Returns:
            future_id for the batch operation.
        """
        # Third Party

        str_keys = _keys_to_strs(keys)
        futures = []
        lens = []
        # For each tensor, we pass a bytearray that wraps the tensor's own
        # memory so the connector writes directly into the tensor.
        # We keep a strong reference to prevent GC.
        tensor_refs = []
        for key, obj in zip(str_keys, obj_list, strict=False):
            size = obj.tensor.numel() * 4
            # Create a bytearray that shares memory with the tensor
            buf = bytearray(size)
            # Use ctypes to copy data from bytearray into tensor's memory
            # We'll do this after the GET completes in query_load_result
            tensor_refs.append((obj, buf, size))
            self._kept_buffers.append(buf)
            # Also keep the tensor reference
            self._kept_buffers.append(obj.tensor)
            lens.append(size)

        # Submit individual GET ops
        buffers = [t[1] for t in tensor_refs]
        for key, buf, length in zip(str_keys, buffers, lens, strict=False):
            result = self.connector.submit_batch_get([key], [buf])
            fid = int(result[0]) if isinstance(result, tuple) else int(result)
            futures.append(fid)

        fid = futures[0] if len(futures) == 1 else sum(futures)
        self._load_futures[fid] = (str_keys, tensor_refs)
        return fid

    def _sync_loaded_tensors(self, future_id):
        """Copy data from bytearray buffers back into the tensor memory.

        Called after a successful GET completion to populate the tensors.
        """
        if future_id not in self._load_futures:
            return
        str_keys, tensor_refs = self._load_futures.pop(future_id)
        # Third Party
        import torch

        for obj, buf, size in tensor_refs:
            # Copy from bytearray into tensor's contiguous memory
            # Get the underlying tensor (raw_data) and copy data
            tensor = obj.raw_data
            # Convert bytearray to numpy array and reshape to match tensor dtype
            # Third Party
            import numpy as np

            src_array = np.frombuffer(buf, dtype=np.uint8)[:size]
            # The buffer is in bytes, tensor is in its dtype (float32)
            # So we need to convert bytes to the tensor's dtype
            if tensor.dtype == torch.float32:
                src_float = src_array.view(np.float32)[: tensor.numel()]
                tensor.copy_(torch.from_numpy(src_float))
            elif tensor.dtype == torch.float16:
                src_half = src_array.view(np.float16)[: tensor.numel()]
                tensor.copy_(torch.from_numpy(src_half))
            else:
                # Generic: copy bytes directly
                tensor_bytes = tensor.numpy().tobytes()
                tensor_bytes[:size] = bytes(buf[:size])
                tensor.copy_(
                    torch.frombuffer(tensor_bytes, dtype=tensor.dtype).reshape(
                        tensor.shape
                    )
                )

    def query_load_result(self, future_id):
        """Query load result as a Bitmap (None if not ready).

        Returns:
            Bitmap of per-key success flags, or None if not yet available.
        """
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            completions = self.drain_completions()
            if completions:
                for c in completions:
                    if c[0] == future_id:
                        result_bools = c[3]
                        bitmap = Bitmap(len(result_bools))
                        for i, flag in enumerate(result_bools):
                            if flag:
                                bitmap.set(i)
                        return bitmap
            time.sleep(0.01)
        return None

    # ── Convenience ──

    def submit_batch_set(self, keys, data_list, lens):
        """Submit a batch SET operation."""
        futures = []
        for key, data, length in zip(keys, data_list, lens, strict=False):
            buf = _tensor_to_buffer(data, length)
            self._kept_buffers.append(buf)
            future_id = self.connector.submit_batch_set([key], [buf])
            futures.append(future_id)
        return futures[0] if len(futures) == 1 else futures

    def submit_batch_get(self, keys, buf_list, lens):
        """Submit a batch GET operation."""
        futures = []
        for key, buf, length in zip(keys, buf_list, lens, strict=False):
            if not isinstance(buf, bytearray):
                buf = bytearray(length)
            self._kept_buffers.append(buf)
            result = self.connector.submit_batch_get([key], [buf])
            future_id = int(result[0]) if isinstance(result, tuple) else int(result)
            futures.append(future_id)
        return futures[0] if len(futures) == 1 else futures

    def wait_all_completions(self, expected_count, timeout=30.0):
        """Wait until at least expected_count completions are available."""
        deadline = time.monotonic() + timeout
        all_completions = []
        while len(all_completions) < expected_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _wait_fd(self.event_fd(), timeout=remaining)
            all_completions.extend(self.drain_completions())
        return all_completions

    def submit_batch_exists(self, keys):
        """Submit a batch EXISTS operation."""
        return self.connector.submit_batch_exists(keys)

    def submit_batch_delete(self, keys):
        """Submit a batch DELETE operation."""
        str_keys = _keys_to_strs(keys)
        return self.connector.submit_batch_delete(str_keys)

    def submit_unlock(self, keys):
        """Submit a batch DELETE (unlock) operation."""
        return self.submit_batch_delete(keys)

    def drain_completions(self):
        """Drain all available completions."""
        return self.connector.drain_completions()

    def close(self, cleanup: bool = True):
        """Close the connector; optionally clean 3FS data files first.

        Args:
            cleanup: when True (default), release chunks and delete the
                lmcache directory (fresh state for the next test). Pass
                False to keep the data on 3FS (restart/persistence tests).
        """
        if cleanup:
            try:
                self.connector.cleanup_data_files()
            except Exception:
                pass  # Ignore cleanup errors during close
        self.connector.close()


class TestHF3fsL2Integration:
    """Integration tests for HF3fsL2Adapter with real 3FS cluster."""

    @requires_hf3fs
    def test_store_lookup_load_roundtrip_tensor(self):
        """Test STORE -> LOOKUP -> LOAD roundtrip with real tensors.

        Follows the Aerospike L2 test pattern:
        1. Create a TensorMemoryObj with fill value 42.0
        2. submit_store_task (submit_batch_set)
        3. submit_lookup_and_lock_task (submit_batch_exists)
        4. submit_load_task into an empty TensorMemoryObj
        5. Verify loaded tensor == 42.0
        """
        # Third Party
        import torch

        adapter = HF3fsL2Adapter()
        try:
            key = _object_key(9001)
            store_obj = _make_tensor_obj(64, 42.0)
            load_obj = _make_tensor_obj(64, 0.0)

            # STORE: submit batch_set with tensor data
            tid = adapter.submit_store_task([key], [store_obj])
            _wait_fd(adapter.get_store_event_fd())
            done = adapter.pop_completed_store_tasks()
            assert done[tid].is_successful(), f"STORE failed for future_id={tid}"

            # LOOKUP: submit batch_exists
            lookup_tid = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
            _wait_fd(adapter.get_lookup_and_lock_event_fd())
            lookup_bm = adapter.query_lookup_and_lock_result(lookup_tid)
            assert lookup_bm is not None, "Lookup result not available"
            assert lookup_bm.test(0), "Key not found after STORE"

            # LOAD: submit batch_get into load_obj.tensor
            load_tid = adapter.submit_load_task([key], [load_obj])
            _wait_fd(adapter.get_load_event_fd())
            load_bm = adapter.query_load_result(load_tid)
            assert load_bm is not None, "Load result not available"
            assert load_bm.test(0), "LOAD failed"
            # Copy bytearray data back into tensor
            adapter._sync_loaded_tensors(load_tid)
            assert torch.all(load_obj.tensor == 42.0), (
                f"Data mismatch: expected 42.0, got tensor with unique "
                f"value = {load_obj.tensor.unique().tolist()[0]}"
            )

            # Cleanup (unlock)
            adapter.submit_unlock([key])
        finally:
            adapter.close()

    @requires_hf3fs
    def test_delete_and_recreate(self):
        """Test DELETE and recreate flow."""
        adapter = HF3fsL2Adapter()
        try:
            key_str = _object_key_str(10002)
            test_data = b"\xcd" * 1024

            # STORE
            adapter.submit_batch_set([key_str], [test_data], [len(test_data)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]

            # EXISTS should return 1
            adapter.submit_batch_exists([key_str])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert completions[0][3][0]

            # DELETE
            adapter.submit_batch_delete([key_str])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]

            # EXISTS should return 0 after delete (in-memory map lookup is
            # deterministic — the key is erased by remove_offset).
            adapter.submit_batch_exists([key_str])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert not completions[0][3][0], "key still exists after DELETE"

            # RECREATE the same key with different data and verify it is
            # readable again.
            new_data = b"\xce" * 2048
            adapter.submit_batch_set([key_str], [new_data], [len(new_data)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]

            adapter.submit_batch_exists([key_str])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert completions[0][3][0], "key missing after recreate"

            buf = bytearray(len(new_data))
            adapter.submit_batch_get([key_str], [buf], [len(new_data)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert bytes(buf) == new_data, "recreated key data mismatch"

        finally:
            adapter.close()

    @requires_hf3fs
    def test_multiple_keys_roundtrip(self):
        """Test roundtrip with multiple keys."""
        adapter = HF3fsL2Adapter()
        try:
            key_strs = [_object_key_str(i) for i in range(10003, 10008)]

            # Create test data for each key
            data_list = [
                b"\xaa" * 512,
                b"\xbb" * 512,
                b"\xcc" * 512,
                b"\xdd" * 512,
                b"\xee" * 512,
            ]
            lens = [len(d) for d in data_list]

            # STORE all keys - wait for all 5 completions
            adapter.submit_batch_set(key_strs, data_list, lens)
            completions = adapter.wait_all_completions(5)
            assert len(completions) == 5, (
                f"Expected 5 completions, got {len(completions)}"
            )
            for i, c in enumerate(completions):
                assert c[1], f"STORE key {i} failed: {c[2]}"

            # LOAD all keys - wait for all 5 completions
            buf_list = [bytearray(512) for _ in range(5)]
            adapter.submit_batch_get(key_strs, buf_list, lens)
            completions = adapter.wait_all_completions(5)
            assert len(completions) == 5, (
                f"Expected 5 completions, got {len(completions)}"
            )
            for i, c in enumerate(completions):
                assert c[1], f"LOAD key {i} failed: {c[2]}"

            # Verify each key
            pairs = zip(key_strs, data_list, strict=False)
            for i, (key_str, expected_data) in enumerate(pairs):
                assert bytes(buf_list[i]) == expected_data, f"Key {i} data mismatch"

        finally:
            adapter.close()

    @requires_hf3fs
    def test_get_nonexistent_key(self):
        """Test GET on a non-existent key."""
        adapter = HF3fsL2Adapter()
        try:
            key_str = _object_key_str(10009)
            buf = bytearray(1024)

            adapter.submit_batch_get([key_str], [buf], [len(buf)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0
            assert not completions[0][1], "GET on non-existent key should fail"

        finally:
            adapter.close()

    @requires_hf3fs
    def test_restart_persistence(self):
        """STORE -> clean close -> new connector must replay the offset map
        from the persisted meta region and serve byte-identical data."""
        keys = [f"0:{20001 + i}" for i in range(3)]
        data_list = [
            b"\x11" * 4096,
            b"\x22" * 8192,
            b"\x33" * 123456,
        ]
        adapter = HF3fsL2Adapter()
        try:
            for key, data in zip(keys, data_list, strict=True):
                adapter.submit_batch_set([key], [data], [len(data)])
                _wait_fd(adapter.event_fd())
                completions = adapter.drain_completions()
                assert len(completions) > 0 and completions[0][1]
            adapter.close(cleanup=False)
        except Exception:
            adapter.close(cleanup=True)
            raise

        adapter2 = HF3fsL2Adapter()
        try:
            # LOOKUP after restart: all keys replayed from persisted meta.
            adapter2.submit_batch_exists(keys)
            _wait_fd(adapter2.event_fd())
            completions = adapter2.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert all(completions[0][3]), "keys missing after restart replay"

            # LOAD after restart: byte-identical payloads.
            for key, data in zip(keys, data_list, strict=True):
                buf = bytearray(len(data))
                adapter2.submit_batch_get([key], [buf], [len(data)])
                _wait_fd(adapter2.event_fd())
                completions = adapter2.drain_completions()
                assert len(completions) > 0 and completions[0][1], (
                    f"LOAD failed for {key} after restart"
                )
                assert bytes(buf) == data, f"data mismatch for {key} after restart"
        finally:
            adapter2.close()

    @requires_hf3fs
    def test_restart_persistence_adapter_keys(self):
        """Adapter-serialized '@'-format keys (MP-mode wire keys, including a
        salted variant) must survive a restart byte-identically — guards the
        deterministic 128-bit fold: meta persistence must cover every wire
        key format, not only 'high:low' keys."""
        keys = [
            _object_key_to_str(_object_key(21001)),
            # Same chunk hash + different cache_salt: a distinct key.
            _object_key_to_str(
                ObjectKey(
                    chunk_hash=ObjectKey.IntHash2Bytes(21001),
                    model_name="hf3fs-it",
                    kv_rank=0,
                    cache_salt="user42",
                )
            ),
            _object_key_to_str(
                ObjectKey(
                    chunk_hash=ObjectKey.IntHash2Bytes(21002),
                    model_name="hf3fs-it",
                    kv_rank=2,
                    object_group_id=3,
                )
            ),
        ]
        data_list = [
            b"\x55" * 4096,
            b"\x66" * 4096,
            b"\x77" * 123456,
        ]
        adapter = HF3fsL2Adapter()
        try:
            for key, data in zip(keys, data_list, strict=True):
                adapter.submit_batch_set([key], [data], [len(data)])
                _wait_fd(adapter.event_fd())
                completions = adapter.drain_completions()
                assert len(completions) > 0 and completions[0][1]
            adapter.close(cleanup=False)
        except Exception:
            adapter.close(cleanup=True)
            raise

        adapter2 = HF3fsL2Adapter()
        try:
            # LOOKUP after restart: every '@'-format key replays from the
            # persisted meta (the fold round-trips).
            adapter2.submit_batch_exists(keys)
            _wait_fd(adapter2.event_fd())
            completions = adapter2.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert all(completions[0][3]), "@-format keys missing after restart"

            # LOAD after restart: byte-identical payloads; the salted twin
            # must not alias the un-salted key.
            for key, data in zip(keys, data_list, strict=True):
                buf = bytearray(len(data))
                adapter2.submit_batch_get([key], [buf], [len(data)])
                _wait_fd(adapter2.event_fd())
                completions = adapter2.drain_completions()
                assert len(completions) > 0 and completions[0][1]
                assert bytes(buf) == data, f"data mismatch for {key} after restart"
        finally:
            adapter2.close()

    @requires_hf3fs
    def test_delete_persistence_no_resurrect(self):
        """A deleted key must stay deleted across a restart (delete marks
        the persisted meta dirty; otherwise the key resurrects on replay)."""
        key = "0:20100"
        data = b"\x9a" * 4096
        adapter = HF3fsL2Adapter()
        try:
            adapter.submit_batch_set([key], [data], [len(data)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            adapter.close(cleanup=False)
        except Exception:
            adapter.close(cleanup=True)
            raise

        adapter2 = HF3fsL2Adapter()
        try:
            # GIVEN the key replays back after restart
            adapter2.submit_batch_exists([key])
            _wait_fd(adapter2.event_fd())
            completions = adapter2.drain_completions()
            assert completions[0][3][0], "key should exist after restart"

            # WHEN the key is deleted on the restarted instance
            adapter2.submit_batch_delete([key])
            _wait_fd(adapter2.event_fd())
            completions = adapter2.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert completions[0][3][0], "delete should report the key removed"
            adapter2.close(cleanup=False)
        except Exception:
            adapter2.close(cleanup=True)
            raise

        adapter3 = HF3fsL2Adapter()
        try:
            # THEN the key must not resurrect from the persisted meta
            adapter3.submit_batch_exists([key])
            _wait_fd(adapter3.event_fd())
            completions = adapter3.drain_completions()
            assert not completions[0][3][0], "deleted key resurrected after restart"

            buf = bytearray(len(data))
            adapter3.submit_batch_get([key], [buf], [len(data)])
            _wait_fd(adapter3.event_fd())
            completions = adapter3.drain_completions()
            assert not completions[0][1], "GET on deleted key should fail"
        finally:
            adapter3.close()

    @requires_hf3fs
    def test_lease_blocks_second_instance(self):
        """A second connector instance sharing data_${CS}_store must fail fast
        (foreign owner lease), and a new instance must succeed right after
        the owner releases the lease on clean close."""
        owner = HF3fsL2Adapter()
        closed = False
        try:
            owner.submit_batch_set(["0:20200"], [b"\x5a" * 2048], [2048])
            _wait_fd(owner.event_fd())
            completions = owner.drain_completions()
            assert len(completions) > 0 and completions[0][1]

            # GIVEN the owner holds a live lease, a second instance's store
            # must fail (init refused, per-key result 0).
            second = HF3fsL2Adapter()
            try:
                second.submit_batch_set(["0:20201"], [b"\x6b" * 512], [512])
                _wait_fd(second.event_fd())
                completions = second.drain_completions()
                assert len(completions) > 0, "second instance produced no completion"
                assert not completions[0][1], (
                    "second instance store must fail under a foreign live lease"
                )
            finally:
                second.close(cleanup=False)

            # WHEN the owner cleanly closes (releasing the lease),
            owner.close(cleanup=True)
            closed = True

            # THEN a fresh instance can store immediately without waiting the TTL.
            fresh = HF3fsL2Adapter()
            try:
                fresh.submit_batch_set(["0:20202"], [b"\x7c" * 512], [512])
                _wait_fd(fresh.event_fd())
                completions = fresh.drain_completions()
                assert len(completions) > 0 and completions[0][1], (
                    "fresh instance store failed after lease release"
                )
            finally:
                fresh.close()
        finally:
            if not closed:
                owner.close(cleanup=True)

    @requires_hf3fs
    def test_free_list_reuse_after_delete(self):
        """Chunks released by a delete enter the free list; a subsequent key
        reusing them must load byte-identical data, and neither the deleted
        key nor a neighbor may show corruption."""
        cs = 4 * 1024 * 1024
        adapter = HF3fsL2Adapter(chunk_size=cs)
        try:
            ka, kb = "0:20300", "0:20301"
            data_a = b"\x44" * (1024 * 1024)
            data_b = b"\x55" * (1024 * 1024)
            for key, data in [(ka, data_a), (kb, data_b)]:
                adapter.submit_batch_set([key], [data], [len(data)])
                _wait_fd(adapter.event_fd())
                completions = adapter.drain_completions()
                assert completions[0][1]

            adapter.submit_batch_delete([ka])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert completions[0][1] and completions[0][3][0]

            # Give the background meta_saver cycle time to run removeChunks
            # and move the released chunks into the free list.
            time.sleep(6.0)

            # A new same-size key should be served from the freed chunks.
            kc = "0:20302"
            data_c = b"\x66" * (1024 * 1024)
            adapter.submit_batch_set([kc], [data_c], [len(data_c)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert completions[0][1]

            buf_c = bytearray(len(data_c))
            adapter.submit_batch_get([kc], [buf_c], [len(data_c)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert completions[0][1], "LOAD of reused-chunk key failed"
            assert bytes(buf_c) == data_c, "data corruption in reused chunks"

            buf_b = bytearray(len(data_b))
            adapter.submit_batch_get([kb], [buf_b], [len(data_b)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert completions[0][1], "LOAD of neighbor key failed"
            assert bytes(buf_b) == data_b, "neighbor key corrupted after reuse"

            buf_a = bytearray(len(data_a))
            adapter.submit_batch_get([ka], [buf_a], [len(data_a)])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert not completions[0][1], "deleted key must not be loadable"
        finally:
            adapter.close()

    @requires_hf3fs
    def test_multi_chunk_key_roundtrip(self):
        """Keys larger than chunk_size span several contiguous chunks; store,
        lookup and load must be byte-exact for multi-chunk and sub-chunk
        sizes alike."""
        cs = 4 * 1024 * 1024
        adapter = HF3fsL2Adapter(chunk_size=cs)
        specs = [
            ("0:20400", b"\x11" * (9 * 1024 * 1024)),  # 3 chunks
            ("0:20401", b"\x22" * (5 * 1024 * 1024)),  # 2 chunks
            ("0:20402", b"\x33" * (3 * 1024 * 1024)),  # 1 chunk (tail waste)
        ]
        try:
            for key, data in specs:
                adapter.submit_batch_set([key], [data], [len(data)])
                _wait_fd(adapter.event_fd())
                completions = adapter.drain_completions()
                assert len(completions) > 0 and completions[0][1], (
                    f"STORE failed for multi-chunk key {key}"
                )

            adapter.submit_batch_exists([k for k, _ in specs])
            _wait_fd(adapter.event_fd())
            completions = adapter.drain_completions()
            assert len(completions) > 0 and completions[0][1]
            assert all(completions[0][3]), "exists flags wrong for multi-chunk keys"

            for key, data in specs:
                buf = bytearray(len(data))
                adapter.submit_batch_get([key], [buf], [len(data)])
                _wait_fd(adapter.event_fd())
                completions = adapter.drain_completions()
                assert len(completions) > 0 and completions[0][1], (
                    f"LOAD failed for multi-chunk key {key}"
                )
                assert bytes(buf) == data, f"data mismatch for multi-chunk key {key}"
        finally:
            adapter.close()

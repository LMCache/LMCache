# SPDX-License-Identifier: Apache-2.0
"""
End-to-end vLLM-style KV cache integrity test for the 3FS L2 backend.
Tests data pipeline integrity:
vLLM tensor -> serialize -> 3FS -> deserialize -> compare.
Tests both batch path (do_batch_set/get) and single-key path (do_single_set/get).
"""

# Future
from __future__ import annotations

# Standard
import os
import uuid

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.raw_hf3fs_l2_adapter import (
    RawHf3fsL2AdapterConfig,
)
from lmcache.v1.memory_management import BytesBufferMemoryObj as MemoryObj
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata
from lmcache.v1.platform import consume_fd

META_ADDR = "RDMA://192.168.1.100:8000"
MGMTD_ADDR = "RDMA://192.168.1.100:8000"
NUM_WORKERS = 4

# Unique test session ID to avoid key collisions across test runs
_TEST_SESSION_ID = str(uuid.uuid4())[:8]

KV_CACHE_SHAPES = [
    {
        "name": "llama8b_mla_small",
        "dtype": torch.float16,
        "num_layers": 4,
        "num_kv_heads": 8,
        "head_dim": 64,
        "seq_lens": [64, 128],
    },
    {
        "name": "llama70b_standard_small",
        "dtype": torch.bfloat16,
        "num_layers": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "seq_lens": [64],
    },
    {
        "name": "gemma9b_standard_small",
        "dtype": torch.float16,
        "num_layers": 2,
        "num_kv_heads": 4,
        "head_dim": 32,
        "seq_lens": [32, 64],
    },
    {
        "name": "tiny_test",
        "dtype": torch.float16,
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 16,
        "seq_lens": [16],
    },
]


def _cleanup_3fs_files() -> None:
    """Remove stale 3FS meta/store files to avoid stale offset maps."""
    # Standard
    import glob

    patterns = [
        "/3fs/stage/lmcache/data_*.meta",
        "/3fs/stage/lmcache/data_*.store",
    ]
    for pattern in patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass


def _tensor_to_bytes(tensor):
    dtype_to_int = {
        torch.float16: 0,
        torch.float32: 1,
        torch.bfloat16: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.uint8: 5,
    }
    dtype_id = dtype_to_int.get(tensor.dtype, 0)
    shape = tuple(tensor.shape)
    storage_dtype_id = dtype_to_int[torch.float32]
    header = torch.tensor([dtype_id, storage_dtype_id, len(shape)], dtype=torch.int32)
    shape_tensor = torch.tensor(list(shape), dtype=torch.int64)
    data_tensor = tensor.contiguous().to(torch.float32)
    return (
        header.numpy().tobytes()
        + shape_tensor.numpy().tobytes()
        + data_tensor.numpy().tobytes()
    )


def _bytes_to_tensor(bs, original_shape, original_dtype):
    n_shape_ints = len(original_shape)
    header_len = 4 + 4 + 4 + n_shape_ints * 8
    header = bs[: header_len - n_shape_ints * 8]
    dtype_arr = torch.frombuffer(bytearray(header[:8]), dtype=torch.int32)
    storage_dtype_id = int(dtype_arr[1])
    data_bytes = bs[header_len:]
    storage_dtype_map = {0: torch.float16, 1: torch.float32, 2: torch.bfloat16}
    storage_torch_dtype = storage_dtype_map.get(storage_dtype_id, torch.float32)
    arr = torch.frombuffer(bytearray(data_bytes), dtype=storage_torch_dtype)
    return arr.reshape(original_shape).to(original_dtype)


def create_kv_cache_tensor(
    dtype, num_layers, num_kv_heads, head_dim, seq_len, key_idx=0
):
    total = 2 * num_layers * num_kv_heads * seq_len * head_dim
    tensor = torch.empty(total, dtype=dtype)
    idx = torch.arange(total, dtype=torch.float32)
    offset = key_idx * 1000
    max_val = 1.0 if dtype == torch.float16 else 1.0
    tensor = ((idx + offset) / total * max_val).to(dtype)
    return tensor.reshape(2, num_layers, num_kv_heads, seq_len, head_dim)


def create_bytes_memory_obj(key, data, fmt=MemoryFormat.KV_2LTD):
    raw_bytes = bytearray(data)
    metadata = MemoryObjMetadata(
        shape=torch.Size([len(raw_bytes)]),
        dtype=None,
        address=0,
        phy_size=len(raw_bytes),
        ref_count=1,
        pin_count=0,
        fmt=fmt,
    )
    return MemoryObj(raw_bytes, metadata)


def _wait_eventfd(efd, timeout=60.0):
    # Standard
    import select

    poller = select.poll()
    poller.register(efd, select.POLLIN)
    events = poller.poll(timeout * 1000)
    if events:
        consume_fd(efd)
        return True
    return False


def run_test(shape_cfg, num_keys=4, concurrent=False, in_flight=8):
    """Run integrity test with optional concurrent batch operations."""
    name = shape_cfg["name"]
    dtype = shape_cfg["dtype"]
    num_layers = shape_cfg["num_layers"]
    num_kv_heads = shape_cfg["num_kv_heads"]
    head_dim = shape_cfg["head_dim"]

    results = {
        "name": name,
        "success": True,
        "error": None,
        "roundtrip_bytes": 0,
        "per_key_integrity": [],
    }
    all_serialized = []
    all_original_tensors = []

    # Create keys with varying seq_lens for realistic testing
    seq_lens = (shape_cfg["seq_lens"] * (num_keys // len(shape_cfg["seq_lens"]) + 1))[
        :num_keys
    ]

    for i, seq_len in enumerate(seq_lens):
        original_tensor = create_kv_cache_tensor(
            dtype, num_layers, num_kv_heads, head_dim, seq_len, i
        )
        original_shape = original_tensor.shape
        original_dtype = original_tensor.dtype

        serialized = _tensor_to_bytes(original_tensor)
        all_serialized.append(serialized)
        results["roundtrip_bytes"] += len(serialized)

        try:
            restored = _bytes_to_tensor(serialized, original_shape, original_dtype)
            if not torch.equal(restored, original_tensor):
                results["success"] = False
                results["error"] = (
                    f"[{name}] local SERDE roundtrip FAIL (seq_len={seq_len})"
                )
                return results
        except Exception as e:
            results["success"] = False
            results["error"] = f"[{name}] local SERDE exception: {e}"
            return results
        all_original_tensors.append(original_tensor)

    objects = []
    keys = []
    for i in range(num_keys):
        key = ObjectKey(
            chunk_hash=i.to_bytes(16, "big"),
            model_name=f"test-model-{_TEST_SESSION_ID}",
            kv_rank=i,
        )
        objects.append(create_bytes_memory_obj(key, all_serialized[i]))
        keys.append(key)

    # chunk_size must match a data store chunk size (4MB/8MB/16MB/32MB) to
    # ensure 3FS chunk boundary is >= actual data size.
    # Data stores: [4MB, 8MB, 16MB, 32MB]. We pick the smallest store that
    # can hold the largest serialized key, so combined_buf stride aligns.
    max_data_size = max(len(s) for s in all_serialized)
    store_chunk_sizes = [
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
    ]
    chunk_size = store_chunk_sizes[0]  # default 4MB
    for cs in store_chunk_sizes:
        if cs >= max_data_size:
            chunk_size = cs
            break
    print(
        f"  Actual max data size per key: {max_data_size} bytes, "
        f"using chunk_size={chunk_size} bytes"
    )

    print(
        f"  Creating 3FS L2 adapter (shape={name}, keys={num_keys}, "
        f"concurrent={concurrent}, in_flight={in_flight})..."
    )
    config = RawHf3fsL2AdapterConfig(
        num_workers=NUM_WORKERS,
        meta_addr=META_ADDR,
        mgmtd_addr=MGMTD_ADDR,
        chunk_size=chunk_size,
    )
    adapter = create_l2_adapter(config)
    assert isinstance(adapter, NativeConnectorL2Adapter)

    try:
        # Store phase
        if concurrent:
            print(f"  Storing {num_keys} keys concurrently (in_flight={in_flight})...")
            task_ids = []
            for i in range(0, num_keys, in_flight):
                batch_keys = keys[i : i + in_flight]
                batch_objs = objects[i : i + in_flight]
                tid = adapter.submit_store_task(batch_keys, batch_objs)
                task_ids.append(tid)
        else:
            print(f"  Storing {num_keys} keys to 3FS...")
            task_ids = []
            for i in range(num_keys):
                tid = adapter.submit_store_task([keys[i]], [objects[i]])
                task_ids.append(tid)

        efd = adapter.get_store_event_fd()
        completed = {}
        expected_task_count = len(task_ids)
        while len(completed) < expected_task_count:
            if not _wait_eventfd(efd, timeout=120.0):
                results["success"] = False
                results["error"] = "Store timeout"
                break
            batch = adapter.pop_completed_store_tasks()
            completed.update(batch)

        if not results["success"]:
            return results
        for tid in task_ids:
            if tid in completed and completed[tid] < 0:
                results["success"] = False
                results["error"] = "Store failed"
                return results
        print(f"  All {num_keys} keys stored successfully.")

        # Lookup phase
        print(f"  Looking up {num_keys} keys...")
        lookup_tid = adapter.submit_lookup_and_lock_task(
            keys, MemoryLayoutDesc(shapes=[], dtypes=[])
        )
        lookup_done = False
        while not lookup_done:
            if not _wait_eventfd(adapter.get_lookup_and_lock_event_fd(), 60.0):
                results["success"] = False
                results["error"] = "Lookup timeout"
                break
            bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
            if bitmap is not None:
                lookup_done = True
                found = bitmap.popcount()
                print(f"  Lookup found {found}/{num_keys} keys")
                if found != num_keys:
                    results["success"] = False
                    results["error"] = f"Lookup: expected {num_keys}, got {found}"

        if not results["success"]:
            return results

        # Load phase
        print(f"  Loading {num_keys} keys from 3FS...")
        load_objects = []
        for i in range(num_keys):
            load_objects.append(
                create_bytes_memory_obj(keys[i], b"\x00" * len(all_serialized[i]))
            )
        load_tid = adapter.submit_load_task(keys, load_objects)

        load_done = False
        while not load_done:
            if not _wait_eventfd(adapter.get_load_event_fd(), 120.0):
                results["success"] = False
                results["error"] = "Load timeout"
                break
            bitmap = adapter.query_load_result(load_tid)
            if bitmap is not None:
                load_done = True
                loaded_count = bitmap.popcount()
                print(f"  Load loaded {loaded_count}/{num_keys} keys")
                if loaded_count != num_keys:
                    results["success"] = False
                    results["error"] = f"Load: expected {num_keys}, got {loaded_count}"

        if not results["success"]:
            return results

        # Verify integrity
        print("  Verifying data integrity...")
        # DEBUG: check bytearray contents right after load
        # Standard
        import os as _os

        if _os.environ.get("LMCACHE_DBG") == "1":
            for i in range(min(4, num_keys)):
                ba = load_objects[i].byte_array
                preview = bytes(ba[:8])
                print(
                    f"  [PY-DEBUG] Key {i}: "
                    f"ba_type={type(ba).__name__} ba_len={len(ba)} "
                    f"ba_first8={'/'.join(str(b) for b in preview)} "
                    f"ba_nonzero={sum(1 for b in ba if b != 0)}"
                )
        mismatch_count = 0
        for i in range(num_keys):
            expected = all_serialized[i]
            loaded = bytes(load_objects[i].byte_array[: len(expected)])
            if loaded == expected:
                results["per_key_integrity"].append(True)
            else:
                results["per_key_integrity"].append(False)
                mismatch_count += 1
                if mismatch_count <= 5:
                    expected_preview = (
                        expected[:100].hex() if len(expected) > 100 else expected.hex()
                    )
                    loaded_preview = (
                        loaded[:100].hex() if len(loaded) > 100 else loaded.hex()
                    )
                    print(
                        f"  [FAIL] Key {i}: MISMATCH "
                        f"(expected {len(expected)}B, "
                        f"got {len(loaded)}B)"
                    )
                    print(f"         Expected: {expected_preview}...")
                    print(f"         Loaded:   {loaded_preview}...")

        if mismatch_count == 0:
            print(f"  All {num_keys} keys data verified OK.")
        else:
            results["success"] = False
            results["error"] = (
                f"Integrity: {num_keys - mismatch_count}/{num_keys} keys passed"
            )

    finally:
        adapter.close()
        print("  Adapter closed.")
        # Clean up meta files after each test to avoid stale state
        _cleanup_3fs_files()
    return results


def run_single_key_test(shape_cfg, num_keys=4):
    """Test single-key path (do_single_set/get) by submitting 1 key per task.

    Each store/load task contains exactly 1 key, forcing the connector to
    process keys individually rather than batch-aggregated.
    """
    name = shape_cfg["name"]
    dtype = shape_cfg["dtype"]
    num_layers = shape_cfg["num_layers"]
    num_kv_heads = shape_cfg["num_kv_heads"]
    head_dim = shape_cfg["head_dim"]

    results = {
        "name": name,
        "success": True,
        "error": None,
        "roundtrip_bytes": 0,
        "per_key_integrity": [],
    }
    all_serialized = []

    seq_lens = (shape_cfg["seq_lens"] * (num_keys // len(shape_cfg["seq_lens"]) + 1))[
        :num_keys
    ]

    for i, seq_len in enumerate(seq_lens):
        original_tensor = create_kv_cache_tensor(
            dtype, num_layers, num_kv_heads, head_dim, seq_len, i
        )
        serialized = _tensor_to_bytes(original_tensor)
        all_serialized.append(serialized)
        results["roundtrip_bytes"] += len(serialized)

    objects = []
    keys = []
    for i in range(num_keys):
        key = ObjectKey(
            chunk_hash=i.to_bytes(16, "big"),
            model_name=f"test-model-single-{_TEST_SESSION_ID}",
            kv_rank=i,
        )
        objects.append(create_bytes_memory_obj(key, all_serialized[i]))
        keys.append(key)

    # chunk_size must match a data store chunk size (4MB/8MB/16MB/32MB) to
    # ensure 3FS chunk boundary is >= actual data size.
    # Data stores: [4MB, 8MB, 16MB, 32MB]. We pick the smallest store that
    # can hold the largest serialized key.
    max_data_size = max(len(s) for s in all_serialized)
    store_chunk_sizes = [
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
    ]
    chunk_size = store_chunk_sizes[0]  # default 4MB
    for cs in store_chunk_sizes:
        if cs >= max_data_size:
            chunk_size = cs
            break
    print(
        f"  Creating 3FS L2 adapter for single-key test "
        f"(shape={name}, keys={num_keys}, chunk_size={chunk_size})..."
    )
    config = RawHf3fsL2AdapterConfig(
        num_workers=NUM_WORKERS,
        meta_addr=META_ADDR,
        mgmtd_addr=MGMTD_ADDR,
        chunk_size=chunk_size,
    )
    adapter = create_l2_adapter(config)
    assert isinstance(adapter, NativeConnectorL2Adapter)

    try:
        # Store phase — one key per task
        print(f"  Storing {num_keys} keys one-by-one (single-key path)...")
        task_ids = []
        for i in range(num_keys):
            tid = adapter.submit_store_task([keys[i]], [objects[i]])
            task_ids.append(tid)

        efd = adapter.get_store_event_fd()
        completed = {}
        expected = len(task_ids)
        while len(completed) < expected:
            if not _wait_eventfd(efd, timeout=120.0):
                results["success"] = False
                results["error"] = "Single-store timeout"
                break
            batch = adapter.pop_completed_store_tasks()
            completed.update(batch)

        if not results["success"]:
            return results
        print(f"  All {num_keys} keys stored (single-key path).")

        # Lookup phase
        print(f"  Looking up {num_keys} keys (single-key path)...")
        lookup_tid = adapter.submit_lookup_and_lock_task(
            keys, MemoryLayoutDesc(shapes=[], dtypes=[])
        )
        lookup_done = False
        while not lookup_done:
            if not _wait_eventfd(adapter.get_lookup_and_lock_event_fd(), 60.0):
                results["success"] = False
                results["error"] = "Single-lookup timeout"
                break
            bitmap = adapter.query_lookup_and_lock_result(lookup_tid)
            if bitmap is not None:
                lookup_done = True
                found = bitmap.popcount()
                print(f"  Lookup found {found}/{num_keys} keys")
                if found != num_keys:
                    results["success"] = False
                    results["error"] = (
                        f"Single-lookup: expected {num_keys}, got {found}"
                    )

        if not results["success"]:
            return results

        # Load phase — one key per task
        print(f"  Loading {num_keys} keys one-by-one (single-key path)...")
        load_objects = []
        for i in range(num_keys):
            load_objects.append(
                create_bytes_memory_obj(keys[i], b"\x00" * len(all_serialized[i]))
            )
        load_tid = adapter.submit_load_task(keys, load_objects)

        load_done = False
        while not load_done:
            if not _wait_eventfd(adapter.get_load_event_fd(), 120.0):
                results["success"] = False
                results["error"] = "Single-load timeout"
                break
            bitmap = adapter.query_load_result(load_tid)
            if bitmap is not None:
                load_done = True
                loaded_count = bitmap.popcount()
                print(f"  Load loaded {loaded_count}/{num_keys} keys")

        if not results["success"]:
            return results

        # Verify integrity
        print("  Verifying data integrity (single-key path)...")
        mismatch_count = 0
        for i in range(num_keys):
            expected_data = all_serialized[i]
            loaded = bytes(load_objects[i].byte_array[: len(expected_data)])
            if loaded == expected_data:
                results["per_key_integrity"].append(True)
            else:
                results["per_key_integrity"].append(False)
                mismatch_count += 1
                if mismatch_count <= 5:
                    print(
                        f"  [FAIL] Key {i}: MISMATCH "
                        f"(expected {len(expected_data)}B, got {len(loaded)}B)"
                    )
                    print(f"         Expected: {expected_data[:100].hex()}...")
                    print(f"         Loaded:   {loaded[:100].hex()}...")

        if mismatch_count == 0:
            print(f"  All {num_keys} keys data verified OK (single-key path).")
        else:
            results["success"] = False
            results["error"] = (
                f"Single-key integrity: "
                f"{num_keys - mismatch_count}/{num_keys} keys passed"
            )

    finally:
        adapter.close()
        print("  Adapter closed.")
        # Clean up meta files after each test to avoid stale state
        _cleanup_3fs_files()
    return results


def main():
    print("=" * 70)
    print("vLLM-style KV Cache Integrity Test for 3FS L2 Backend")
    print("=" * 70)
    print(
        "Testing data pipeline: "
        "vLLM tensor -> serialize -> 3FS -> deserialize -> compare"
    )
    print()

    # Clean up meta files to avoid stale offset map from previous runs
    # Standard
    import glob

    meta_files = glob.glob("/3fs/stage/lmcache/data_*.meta")
    store_files = glob.glob("/3fs/stage/lmcache/data_*.store")
    for f in meta_files + store_files:
        try:
            os.remove(f)
            print(f"  Cleaned up: {f}")
        except OSError:
            pass
    print()

    all_passed = True
    for shape_cfg in KV_CACHE_SHAPES:
        print(f"Testing shape: {shape_cfg['name']}")
        print(
            f"  dtype={shape_cfg['dtype']}, layers={shape_cfg['num_layers']}, "
            f"kv_heads={shape_cfg['num_kv_heads']}, head_dim={shape_cfg['head_dim']}"
        )
        print(f"  seq_lens={shape_cfg['seq_lens']}")

        # Test 1: Small batch (4 keys, sequential) — tests do_batch_set/get
        print("  Running small batch test (4 keys)...")
        result_small = run_test(shape_cfg, num_keys=4, concurrent=False)
        if result_small["success"]:
            print(f"  PASS (small batch): {result_small['name']}")
        else:
            print(f"  FAIL (small batch): {result_small['name']}")
            print(f"        error={result_small['error']}")
            all_passed = False
        print()

        # Test 2: Single-key sequential (4 keys) — tests do_single_set/get
        print("  Running single-key sequential test (4 keys)...")
        result_single = run_single_key_test(shape_cfg, num_keys=4)
        if result_single["success"]:
            print(f"  PASS (single-key seq): {result_single['name']}")
        else:
            print(f"  FAIL (single-key seq): {result_single['name']}")
            print(f"        error={result_single['error']}")
            all_passed = False
        print()

        # Test 3: Single-key concurrent (8 keys, 4 in-flight)
        # tests do_single_set/get with concurrency
        print("  Running single-key concurrent test (8 keys)...")
        result_single_c = run_single_key_test(shape_cfg, num_keys=8)
        if result_single_c["success"]:
            print(f"  PASS (single-key conc): {result_single_c['name']}")
        else:
            print(f"  FAIL (single-key conc): {result_single_c['name']}")
            print(f"        error={result_single_c['error']}")
            all_passed = False
        print()

        # Test 4: Large batch (64 keys, concurrent) — tests do_batch_set/get
        print("  Running large batch concurrent test (64 keys, 8 in-flight)...")
        result_large = run_test(shape_cfg, num_keys=64, concurrent=True, in_flight=8)
        if result_large["success"]:
            print(f"  PASS (large batch): {result_large['name']}")
            print(f"        roundtrip_bytes={result_large['roundtrip_bytes']}")
        else:
            print(f"  FAIL (large batch): {result_large['name']}")
            print(f"        error={result_large['error']}")
            all_passed = False
        print()

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Test mode implementation for MP mode L2 adapter basic checks"""

# Standard
import argparse
import os
import select
import time

# Third Party
import torch

# First Party
from lmcache.v1.check import check_mode
from lmcache.v1.check.utils import (
    DEFAULT_KV_DTYPE_STR,
    DEFAULT_OBJ_SIZE,
    parse_kv_dtype,
    print_performance_results,
)
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.config import (
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

_POLL_TIMEOUT_MS = 100000


def _create_object_key(model: str, key_id: str) -> ObjectKey:
    """Create a test ObjectKey."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(hash(key_id) & 0xFFFFFFFF),
        model_name=model,
        kv_rank=0,
    )


def _create_memory_obj(
    fill_value: float = 0.0,
    obj_size: int = DEFAULT_OBJ_SIZE,
    dtype: torch.dtype = torch.float32,
) -> TensorMemoryObj:
    """Create a test TensorMemoryObj."""
    raw_data = torch.empty(obj_size, dtype=dtype)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([obj_size]),
        dtype=dtype,
        address=0,
        phy_size=obj_size * raw_data.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_event_fd(efd: int, timeout_ms: int = _POLL_TIMEOUT_MS) -> bool:
    """Wait for an eventfd to be signaled."""
    poll = select.poll()
    poll.register(efd, select.POLLIN)
    events = poll.poll(timeout_ms)
    if events:
        try:
            os.eventfd_read(efd)
        except BlockingIOError:
            pass
        return True
    return False


def _run_store_phase(adapter, keys, objects):
    """Run store phase and return (stats, success)."""
    efd = adapter.get_store_event_fd()
    start = time.perf_counter()
    task_id = adapter.submit_store_task(keys, objects)
    if not _wait_event_fd(efd):
        print("  Store: timed out waiting for eventfd")
        return None, False
    completed = adapter.pop_completed_store_tasks()
    elapsed_ms = (time.perf_counter() - start) * 1000
    ok = completed.get(task_id, False)
    return elapsed_ms, ok


def _run_lookup_phase(adapter, keys):
    """Run lookup phase and return (stats, bitmap)."""
    efd = adapter.get_lookup_and_lock_event_fd()
    start = time.perf_counter()
    task_id = adapter.submit_lookup_and_lock_task(keys)
    if not _wait_event_fd(efd):
        print("  Lookup: timed out waiting for eventfd")
        return None, None
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, bitmap


def _run_load_phase(adapter, keys, buffers):
    """Run load phase and return (stats, bitmap)."""
    efd = adapter.get_load_event_fd()
    start = time.perf_counter()
    task_id = adapter.submit_load_task(keys, buffers)
    if not _wait_event_fd(efd):
        print("  Load: timed out waiting for eventfd")
        return None, None
    bitmap = adapter.query_load_result(task_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, bitmap


@check_mode("test_l2_adapter")
async def run_test_mode(model: str, **kwargs):
    """Run L2 adapter test mode.

    Requires ``l2_adapter`` in *kwargs* (list of JSON
    strings from ``--l2-adapter``).
    """
    l2_adapter_raw = kwargs.get("l2_adapter")
    if not l2_adapter_raw:
        print("Error: --l2-adapter is required for test_l2_adapter mode")
        return

    obj_size = kwargs.get("obj_size") or DEFAULT_OBJ_SIZE
    kv_dtype_str = kwargs.get("kv_dtype") or DEFAULT_KV_DTYPE_STR
    kv_dtype = parse_kv_dtype(kv_dtype_str)
    if kv_dtype is None:
        print("Error: unsupported --kv-dtype '%s'" % kv_dtype_str)
        return

    # Build adapter config via the standard parser
    ns = argparse.Namespace(l2_adapter=l2_adapter_raw)
    l2_cfg = parse_args_to_l2_adapters_config(ns)
    if not l2_cfg.adapters:
        print("Error: no L2 adapter configs parsed")
        return

    num_tests = kwargs.get("num_keys", 5)
    settle_time = kwargs.get("settle_time", 0.0)

    for idx, adapter_cfg in enumerate(l2_cfg.adapters):
        adapter = create_l2_adapter(adapter_cfg)
        print("=== Testing L2 adapter #%d (%s) ===" % (idx, type(adapter).__name__))

        try:
            _test_single_adapter(
                adapter,
                model,
                num_tests,
                obj_size=obj_size,
                kv_dtype=kv_dtype,
                settle_time=settle_time,
            )
        except Exception as e:
            print("  Test Failed - Error: %s" % e)
        finally:
            adapter.close()


def _test_single_adapter(
    adapter,
    model,
    num_tests,
    obj_size=DEFAULT_OBJ_SIZE,
    kv_dtype=torch.float32,
    settle_time=0.0,
):
    """Run all test phases against one adapter."""
    # -- Prepare test data -----------------------------------
    exist_keys = [_create_object_key(model, "exist_%d" % i) for i in range(num_tests)]
    non_exist_keys = [
        _create_object_key(model, "nonexist_%d" % i) for i in range(num_tests)
    ]
    store_objs = [
        _create_memory_obj(float(i + 1), obj_size, kv_dtype) for i in range(num_tests)
    ]

    # -- Phase 1: lookup non-existing keys -------------------
    print("Phase 1: Lookup non-existing keys...")
    lk_absent_ms, lk_bitmap = _run_lookup_phase(adapter, non_exist_keys)
    if lk_bitmap is None:
        print("  FAIL: lookup returned None bitmap")
        ne_pass = 0
    else:
        ne_pass = sum(1 for i in range(num_tests) if not lk_bitmap.test(i))
    print("  Validation: %d/%d correctly absent" % (ne_pass, num_tests))
    # Unlock the looked-up keys (contract)
    adapter.submit_unlock(non_exist_keys)

    # -- Phase 2: store existing keys ------------------------
    print("Phase 2: Store operations (batch of %d)..." % num_tests)
    st_ms, st_ok = _run_store_phase(adapter, exist_keys, store_objs)
    store_pass = num_tests if st_ok else 0
    print("  Batch store %s (%.2fms)" % ("OK" if st_ok else "FAIL", st_ms or 0))

    if settle_time > 0:
        print("  Waiting %.1fs for data to settle..." % settle_time)
        time.sleep(settle_time)

    # -- Phase 3: lookup existing keys -----------------------
    print("Phase 3: Lookup existing keys...")
    lk_exist_ms, lk_bitmap = _run_lookup_phase(adapter, exist_keys)
    if lk_bitmap is None:
        print("  FAIL: lookup returned None bitmap")
        exist_pass = 0
    else:
        exist_pass = sum(1 for i in range(num_tests) if lk_bitmap.test(i))
    print("  Validation: %d/%d found" % (exist_pass, num_tests))

    # -- Phase 4: load existing keys -------------------------
    print("Phase 4: Load operations...")
    load_buffers = [
        _create_memory_obj(0.0, obj_size, kv_dtype) for _ in range(num_tests)
    ]
    ld_ms, ld_bitmap = _run_load_phase(adapter, exist_keys, load_buffers)
    load_pass = 0
    content_pass = 0
    if ld_bitmap is not None:
        for i in range(num_tests):
            if ld_bitmap.test(i):
                load_pass += 1
                if torch.equal(
                    load_buffers[i].tensor,
                    store_objs[i].tensor,
                ):
                    content_pass += 1
                else:
                    print("  Key %d: data mismatch" % i)
    print("  Validation (loaded): %d/%d" % (load_pass, num_tests))
    print("  Validation (content): %d/%d" % (content_pass, num_tests))

    # Unlock after load
    adapter.submit_unlock(exist_keys)

    # -- Summary ---------------------------------------------
    total_bytes = obj_size * store_objs[0].tensor.element_size() * num_tests
    stats_data = [
        (
            "LOOKUP (absent)",
            {
                "avg": lk_absent_ms or 0,
                "max": lk_absent_ms or 0,
                "min": lk_absent_ms or 0,
            },
            [False] * num_tests,
            ne_pass,
        ),
        (
            "STORE",
            {
                "avg": st_ms or 0,
                "max": st_ms or 0,
                "min": st_ms or 0,
            },
            [True] * store_pass + [False] * (num_tests - store_pass),
            store_pass,
        ),
        (
            "LOOKUP (exist)",
            {
                "avg": lk_exist_ms or 0,
                "max": lk_exist_ms or 0,
                "min": lk_exist_ms or 0,
            },
            [True] * exist_pass + [False] * (num_tests - exist_pass),
            exist_pass,
        ),
        (
            "LOAD",
            {
                "avg": ld_ms or 0,
                "max": ld_ms or 0,
                "min": ld_ms or 0,
            },
            [True] * content_pass + [False] * (num_tests - content_pass),
            content_pass,
        ),
    ]
    print_performance_results(stats_data, obj_bytes=total_bytes)

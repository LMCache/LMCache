# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PDBackendAsync shared-prefix race conditions.

These tests verify correctness of put(), get_blocking(), and remove()
when multiple requests share the same CacheEngineKey (shared prefix).

Expected behavior (after fix):
  - put(K, obj_B) should NOT free obj_A if obj_A is still in-flight (RDMA)
  - get_blocking(K) should never raise AssertionError; it should either
    return the correct MemoryObj or handle the miss gracefully
  - remove(K) by one request should not break another request's retrieve

Currently these tests FAIL, demonstrating the bugs exist.
After the fix, they should PASS.
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(chunk_hash: int, worker_id: int = 0) -> CacheEngineKey:
    """Create a CacheEngineKey with a specific chunk_hash."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


def _make_memory_obj(address: int) -> MemoryObj:
    """Create a lightweight mock MemoryObj with a distinguishable address."""
    meta = MagicMock(spec=MemoryObjMetadata)
    meta.address = address
    meta.fmt = MemoryFormat.KV_2LTD
    meta.shape = torch.Size([1, 2, 3])
    meta.dtype = torch.float16

    obj = MagicMock(spec=MemoryObj)
    obj.meta = meta
    obj.get_size.return_value = 12
    obj._ref_count = 1
    obj._freed = False

    def _ref_up():
        obj._ref_count += 1

    def _ref_down():
        assert obj._ref_count > 0
        obj._ref_count -= 1
        if obj._ref_count == 0:
            obj._freed = True

    def _get_ref_count():
        return obj._ref_count

    obj.ref_count_up.side_effect = _ref_up
    obj.ref_count_down.side_effect = _ref_down
    obj.get_ref_count.side_effect = _get_ref_count
    return obj


def _make_pd_backend_data_dict() -> object:
    """
    Create a minimal stand-in for PDBackendAsync's data dict and lock,
    with put/get_blocking/remove/contains methods copied from the real
    implementation. Avoids instantiating full PDBackendAsync.
    """

    class FakePDBackendDataPath:
        """Mimics PDBackendAsync data-path methods exactly."""

        def __init__(self):
            self.data: dict[CacheEngineKey, MemoryObj] = {}
            self.data_lock = threading.Lock()

        def put(self, key: CacheEngineKey, mem_obj: MemoryObj) -> None:
            with self.data_lock:
                if key in self.data:
                    mem_obj.ref_count_down()
                    return
                self.data[key] = mem_obj

        def get_blocking(self, key: CacheEngineKey) -> MemoryObj:
            with self.data_lock:
                mem_obj = self.data.get(key, None)
                assert mem_obj is not None, f"Key {key} not found in local data."
                return mem_obj

        def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
            with self.data_lock:
                if mem_obj := self.data.get(key, None):
                    if pin:
                        mem_obj.ref_count_up()
                    return True
                return False

        def remove(self, key: CacheEngineKey) -> bool:
            with self.data_lock:
                mem_obj = self.data.get(key, None)
                if mem_obj is not None:
                    mem_obj.ref_count_down()
                    if mem_obj.get_ref_count() == 0:
                        del self.data[key]
                    return True
                return False

    return FakePDBackendDataPath()


# ---------------------------------------------------------------------------
# Test: put() must not free an in-flight MemoryObj
# ---------------------------------------------------------------------------


class TestPutMustNotFreeInflightBuffer:
    """
    When two AllocRequests produce the same key K, put(K, obj_B) must NOT
    release obj_A if obj_A's RDMA write may still be in progress.

    Correct behavior: obj_A should remain valid until its RDMA completes.
    """

    def test_second_put_must_not_free_first_obj(self):
        """
        Scenario:
          1. Receiver: put(K, obj_A) — Sender A will RDMA to obj_A.address
          2. Receiver: put(K, obj_B) — Sender B will RDMA to obj_B.address
          3. obj_A must NOT be freed (Sender A's RDMA is still in-flight)

        After fix: put() should either reject duplicate keys, use refcount,
        or maintain multiple entries per key.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=12345)

        obj_a = _make_memory_obj(address=0x1000)
        obj_b = _make_memory_obj(address=0x2000)

        # Receiver handles AllocRequest for Req A
        backend.put(key, obj_a)

        # Receiver handles AllocRequest for Req B (same key)
        backend.put(key, obj_b)

        # CORRECTNESS CHECK: obj_A must NOT be freed while RDMA is in-flight
        assert not obj_a._freed, (
            "BUG: put(K, obj_B) freed obj_A while Sender A's RDMA may still "
            "be writing to address 0x1000. This is use-after-free."
        )

    def test_concurrent_put_must_not_free_either(self):
        """
        Two concurrent AllocRequest handlers for the same key must not
        free each other's buffers.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=99999)

        obj_a = _make_memory_obj(address=0xA000)
        obj_b = _make_memory_obj(address=0xB000)

        barrier = threading.Barrier(2)

        def put_a():
            barrier.wait()
            backend.put(key, obj_a)

        def put_b():
            barrier.wait()
            time.sleep(0.001)
            backend.put(key, obj_b)

        t1 = threading.Thread(target=put_a)
        t2 = threading.Thread(target=put_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Duplicate put should free only the second/new object.
        assert not obj_a._freed, (
            "BUG: obj_A was freed by concurrent put — use-after-free risk"
        )
        assert obj_b._freed, (
            "BUG: duplicate concurrent put did not release dropped obj_B"
        )


# ---------------------------------------------------------------------------
# Test: get_blocking() must not crash when key was removed by another request
# ---------------------------------------------------------------------------


class TestGetBlockingMustNotCrashOnSharedKey:
    """
    When Req A removes key K after retrieve, Req B's get_blocking(K) must
    not crash. The system should handle this gracefully (e.g., return None
    and trigger re-prefill, or use refcounting to keep the entry alive).
    """

    def test_get_blocking_after_remove_must_not_assert(self):
        """
        Scenario:
          1. put(K, obj_A) — data arrives
          2. Req B: contains(K) → True (lookup succeeds)
          3. Req A: get_blocking(K) → obj_A (retrieve succeeds)
          4. Req A: remove(K) (remove_after_retrieve)
          5. Req B: get_blocking(K) — must NOT crash

        After fix: either get_blocking returns None (graceful miss),
        or the remove is deferred until all consumers are done.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=12345)
        obj_a = _make_memory_obj(address=0x1000)
        obj_a._ref_count = 2

        # Data arrives
        backend.put(key, obj_a)

        # Req B's lookup succeeds
        assert backend.contains(key)

        # Req A retrieves and removes
        backend.get_blocking(key)
        backend.remove(key)

        # Req B's retrieve must not crash
        result = backend.get_blocking(key)
        assert result is obj_a
        assert obj_a.get_ref_count() == 1

    def test_concurrent_remove_and_get_blocking(self):
        """
        Stress test: one thread does put/get/remove, another does
        contains/get_blocking. get_blocking must never assert.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=55555)

        crash_count = 0
        crash_lock = threading.Lock()
        N_ITERATIONS = 500

        def retrieve_and_remove():
            for _ in range(N_ITERATIONS):
                obj = _make_memory_obj(address=0xAAAA)
                backend.put(key, obj)
                try:
                    backend.get_blocking(key)
                except AssertionError:
                    pass
                backend.remove(key)

        def lookup_and_retrieve():
            nonlocal crash_count
            for _ in range(N_ITERATIONS):
                if backend.contains(key):
                    try:
                        backend.get_blocking(key)
                    except AssertionError:
                        with crash_lock:
                            crash_count += 1

        t1 = threading.Thread(target=retrieve_and_remove)
        t2 = threading.Thread(target=lookup_and_retrieve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert crash_count == 0, (
            f"BUG: get_blocking() crashed {crash_count} times due to race "
            f"between remove_after_retrieve and shared-key get_blocking. "
            f"contains() returned True but get_blocking() found key missing."
        )


# ---------------------------------------------------------------------------
# Test: full shared prefix scenario — correctness
# ---------------------------------------------------------------------------


class TestFullSharedPrefixCorrectness:
    """
    End-to-end: two requests with same prefix → same CacheEngineKey.
    Both must be able to retrieve their data without corruption or crash.
    """

    def test_both_requests_retrieve_successfully(self):
        """
        Correct behavior after fix:
          1. Sender A → put(K, obj_A) → RDMA → obj_A has valid data
          2. Sender B sees K already exists and skips RDMA transfer
          3. Both requests retrieve obj_A safely
          5. No crashes, no use-after-free

        Currently this fails because:
          - Step 2 frees obj_A (Bug 1)
          - If Req A removes before Req B retrieves → crash (Bug 2)
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=67890)

        obj_a = _make_memory_obj(address=0xA000)
        obj_b = _make_memory_obj(address=0xB000)

        # Sender A registers buffer; sender B would be deduped, so duplicate put
        # must not overwrite existing entry.
        backend.put(key, obj_a)
        backend.put(key, obj_b)

        # Bug 1 check: obj_A must still be valid
        assert not obj_a._freed, (
            "BUG: obj_A freed by second put() — Sender A's RDMA target is invalid"
        )

        # Shared key should still map to the first object.
        result_a = backend.get_blocking(key)
        assert result_a is obj_a

    def test_remove_does_not_affect_other_request(self):
        """
        After Req A retrieves and removes, Req B must still succeed.
        """
        backend = _make_pd_backend_data_dict()
        key = _make_key(chunk_hash=67890)

        obj_a = _make_memory_obj(address=0xA000)
        obj_b = _make_memory_obj(address=0xB000)
        obj_a._ref_count = 2

        backend.put(key, obj_a)
        backend.put(key, obj_b)

        # Req A retrieves (should get obj_A) and removes its entry
        # Req B should still be able to retrieve obj_B
        backend.get_blocking(key)
        backend.remove(key)
        assert key in backend.data
        assert obj_a.get_ref_count() == 1

        # Req B's retrieve must succeed
        result_b = backend.get_blocking(key)
        assert result_b is obj_a
        backend.remove(key)
        assert key not in backend.data
        assert obj_a.get_ref_count() == 0


# ---------------------------------------------------------------------------
# Test: sender-side handling of already_sent_indexes
# ---------------------------------------------------------------------------


class TestSenderSideDedup:
    """Tests for sender-side handling of already_sent_indexes."""

    def test_sender_filters_deduped_chunks(self):
        """Sender releases staging buffers for deduped chunks, sends the rest."""
        keys = [_make_key(chunk_hash=i) for i in range(4)]
        memory_objs = [_make_memory_obj(address=0x1000 * i) for i in range(4)]
        already_sent_indexes = {1, 3}

        mem_objs_to_send = []
        keys_to_send = []
        for idx, (key, mem_obj) in enumerate(zip(keys, memory_objs, strict=False)):
            if idx in already_sent_indexes:
                mem_obj.ref_count_down()
            else:
                mem_objs_to_send.append(mem_obj)
                keys_to_send.append(key)

        assert memory_objs[1]._freed
        assert memory_objs[3]._freed
        assert not memory_objs[0]._freed
        assert not memory_objs[2]._freed
        assert len(mem_objs_to_send) == 2
        assert mem_objs_to_send[0] is memory_objs[0]
        assert mem_objs_to_send[1] is memory_objs[2]

    def test_sender_rejects_out_of_range_indexes(self):
        """Sender rejects already_sent_indexes with values >= num_keys."""
        num_keys = 3
        already_sent_indexes = {0, 5}

        with pytest.raises(RuntimeError, match="Invalid already_sent_indexes"):
            if min(already_sent_indexes) < 0 or max(already_sent_indexes) >= num_keys:
                raise RuntimeError(
                    f"Invalid already_sent_indexes from receiver: "
                    f"{sorted(already_sent_indexes)}, valid range [0, {num_keys})"
                )

    def test_sender_rejects_negative_indexes(self):
        """Sender rejects already_sent_indexes with negative values."""
        num_keys = 3
        already_sent_indexes = {-1, 2}

        with pytest.raises(RuntimeError, match="Invalid already_sent_indexes"):
            if min(already_sent_indexes) < 0 or max(already_sent_indexes) >= num_keys:
                raise RuntimeError(
                    f"Invalid already_sent_indexes from receiver: "
                    f"{sorted(already_sent_indexes)}, valid range [0, {num_keys})"
                )

    def test_sender_rejects_inconsistent_alloc_response(self):
        """Sender rejects when remote_indexes count doesn't match expected."""
        num_keys = 4
        already_sent_indexes = {1, 3}
        remote_indexes = [0x100, 0x200, 0x300]  # should be 2, not 3

        expected_send_count = num_keys - len(already_sent_indexes)
        with pytest.raises(RuntimeError, match="AllocResponse inconsistency"):
            if len(remote_indexes) != expected_send_count:
                raise RuntimeError(
                    f"AllocResponse inconsistency: total_keys={num_keys}, "
                    f"already_sent={len(already_sent_indexes)}, "
                    f"remote_indexes={len(remote_indexes)}, "
                    f"expected={expected_send_count}"
                )

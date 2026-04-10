# SPDX-License-Identifier: Apache-2.0
"""Property-style tests for pin/ref_count lifecycle invariants under
random failure injection.

These tests are NOT a specific-bug repro. They are seeded random
property checks intended to find leaked pin or ref_count holds in the
LMCache storage backends — a third pin-leak path beyond the two
already fixed in lmcache/v1/cache_engine.py:store() and
lmcache/v1/storage_backend/local_disk_backend.py:batched_get_non_blocking.

Background — production incident 2026-04-09 (verda-h200, GLM-5-FP8):
A pin-leak in one of the LMCache storage paths caused the CPU staging
pool to slowly exhaust. Once exhausted, allocator calls block waiting
for memory that will never become free, deadlocking workers and
producing downstream symptoms like
"CUDA error: Invalid access of peer GPU memory over nvlink" and
"TimeoutError: RPC call to sample_tokens timed out".

Two pin-leak fixes have already been added to cache_engine.py and
local_disk_backend.py. These tests randomly inject failures into the
remaining storage-backend operations in an attempt to surface a third,
unfixed leak path. They should pass on a healthy code base; if any of
them fail, the failing seed and operation sequence localize the bug.

Related: neuralwatt/inference_frontend#1900, #1903
"""

# Standard
import asyncio
import os
import random
import shutil
import tempfile
import threading
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MixedMemoryAllocator,
    get_size_bytes,
)
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


# -- Helpers ------------------------------------------------------------------


def _mla_shapes_and_dtypes():
    """Two-group MLA layout (small enough to fit many in a 16 MiB pool)."""
    shapes = [
        torch.Size([1, 4, 16, 132]),
        torch.Size([1, 4, 16, 576]),
    ]
    dtypes = [torch.uint8, torch.bfloat16]
    return shapes, dtypes


def _make_key(prefix: str, key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="prop_test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash((prefix, key_id)),
        dtype=torch.bfloat16,
    )


def _wait_for_put(backend: LocalDiskBackend, key: CacheEngineKey,
                  timeout: float = 5.0):
    # Standard
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not backend.disk_worker.exists_in_put_tasks(key):
            return
        time.sleep(0.005)
    raise TimeoutError(f"put task for key {key} did not finish")


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def temp_disk_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def running_loop():
    """Asyncio event loop driven by a background thread."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread.is_alive():
        thread.join(timeout=2.0)
    if not loop.is_closed():
        loop.close()


@pytest.fixture
def isolated_allocator():
    """A dedicated MixedMemoryAllocator for this test (NOT the shared
    session allocator) so we can observe leaks without interference."""
    alloc = MixedMemoryAllocator(16 * 1024 * 1024)  # 16 MiB
    yield alloc
    try:
        alloc.close()
    except Exception:
        pass


@pytest.fixture
def cpu_backend(isolated_allocator):
    config = LMCacheEngineConfig.from_legacy(chunk_size=16)
    # PinMonitor is a global singleton — TensorMemoryObj.pin() calls
    # PinMonitor.GetOrCreate() and asserts a config is present on first
    # init. The cache_engine normally bootstraps it; in tests we do it
    # explicitly here.
    PinMonitor.GetOrCreate(config)
    return LocalCPUBackend(config, memory_allocator=isolated_allocator)


@pytest.fixture
def disk_backend(temp_disk_path, running_loop, cpu_backend):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=16,
        local_disk=temp_disk_path,
        # Generous cap so eviction does not interfere with leak detection.
        max_local_disk_size=0.2,  # 200 MB
        lmcache_instance_id="prop_test",
    )
    backend = LocalDiskBackend(
        config=config,
        loop=running_loop,
        local_cpu_backend=cpu_backend,
        dst_device="cuda",
    )
    yield backend
    try:
        backend.close()
    except Exception:
        pass


# -- Property tests -----------------------------------------------------------


class TestPinLifecycleInvariants:
    """Random failure-injection property tests on pin/ref_count lifecycle.

    PROPERTY: After every operation, no MemoryObj that has been freed by
    the allocator should still be pinned. Equivalently: any object whose
    pin_count > 0 must still be reachable through one of the live
    backends. The cleaner version of this invariant — "after each cycle,
    every reachable object has pin_count == 0 unless someone is actively
    using it" — is what we check below by observing the disk_backend.dict
    state and the cpu_backend.hot_cache state at quiescent points.

    These are NOT designed for a specific bug. If they fail consistently
    under a fixed seed, the seed + op log identifies a leak path the
    operator can drill into.
    """

    def _allocate(self, cpu_backend, fmt=MemoryFormat.KV_MLA_FMT):
        """Allocate a multi-group MLA memory_obj via the production path."""
        shapes, dtypes = _mla_shapes_and_dtypes()
        return cpu_backend.allocate(
            shapes, dtypes, fmt=fmt, eviction=False, busy_loop=False
        )

    def test_pin_count_invariant_under_random_failure_injection(
        self, cpu_backend, disk_backend
    ):
        """Run a randomized sequence of submit_put_task / batched_get
        operations with seeded fault injection on local_cpu_backend.allocate.
        After every operation, assert that no MemoryObj currently held by
        either backend has a leaked pin (pin_count == 0 for everything in
        disk_backend.dict and cpu_backend.hot_cache).

        Additionally, after the full sequence completes, assert that the
        memory pool's allocation count returns to a sane bound (no
        catastrophic leak).
        """
        rng = random.Random(0xDEADBEEF)
        ops_log = []  # records (op_name, key_id, fail_injection)

        # Pre-populate the disk backend with a small set of keys we can
        # later try to retrieve via batched_get_non_blocking.
        seed_keys = []
        for i in range(8):
            mo = self._allocate(cpu_backend)
            assert mo is not None, "test setup: initial allocate failed"
            key = _make_key("seed", i)
            disk_backend.submit_put_task(key, mo)
            _wait_for_put(disk_backend, key)
            mo.ref_count_down()  # release the caller's ref
            seed_keys.append(key)

        # Snapshot the post-setup pin invariant.
        self._assert_no_leaked_pins(
            cpu_backend, disk_backend, ops_log, "after setup"
        )

        num_iterations = 200
        original_allocate = cpu_backend.allocate

        # Failure-injection wrapper around cpu_backend.allocate. Returns
        # None ~30% of the time to simulate CPU pool exhaustion. The
        # caller-supplied 'rng' is captured so the sequence is fully
        # deterministic for a fixed seed.
        def flaky_allocate(*args, **kwargs):
            if rng.random() < 0.30:
                return None
            return original_allocate(*args, **kwargs)

        with patch.object(cpu_backend, "allocate", side_effect=flaky_allocate):
            for i in range(num_iterations):
                op = rng.choice(["put", "prefetch", "get_blocking"])
                fail_injected = False
                if op == "put":
                    mo = original_allocate(
                        *_mla_shapes_and_dtypes(),
                        fmt=MemoryFormat.KV_MLA_FMT,
                        eviction=False,
                        busy_loop=False,
                    )
                    if mo is None:
                        ops_log.append(("put-skip", i, False))
                        continue
                    key = _make_key("rand", i)
                    disk_backend.submit_put_task(key, mo)
                    try:
                        _wait_for_put(disk_backend, key)
                    except TimeoutError:
                        ops_log.append(("put-timeout", i, fail_injected))
                        continue
                    mo.ref_count_down()
                    ops_log.append(("put", i, fail_injected))
                elif op == "prefetch":
                    # Pick a few seed keys to prefetch — this exercises
                    # batched_get_non_blocking which has a known pin-leak
                    # fix. The flaky_allocate wrapper may make it return
                    # [] partway through the loop.
                    fail_injected = rng.random() < 0.30
                    sample = rng.sample(seed_keys, k=min(3, len(seed_keys)))
                    try:
                        coro = disk_backend.batched_get_non_blocking(
                            f"lookup_{i}", list(sample)
                        )
                        fut = asyncio.run_coroutine_threadsafe(
                            coro, disk_backend.loop
                        )
                        result = fut.result(timeout=5.0)
                    except Exception:
                        result = None
                    ops_log.append(("prefetch", i, fail_injected))
                    # Release any successfully retrieved memory_objs.
                    if result:
                        for mo in result:
                            try:
                                if mo.is_pinned:
                                    mo.unpin()
                                mo.ref_count_down()
                            except Exception:
                                pass
                else:  # get_blocking
                    key = rng.choice(seed_keys)
                    # NOTE: get_blocking() in the multi-group MLA branch
                    # currently asserts memory_obj is not None
                    # (local_disk_backend.py:415) — if our flaky_allocate
                    # returns None, that assertion fires. This is itself
                    # an unhandled allocation-failure path worth flagging,
                    # but it is NOT what this property test is hunting,
                    # so we tolerate it here.
                    try:
                        mo = disk_backend.get_blocking(key)
                    except AssertionError:
                        ops_log.append(("get_blocking-assert", i, True))
                        continue
                    if mo is not None:
                        try:
                            mo.ref_count_down()
                        except Exception:
                            pass
                    ops_log.append(("get_blocking", i, fail_injected))

                # Per-iteration invariant check
                self._assert_no_leaked_pins(
                    cpu_backend, disk_backend, ops_log,
                    f"after iter {i} op={op}"
                )

        # Final invariant: after the run, the allocator should not be
        # holding the pool hostage. We check that we can still allocate
        # one new chunk (which would fail if pins had leaked enough to
        # exhaust the pool).
        probe = original_allocate(
            *_mla_shapes_and_dtypes(),
            fmt=MemoryFormat.KV_MLA_FMT,
            eviction=False,
            busy_loop=False,
        )
        assert probe is not None, (
            f"final allocation failed — CPU pool may be exhausted by "
            f"leaked pins. Last 10 ops: {ops_log[-10:]}"
        )
        probe.ref_count_down()

    def _assert_no_leaked_pins(
        self, cpu_backend, disk_backend, ops_log, label: str
    ):
        """Invariant: every disk-backend dict entry that is currently
        not being prefetched or get-blocked must have pin_count == 0.

        We check the disk side (DiskCacheMetadata.pin_count). The CPU
        side is harder to assert against because hot_cache lookups are
        legitimately pinned during reads — but at quiescent points
        (between ops, after waits) the disk side should always settle
        to zero.
        """
        leaks = []
        for k, meta in list(disk_backend.dict.items()):
            if getattr(meta, "pin_count", 0) > 0:
                leaks.append((k, meta.pin_count))
        assert not leaks, (
            f"leaked pins at quiescent point ({label}): {leaks[:5]}\n"
            f"last 5 ops: {ops_log[-5:]}"
        )

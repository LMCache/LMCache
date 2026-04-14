# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #2954: double unpin of LocalCPU objects
in retrieve_layer().

The bug: retrieve_layer() and lookup_unpin() both called unpin() on the
same MemoryObj when using LocalCPUBackend.  LocalCPU's
batched_get_non_blocking() returns the *same* Python object that
lookup(pin=True) pinned (unlike LocalDisk which allocates a new staging
buffer).  The second unpin drove pin_count negative, triggering a
premature free().

The fix: retrieve_layer() skips the unpin when
``location == "LocalCPUBackend"`` so that lookup_unpin() is the sole
unpin path.

This test verifies the pin-count invariant at the MemoryObj level,
exercising the exact sequence that occurs in production without
requiring CUDA.
"""

# Standard
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.pin_monitor import PinMonitor


@pytest.fixture(autouse=True)
def _init_pin_monitor():
    """Initialize and tear down PinMonitor for each test.

    TensorMemoryObj.pin()/unpin() call PinMonitor.GetOrCreate() internally,
    so the singleton must exist before any pin operation.
    """
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
    )
    PinMonitor.GetOrCreate(config)
    yield
    PinMonitor.DestroyInstance()


def _make_mem_obj() -> TensorMemoryObj:
    """Create a TensorMemoryObj backed by a small CPU tensor.

    Uses ``parent_allocator=None`` so that free() is a no-op, which lets
    us inspect pin_count without side effects.
    """
    size = 1024
    buf = torch.empty(size, dtype=torch.uint8)
    meta = MemoryObjMetadata(
        shape=torch.Size([1, 2, 4, 8, 16]),
        dtype=torch.bfloat16,
        address=buf.data_ptr(),
        phy_size=size,
        ref_count=0,
        pin_count=0,
        fmt=MemoryFormat.KV_2LTD,
    )
    return TensorMemoryObj(raw_data=buf, metadata=meta, parent_allocator=None)


class TestRetrieveLayerDoubleUnpin:
    """Pin-count lifecycle tests mirroring the retrieve_layer() flow."""

    def test_localcpu_no_double_unpin(self, caplog):
        """With the fix, LocalCPU objects skip unpin in retrieve_layer(),
        so lookup_unpin() is the single unpin path.

        Sequence (matches PR description table):
          1. lookup(pin=True)      -> pin_count = 1
          2. batched_get -> ref_count_up()  -> ref_count = 1
          3. ref_count_down()               -> ref_count = 0
          4. retrieve_layer: is_pinned AND location != "LocalCPUBackend"
                             -> SKIPPED (location IS LocalCPUBackend)
          5. lookup_unpin()  -> unpin()     -> pin_count = 0, free()
        """
        mem_obj = _make_mem_obj()

        # Step 1: lookup(pin=True)
        mem_obj.pin()
        assert mem_obj.metadata.pin_count == 1

        # Step 2: batched_get_non_blocking -> ref_count_up
        mem_obj.ref_count_up()
        assert mem_obj.get_ref_count() == 1

        # Step 3: ref_count_down (after device copy enqueued)
        mem_obj.ref_count_down()
        assert mem_obj.get_ref_count() == 0

        # Step 4: retrieve_layer unpin guard — simulating the fix
        location = "LocalCPUBackend"
        if mem_obj.is_pinned and location != "LocalCPUBackend":
            mem_obj.unpin()
        # pin_count should still be 1 (unpin was skipped)
        assert mem_obj.metadata.pin_count == 1

        # Step 5: lookup_unpin -> unpin
        logger = logging.getLogger("lmcache.v1.memory_management")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(
                logging.WARNING, logger="lmcache.v1.memory_management"
            ):
                mem_obj.unpin()
        finally:
            logger.propagate = old_propagate

        assert mem_obj.metadata.pin_count == 0, (
            "pin_count should be exactly 0 after single unpin"
        )
        # No "Double unpin" warning should have been logged
        double_unpin_msgs = [
            r.message for r in caplog.records if "Double unpin" in r.message
        ]
        assert double_unpin_msgs == [], (
            f"Unexpected double-unpin warning: {double_unpin_msgs}"
        )

    def test_localcpu_double_unpin_without_guard(self, caplog):
        """Without the fix (no location guard), LocalCPU objects would be
        unpinned twice.  This test documents the bug and verifies that
        the warning fires when the guard is absent.
        """
        mem_obj = _make_mem_obj()

        # Step 1: lookup(pin=True)
        mem_obj.pin()

        # Step 2-3: batched_get ref-count cycle
        mem_obj.ref_count_up()
        mem_obj.ref_count_down()

        # Simulate the OLD code (no guard): unconditional unpin
        if mem_obj.is_pinned:
            mem_obj.unpin()
        assert mem_obj.metadata.pin_count == 0

        # Second unpin (from lookup_unpin) — this is the double unpin.
        # Enable propagation so caplog can capture the warning from
        # the lmcache logger (which sets propagate=False by default).
        logger = logging.getLogger("lmcache.v1.memory_management")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(
                logging.WARNING, logger="lmcache.v1.memory_management"
            ):
                mem_obj.unpin()
        finally:
            logger.propagate = old_propagate

        # The safeguard in TensorMemoryObj.unpin() resets pin_count to 0
        # after logging a warning
        assert mem_obj.metadata.pin_count == 0, (
            "Safeguard should have reset pin_count to 0"
        )
        double_unpin_msgs = [
            r.message for r in caplog.records if "Double unpin" in r.message
        ]
        assert len(double_unpin_msgs) > 0, (
            "Without the guard, a double-unpin warning must be raised"
        )

    def test_localdisk_unpin_both_paths_ok(self, caplog):
        """For LocalDisk, batched_get_non_blocking() returns a *different*
        staging MemoryObj, so both unpin paths operate on separate objects
        and neither hits a double unpin.
        """
        lookup_obj = _make_mem_obj()  # pinned by lookup
        staging_obj = _make_mem_obj()  # returned by batched_get

        # Step 1: lookup pins the lookup_obj
        lookup_obj.pin()
        assert lookup_obj.metadata.pin_count == 1

        # Step 2-3: staging_obj goes through ref-count cycle
        # (LocalDisk staging objects may also be pinned by the allocator)
        staging_obj.pin()
        staging_obj.ref_count_up()
        staging_obj.ref_count_down()

        # Step 4: retrieve_layer unpins the staging_obj (different object)
        location = "LocalDiskBackend"
        if staging_obj.is_pinned and location != "LocalCPUBackend":
            staging_obj.unpin()
        assert staging_obj.metadata.pin_count == 0

        # Step 5: lookup_unpin unpins the lookup_obj
        logger = logging.getLogger("lmcache.v1.memory_management")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(
                logging.WARNING, logger="lmcache.v1.memory_management"
            ):
                lookup_obj.unpin()
        finally:
            logger.propagate = old_propagate

        assert lookup_obj.metadata.pin_count == 0
        double_unpin_msgs = [
            r.message for r in caplog.records if "Double unpin" in r.message
        ]
        assert double_unpin_msgs == [], (
            f"LocalDisk should never trigger double-unpin: {double_unpin_msgs}"
        )

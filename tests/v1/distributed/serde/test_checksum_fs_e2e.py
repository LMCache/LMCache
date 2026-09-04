# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the checksum serde with a real filesystem L2 adapter.

Mirrors ``test_serde_fs_e2e.py`` (which exercises fp8 through the same real
disk-I/O path), but the scenario this file actually verifies is the one the
checksum serde exists for: a byte on disk gets corrupted/tampered after
store, and the corrupted key must come back as a miss on prefetch -- never
as silently-wrong KV data.
"""

# Standard
import os
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.platform import current_device_spec

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def _make_key(chunk_hash: bytes) -> ObjectKey:
    """Create an ObjectKey with the given raw hash bytes."""
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="test-model",
        kv_rank=0,
    )


def wait_for_condition(
    predicate,
    timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll until predicate returns True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def wait_for_prefetch_status(
    sm: StorageManager,
    handle,
    timeout: float = 15.0,
    poll_interval: float = 0.1,
):
    """Poll query_prefetch_status until non-None or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _stored_file_paths(disk_path: str) -> list[str]:
    return [
        e.path
        for e in os.scandir(disk_path)
        if e.is_file() and not e.name.endswith(".tmp")
    ]


# =============================================================================
# Test
# =============================================================================


class TestChecksumSerdeFsRoundTrip:
    """Full disk-backed checksum serde round-trip through StorageManager."""

    def test_tampered_file_is_reported_as_miss(self) -> None:
        """Write KV -> checksum serialize -> disk -> flip a byte on disk ->
        clear L1 -> prefetch -> the tampered key must come back as a miss,
        and the untouched key must still round-trip correctly.
        """
        disk_path = tempfile.mkdtemp(prefix="lmcache_checksum_fs_test_")
        try:
            self._run(disk_path)
        finally:
            shutil.rmtree(disk_path, ignore_errors=True)

    def _run(self, disk_path: str) -> None:
        # ---- Config ----
        fs_cfg = FSL2AdapterConfig(
            base_path=disk_path,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="checksum", kwargs={})

        sm_cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 30,
                    use_lazy=current_device_spec.is_pin_supported,
                    init_size_in_bytes=1 << 30,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
        )

        sm = StorageManager(sm_cfg)

        kv_shape = torch.Size([2, 4, 256, 128])
        kv_dtype = torch.bfloat16
        layout = MemoryLayoutDesc(shapes=[kv_shape], dtypes=[kv_dtype])

        good_key = _make_key(b"\x00" * 31 + b"\x01")
        tampered_key = _make_key(b"\x00" * 31 + b"\x02")
        keys = [good_key, tampered_key]

        torch.manual_seed(0)
        originals = {k: torch.randn(kv_shape, dtype=kv_dtype) for k in keys}

        def _store_one(key: ObjectKey) -> None:
            reserved = sm.reserve_write([key], layout, mode="new")
            mem_obj = reserved[key]
            assert mem_obj.tensor is not None
            mem_obj.tensor.view(kv_shape).view(kv_dtype).copy_(originals[key])
            sm.finish_write([key])
            ok = wait_for_condition(
                lambda: (
                    sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                    and sm.report_status()["store_controller"]["pending_keys_count"]
                    == 0
                ),
                timeout=10.0,
            )
            assert ok, f"Store controller did not finish storing {key}"

        # ---- Step 1: store good_key alone, then identify its file by
        # elimination (it is the only file present) ----
        _store_one(good_key)
        ok = wait_for_condition(
            lambda: len(_stored_file_paths(disk_path)) >= 1, timeout=10.0
        )
        assert ok, f"Expected good_key's file under {disk_path}"
        good_paths = set(_stored_file_paths(disk_path))
        assert len(good_paths) == 1

        # ---- Step 2: store tampered_key; its file is whatever is new ----
        _store_one(tampered_key)
        ok = wait_for_condition(
            lambda: len(_stored_file_paths(disk_path)) >= 2, timeout=10.0
        )
        assert ok, f"Expected tampered_key's file under {disk_path}"
        all_paths = set(_stored_file_paths(disk_path))
        tampered_paths = all_paths - good_paths
        assert len(tampered_paths) == 1
        tampered_path = next(iter(tampered_paths))

        # ---- Step 3: corrupt tampered_key's file on disk ----
        with open(tampered_path, "r+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(size - 1)
            last_byte = f.read(1)
            f.seek(size - 1)
            f.write(bytes([last_byte[0] ^ 0x01]))

        # ---- Step 4: clear L1 ----
        sm.clear(force=True)
        assert sm.report_status()["l1_manager"]["total_object_count"] == 0

        # ---- Step 5: prefetch (disk load + checksum verify) ----
        # A deserialize failure is all-or-nothing for the batch. The wrapper
        # therefore reports both keys as misses when the tampered key fails.
        handle = sm.submit_prefetch_task(PrefetchRequestSpec(keys, {0: layout}))
        result = wait_for_prefetch_status(sm, handle)
        assert result is not None, "Prefetch never completed"
        assert result.count_leading_ones() == 0, (
            "A checksum failure must report the whole batch as misses; "
            "corrupted payload must never load as a hit"
        )

        # ---- Step 6: an untouched key still round-trips when loaded alone ----
        sm.clear(force=True)
        good_handle = sm.submit_prefetch_task(
            PrefetchRequestSpec([good_key], {0: layout})
        )
        good_result = wait_for_prefetch_status(sm, good_handle)
        assert good_result is not None, "Good-key prefetch never completed"
        assert good_result.count_leading_ones() == 1
        with sm.read_prefetched_results([good_key]) as mem_objs:
            assert mem_objs is not None
            mem_obj = mem_objs[0]
            assert mem_obj.tensor is not None
            got = mem_obj.tensor.view(kv_shape).view(kv_dtype)
            assert torch.equal(got, originals[good_key]), (
                "good_key did not round-trip exactly"
            )
        sm.finish_read_prefetched([good_key])

        sm.close()

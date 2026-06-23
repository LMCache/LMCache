# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the filesystem L2 adapter capacity reporting.
"""

# Standard
from pathlib import Path
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.memory_management import BytesBufferMemoryObj


def _make_key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    """Create an ObjectKey for FS adapter tests."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _wait_for_store_result(
    adapter: FSL2Adapter,
    task_id: L2TaskId,
) -> L2StoreResult:
    """Wait for a completed store task using the adapter's public API."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
        time.sleep(0.01)
    raise AssertionError(f"store task {task_id} did not complete")


class TestFSL2AdapterConfig:
    """Config parsing for the FS adapter's max capacity field."""

    def test_default_max_capacity_gb_disables_global_eviction(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = FSL2AdapterConfig.from_dict({"base_path": str(tmp_path)})
        assert cfg.max_capacity_gb == 0.0

        adapter = FSL2Adapter(cfg)
        try:
            usage = adapter.get_usage()
            assert usage.total_capacity_bytes == 0
            assert usage.usage_fraction == -1.0
            assert adapter.supports_global_eviction is False
        finally:
            adapter.close()

    def test_max_capacity_gb_sets_capacity_bytes(self, tmp_path: Path) -> None:
        cfg = FSL2AdapterConfig.from_dict(
            {
                "base_path": str(tmp_path),
                "max_capacity_gb": 1.5,
            }
        )
        assert cfg.max_capacity_gb == 1.5

        adapter = FSL2Adapter(cfg)
        try:
            usage = adapter.get_usage()
            assert usage.total_capacity_bytes == int(1.5 * (1024**3))
            assert usage.total_bytes_used == 0
            assert usage.usage_fraction == 0.0
            assert adapter.supports_global_eviction is True
        finally:
            adapter.close()

    @pytest.mark.parametrize("value", [-1, True, "1"])
    def test_invalid_max_capacity_gb_raises(
        self,
        tmp_path: Path,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="max_capacity_gb"):
            FSL2AdapterConfig.from_dict(
                {
                    "base_path": str(tmp_path),
                    "max_capacity_gb": value,
                }
            )

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_max_capacity_gb_raises(
        self,
        tmp_path: Path,
        value: float,
    ) -> None:
        with pytest.raises(ValueError, match="max_capacity_gb"):
            FSL2AdapterConfig.from_dict(
                {
                    "base_path": str(tmp_path),
                    "max_capacity_gb": value,
                }
            )


class TestFSL2AdapterUsage:
    """Usage reporting after successful filesystem stores."""

    def test_store_updates_usage(self, tmp_path: Path) -> None:
        adapter = FSL2Adapter(
            FSL2AdapterConfig.from_dict(
                {
                    "base_path": str(tmp_path),
                    "max_capacity_gb": 1,
                }
            )
        )
        try:
            key = _make_key(1, cache_salt="alice")
            obj = BytesBufferMemoryObj(b"abcd")

            task_id = adapter.submit_store_task([key], [obj])
            result = _wait_for_store_result(adapter, task_id)
            assert result.is_successful()
            assert result.bytes_transferred() == 4

            usage = adapter.get_usage()
            assert usage.total_bytes_used == 4
            assert usage.total_capacity_bytes == 1024**3
            assert dict(usage.bytes_by_cache_salt) == {"alice": 4}

            status = adapter.report_status()
            assert status["max_capacity_bytes"] == 1024**3
            assert status["total_bytes_used"] == 4
            assert status["usage_fraction"] == 4 / (1024**3)
        finally:
            adapter.close()

    def test_store_existing_file_does_not_double_count(
        self,
        tmp_path: Path,
    ) -> None:
        adapter = FSL2Adapter(
            FSL2AdapterConfig.from_dict(
                {
                    "base_path": str(tmp_path),
                    "max_capacity_gb": 1,
                }
            )
        )
        try:
            key = _make_key(1, cache_salt="alice")
            obj = BytesBufferMemoryObj(b"abcd")

            task_id = adapter.submit_store_task([key], [obj])
            first = _wait_for_store_result(adapter, task_id)
            assert first.bytes_transferred() == 4

            task_id = adapter.submit_store_task([key], [obj])
            second = _wait_for_store_result(adapter, task_id)
            assert second.is_successful()
            assert second.bytes_transferred() == 0

            usage = adapter.get_usage()
            assert usage.total_bytes_used == 4
            assert dict(usage.bytes_by_cache_salt) == {"alice": 4}
        finally:
            adapter.close()

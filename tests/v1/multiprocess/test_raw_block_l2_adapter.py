# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Third Party
import pytest

# First Party
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    install_native_storage_ops_fallback,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_file,
    memory_obj_bytes,
    wait_for_event_fd,
)

install_native_storage_ops_fallback()
pytest.importorskip("lmcache_rust_raw_block_io")

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc  # noqa: E402
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (  # noqa: E402
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.storage_backend.raw_block import RawBlockPutManyResult  # noqa: E402

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _make_adapter(tmp_path: Path) -> RawBlockL2Adapter:
    path = make_raw_block_file(tmp_path)
    config = RawBlockL2AdapterConfig(
        device_path=str(path),
        capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        io_engine="posix",
        iouring_queue_depth=8,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )
    return RawBlockL2Adapter(config)


def test_raw_block_l2_adapter_store_lookup_load_roundtrip(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(1)
        missing_key = make_object_key(999)
        payload = b"raw-block-l2-adapter-payload"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        lookup_task_id = adapter.submit_lookup_and_lock_task(
            [key, missing_key], _EMPTY_LAYOUT
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is True
        assert lookup_bitmap.test(1) is False

        loaded = make_empty_memory_obj(len(payload))
        missing = make_empty_memory_obj(len(payload))
        load_task_id = adapter.submit_load_task([key, missing_key], [loaded, missing])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        load_bitmap = adapter.query_load_result(load_task_id)
        assert load_bitmap is not None
        assert load_bitmap.test(0) is True
        assert load_bitmap.test(1) is False
        assert memory_obj_bytes(loaded) == payload

        adapter.submit_unlock([key])
    finally:
        adapter.close()


def test_raw_block_l2_adapter_delete_makes_key_miss(tmp_path):
    adapter = _make_adapter(tmp_path)
    try:
        key = make_object_key(2)
        payload = b"delete-from-raw-block-l2"

        store_task_id = adapter.submit_store_task([key], [make_memory_obj(payload)])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        store_result = adapter.pop_completed_store_tasks()[store_task_id]
        assert store_result.is_successful()
        assert store_result.bytes_transferred() == RAW_BLOCK_CI_SLOT_BYTES

        adapter.delete([key])

        lookup_task_id = adapter.submit_lookup_and_lock_task([key], _EMPTY_LAYOUT)
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
        assert lookup_bitmap is not None
        assert lookup_bitmap.test(0) is False
    finally:
        adapter.close()


class _FakeFdpCore:
    def __init__(self, status: list[tuple[int, int]] | None = None) -> None:
        self.status = status if status is not None else [(0, 10), (7, 17)]
        self.slot_bytes = RAW_BLOCK_CI_SLOT_BYTES
        self.put_many_calls: list[list[int] | None] = []

    def fetch_fdp_status(self) -> list[tuple[int, int]]:
        return self.status

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "usable_capacity_bytes": RAW_BLOCK_CI_SLOT_BYTES * 8,
        }

    def put_many(
        self,
        specs: list[Any],
        objects: list[Any],
        placement_ids: list[int] | None = None,
    ) -> RawBlockPutManyResult:
        self.put_many_calls.append(
            None if placement_ids is None else list(placement_ids)
        )
        return RawBlockPutManyResult(
            results=[True] * len(specs),
            stored_keys=[spec.encoded for spec in specs],
        )

    def snapshot_indexed_keys(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


def _make_fdp_config(
    *,
    placement_ids: list[int] | None = None,
    meta_checkpoint_placement_id: int | None = None,
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path="/dev/ng0n1",
        capacity_bytes=RAW_BLOCK_CI_CAPACITY_BYTES,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_enable_periodic=False,
        meta_idle_quiet_ms=0,
        io_engine="io_uring",
        use_uring_cmd=True,
        iouring_queue_depth=8,
        fdp_enabled=True,
        fdp_placement_ids=placement_ids,
        meta_checkpoint_placement_id=meta_checkpoint_placement_id,
        num_store_workers=1,
        num_lookup_workers=1,
        num_load_workers=1,
    )


def _make_fdp_adapter(
    fake_core: _FakeFdpCore,
    config: RawBlockL2AdapterConfig,
) -> RawBlockL2Adapter:
    with patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.RawBlockCore",
        return_value=fake_core,
    ):
        return RawBlockL2Adapter(config)


def test_raw_block_meta_checkpoint_placement_id_reaches_core_config() -> None:
    config = RawBlockL2AdapterConfig.from_dict(
        {
            "device_path": "/dev/ng0n1",
            "slot_bytes": RAW_BLOCK_CI_SLOT_BYTES,
            "io_engine": "io_uring",
            "use_uring_cmd": True,
            "meta_checkpoint_placement_id": 7,
        }
    )

    assert config.meta_checkpoint_placement_id == 7
    assert config.to_core_config().meta_checkpoint_placement_id == 7


def test_raw_block_meta_checkpoint_placement_id_requires_uring_cmd() -> None:
    with pytest.raises(ValueError, match="meta_checkpoint_placement_id requires"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "device_path": "/tmp/raw-block",
                "slot_bytes": RAW_BLOCK_CI_SLOT_BYTES,
                "io_engine": "posix",
                "meta_checkpoint_placement_id": 7,
            }
        )


def test_raw_block_meta_checkpoint_placement_id_rejects_zero() -> None:
    with pytest.raises(ValueError, match="placement identifier 0"):
        _make_fdp_config(meta_checkpoint_placement_id=0)


def test_raw_block_fdp_requires_uring_cmd_config():
    with pytest.raises(ValueError, match="fdp_enabled requires"):
        RawBlockL2AdapterConfig(
            device_path="/dev/ng0n1",
            slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
            io_engine="posix",
            use_uring_cmd=False,
            fdp_enabled=True,
        )


def test_raw_block_fdp_disabled_ignores_placement_ids() -> None:
    config = RawBlockL2AdapterConfig(
        device_path="/tmp/raw-block",
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        fdp_enabled=False,
        fdp_placement_ids=[0, 1],
    )

    assert config.fdp_enabled is False
    assert config.fdp_placement_ids is None


def test_raw_block_fdp_from_dict_validates_enabled_id_elements() -> None:
    with pytest.raises(ValueError, match="fdp_placement_ids must contain"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "device_path": "/dev/ng0n1",
                "slot_bytes": RAW_BLOCK_CI_SLOT_BYTES,
                "io_engine": "io_uring",
                "use_uring_cmd": True,
                "fdp_enabled": True,
                "fdp_placement_ids": [1, "x"],
            }
        )


def test_raw_block_fdp_empty_status_fails_startup():
    fake_core = _FakeFdpCore(status=[])
    config = _make_fdp_config()

    with pytest.raises(RuntimeError, match="no identifiers"):
        _make_fdp_adapter(fake_core, config)


def test_raw_block_fdp_status_reports_registered_nonzero_ids() -> None:
    fake_core = _FakeFdpCore(status=[(0, 10), (1, 11), (7, 17)])
    adapter = _make_fdp_adapter(fake_core, _make_fdp_config())
    try:
        status = adapter.report_status()
        assert status["fdp_enabled"] is True
        assert status["fdp_discovered_status"] == [(0, 10), (1, 11), (7, 17)]
        assert status["fdp_placement_ids"] == [1, 7]
    finally:
        adapter.close()


def test_raw_block_fdp_store_does_not_assign_placement_ids_yet() -> None:
    fake_core = _FakeFdpCore(status=[(0, 10), (1, 11), (7, 17)])
    adapter = _make_fdp_adapter(fake_core, _make_fdp_config())
    try:
        keys = [make_object_key(i) for i in range(4)]
        objects: list[Any] = [make_memory_obj(bytes([i + 1])) for i in range(4)]

        task_id = adapter.submit_store_task(keys, objects)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        result = adapter.pop_completed_store_tasks()[task_id]

        assert result.is_successful()
        assert fake_core.put_many_calls == [None]
    finally:
        adapter.close()


def test_raw_block_fdp_status_preserves_configured_id_order() -> None:
    fake_core = _FakeFdpCore(status=[(0, 10), (1, 11), (7, 17)])
    adapter = _make_fdp_adapter(
        fake_core,
        _make_fdp_config(placement_ids=[7, 1]),
    )
    try:
        status = adapter.report_status()
        assert status["fdp_placement_ids"] == [7, 1]
    finally:
        adapter.close()


def test_raw_block_fdp_rejects_user_placement_id_zero() -> None:
    with patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.logger.warning"
    ) as warning:
        with pytest.raises(ValueError, match="must not contain 0"):
            _make_fdp_config(placement_ids=[0, 1])

    warning.assert_called_once()
    assert "placement identifier 0" in warning.call_args.args[0]


def test_raw_block_fdp_user_ids_must_match_device_identifiers() -> None:
    fake_core = _FakeFdpCore(status=[(0, 10), (1, 11), (7, 17)])

    with pytest.raises(RuntimeError, match="does not match device"):
        _make_fdp_adapter(fake_core, _make_fdp_config(placement_ids=[1]))


def test_raw_block_fdp_requires_nonzero_device_ids() -> None:
    fake_core = _FakeFdpCore(status=[(0, 10)])

    with pytest.raises(RuntimeError, match="no non-zero identifiers"):
        _make_fdp_adapter(fake_core, _make_fdp_config())

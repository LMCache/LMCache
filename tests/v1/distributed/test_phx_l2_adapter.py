# SPDX-License-Identifier: Apache-2.0
"""
Tests for the Phoenix L2 adapter (POSIX fallback mode).

When ``phxcache`` is not installed or ``device_ids`` is not configured,
``PhxL2Adapter`` transparently falls back to POSIX read/write.  These
tests exercise that fallback path end-to-end (store → lookup → load)
plus config parsing, adapter registration, and lifecycle behavior — all
without requiring a GPU or the Phoenix kernel module.
"""

# Standard
import select
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.phx_l2_adapter import (
    PhxL2Adapter,
    PhxL2AdapterConfig,
)
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import (
    AdHocMemoryAllocator,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

_DEFAULT_SHAPE = torch.Size([2, 4, 8])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    kv_rank: int = 0,
    cache_salt: str = "",
) -> ObjectKey:
    """Build an ``ObjectKey`` with a simple integer chunk hash."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=kv_rank,
        cache_salt=cache_salt,
    )


def create_memory_obj(
    *,
    shape: torch.Size = _DEFAULT_SHAPE,
    dtype: torch.dtype = torch.bfloat16,
    fill_value: float = 0,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> MemoryObj:
    """Allocate a CPU ``MemoryObj`` filled with ``fill_value``."""
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate([shape], [dtype], fmt=fmt)
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(fill_value)
    return obj


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Block until *event_fd* is readable or *timeout* expires."""
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    consume_fd(event_fd)
    return True


def wait_for_condition(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll *predicate* until it returns ``True`` or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def make_posix_adapter(tmp_path) -> PhxL2Adapter:
    """Create a ``PhxL2Adapter`` in POSIX-fallback mode (no ``device_ids``).

    Without ``device_ids`` the adapter skips PhxCache/allocator init and
    uses POSIX ``read``/``write`` for all operations, allowing the tests
    to run on any CI machine without GPU or Phoenix hardware.
    """
    config = PhxL2AdapterConfig(base_path=str(tmp_path / "kv_cache"))
    return PhxL2Adapter(config)


def store_and_wait(adapter: PhxL2Adapter, key: ObjectKey, obj: MemoryObj) -> None:
    """Submit a single-key store task and wait for completion."""
    task_id = adapter.submit_store_task([key], [obj])
    assert wait_for_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert completed[task_id].is_successful()


def lookup_and_wait(adapter: PhxL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    """Submit a lookup task and return per-key hit booleans."""
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def load_and_wait(
    adapter: PhxL2Adapter,
    keys: list[ObjectKey],
    objs: list[MemoryObj],
) -> list[bool]:
    """Submit a load task and return per-key success booleans."""
    task_id = adapter.submit_load_task(keys, objs)
    assert wait_for_event_fd(adapter.get_load_event_fd())
    bitmap = adapter.query_load_result(task_id)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPhxL2AdapterConfig:
    def test_from_dict_with_all_fields(self):
        d = {
            "base_path": "/tmp/kv",
            "device_ids": [4, 5, 6, 7],
            "buffer_size_mb": 4096,
            "use_direct_io": False,
            "max_capacity_bytes": 1024,
            "perf_log_dir": "/tmp/perf",
        }
        config = PhxL2AdapterConfig.from_dict(d)
        assert config.base_path == "/tmp/kv"
        assert config.device_ids == [4, 5, 6, 7]
        assert config.buffer_size_mb == 4096
        assert config.use_direct_io is False
        assert config.max_capacity_bytes == 1024
        assert config.perf_log_dir == "/tmp/perf"

    def test_from_dict_defaults(self):
        config = PhxL2AdapterConfig.from_dict({"base_path": "/tmp/kv"})
        assert config.base_path == "/tmp/kv"
        assert config.device_ids is None
        assert config.buffer_size_mb == 2048
        assert config.use_direct_io is True
        assert config.max_capacity_bytes == 0
        assert config.perf_log_dir is None

    def test_from_dict_device_ids_as_strings(self):
        """``device_ids`` may arrive as strings from CLI/JSON parsing."""
        config = PhxL2AdapterConfig.from_dict(
            {"base_path": "/tmp/kv", "device_ids": ["4", "5"]}
        )
        assert config.device_ids == [4, 5]

    def test_to_dict_includes_type_and_required_fields(self):
        config = PhxL2AdapterConfig(
            base_path="/tmp/kv",
            device_ids=[4, 5],
            buffer_size_mb=4096,
        )
        d = config.to_dict()
        assert d["type"] == "phx"
        assert d["base_path"] == "/tmp/kv"
        assert d["device_ids"] == [4, 5]
        assert d["buffer_size_mb"] == 4096

    def test_to_dict_omits_optional_when_default(self):
        config = PhxL2AdapterConfig(base_path="/tmp/kv")
        d = config.to_dict()
        assert "device_ids" not in d
        assert "perf_log_dir" not in d

    def test_help_returns_descriptive_string(self):
        text = PhxL2AdapterConfig.help()
        assert isinstance(text, str)
        assert "base_path" in text
        assert "device_ids" in text


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_phx_adapter_registered():
    assert "phx" in get_registered_l2_adapter_types()


# ---------------------------------------------------------------------------
# Event-fd interface
# ---------------------------------------------------------------------------


def test_phx_adapter_has_distinct_eventfds(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    try:
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()
        assert len({store_fd, lookup_fd, load_fd}) == 3
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# POSIX fallback: store / lookup / load round-trip
# ---------------------------------------------------------------------------


def test_phx_posix_store_lookup_load_roundtrip(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=42)
    load_target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(10)

        store_and_wait(adapter, key, obj)
        assert lookup_and_wait(adapter, [key]) == [True]
        assert load_and_wait(adapter, [key], [load_target]) == [True]

        assert load_target.tensor is not None
        assert torch.all(load_target.tensor == 42)
    finally:
        obj.ref_count_down()
        load_target.ref_count_down()
        adapter.close()


def test_phx_posix_partial_hit(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj0 = create_memory_obj(fill_value=1)
    obj2 = create_memory_obj(fill_value=3)
    miss_target = create_memory_obj(fill_value=0)
    load_target0 = create_memory_obj(fill_value=0)
    load_target2 = create_memory_obj(fill_value=0)
    try:
        key0 = create_object_key(20)
        key1 = create_object_key(21)
        key2 = create_object_key(22)

        store_and_wait(adapter, key0, obj0)
        store_and_wait(adapter, key2, obj2)

        assert lookup_and_wait(adapter, [key0, key1, key2]) == [
            True,
            False,
            True,
        ]
        assert load_and_wait(
            adapter,
            [key0, key1, key2],
            [load_target0, miss_target, load_target2],
        ) == [True, False, True]

        assert load_target0.tensor is not None
        assert load_target2.tensor is not None
        assert miss_target.tensor is not None
        assert torch.all(load_target0.tensor == 1)
        assert torch.all(load_target2.tensor == 3)
        assert torch.all(miss_target.tensor == 0)
    finally:
        obj0.ref_count_down()
        obj2.ref_count_down()
        miss_target.ref_count_down()
        load_target0.ref_count_down()
        load_target2.ref_count_down()
        adapter.close()


def test_phx_store_multiple_keys(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    objs = [create_memory_obj(fill_value=i) for i in range(4)]
    try:
        keys = [create_object_key(100 + i) for i in range(4)]
        task_id = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[task_id].is_successful()
        assert lookup_and_wait(adapter, keys) == [True, True, True, True]
    finally:
        for o in objs:
            o.ref_count_down()
        adapter.close()


def test_phx_store_result_reports_bytes(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=7)
    try:
        key = create_object_key(40)
        task_id = adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        result = completed[task_id]
        assert result.is_successful()
        assert result.bytes_transferred() > 0
        # Second pop returns empty (results already consumed).
        assert adapter.pop_completed_store_tasks() == {}
    finally:
        obj.ref_count_down()
        adapter.close()


def test_phx_pop_completed_store_tasks_empty_after_consume(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=1)
    try:
        key = create_object_key(41)
        adapter.submit_store_task([key], [obj])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        _ = adapter.pop_completed_store_tasks()
        assert adapter.pop_completed_store_tasks() == {}
    finally:
        obj.ref_count_down()
        adapter.close()


# ---------------------------------------------------------------------------
# Lookup / load miss behavior
# ---------------------------------------------------------------------------


def test_phx_lookup_miss_for_nonexistent_key(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    try:
        key = create_object_key(200)
        assert lookup_and_wait(adapter, [key]) == [False]
    finally:
        adapter.close()


def test_phx_load_miss_does_not_modify_target(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    target = create_memory_obj(fill_value=99)
    try:
        key = create_object_key(300)
        assert load_and_wait(adapter, [key], [target]) == [False]
        assert target.tensor is not None
        assert torch.all(target.tensor == 99)
    finally:
        target.ref_count_down()
        adapter.close()


# ---------------------------------------------------------------------------
# One-shot query semantics
# ---------------------------------------------------------------------------


def test_phx_query_lookup_result_is_one_shot(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=1)
    try:
        key = create_object_key(60)
        store_and_wait(adapter, key, obj)
        task_id = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(task_id) is not None
        assert adapter.query_lookup_and_lock_result(task_id) is None
    finally:
        obj.ref_count_down()
        adapter.close()


def test_phx_query_load_result_is_one_shot(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=5)
    target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(50)
        store_and_wait(adapter, key, obj)
        task_id = adapter.submit_load_task([key], [target])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        assert adapter.query_load_result(task_id) is not None
        assert adapter.query_load_result(task_id) is None
    finally:
        obj.ref_count_down()
        target.ref_count_down()
        adapter.close()


# ---------------------------------------------------------------------------
# POSIX-mode specific behavior
# ---------------------------------------------------------------------------


def test_phx_posix_mode_not_phx_available(tmp_path):
    """Without ``device_ids`` the adapter reports PHX DMA as unavailable."""
    adapter = make_posix_adapter(tmp_path)
    try:
        assert adapter.is_phx_available() is False
    finally:
        adapter.close()


def test_phx_posix_pop_device_objs_empty(tmp_path):
    """POSIX load fills CPU objs directly; no device objs are produced."""
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=1)
    target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(30)
        store_and_wait(adapter, key, obj)
        task_id = adapter.submit_load_task([key], [target])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        _ = adapter.query_load_result(task_id)

        assert adapter.pop_loaded_device_objs(task_id) == {}
    finally:
        obj.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_phx_pop_device_objs_unknown_task_returns_empty(tmp_path):
    """Popping a task that never existed returns an empty dict."""
    adapter = make_posix_adapter(tmp_path)
    try:
        assert adapter.pop_loaded_device_objs(99999) == {}
    finally:
        adapter.close()


def test_phx_report_status(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    try:
        status = adapter.report_status()
        assert status["type"] == "phx"
        assert status["is_healthy"] is True
        assert status["dma_enabled"] is False
        assert "base_path" in status
        assert "hot_cache_size" in status
    finally:
        adapter.close()


def test_phx_report_status_hot_cache_grows_after_store(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    obj = create_memory_obj(fill_value=1)
    try:
        key = create_object_key(70)
        assert adapter.report_status()["hot_cache_size"] == 0
        store_and_wait(adapter, key, obj)
        assert adapter.report_status()["hot_cache_size"] == 1
    finally:
        obj.ref_count_down()
        adapter.close()


# ---------------------------------------------------------------------------
# Device routing logic
# ---------------------------------------------------------------------------


def test_phx_kv_rank_to_device(tmp_path):
    """``_kv_rank_to_device`` extracts ``global_rank`` from the packed
    ``kv_rank`` field (bits 16-23)."""
    adapter = make_posix_adapter(tmp_path)
    try:
        # kv_rank = (world_size << 24) | (global_rank << 16)
        #         | (local_world_size << 8) | local_rank
        kv_rank_0 = (8 << 24) | (0 << 16) | (8 << 8) | 0
        assert adapter._kv_rank_to_device(kv_rank_0) == 0

        kv_rank_3 = (8 << 24) | (3 << 16) | (8 << 8) | 3
        assert adapter._kv_rank_to_device(kv_rank_3) == 3

        kv_rank_7 = (8 << 24) | (7 << 16) | (8 << 8) | 7
        assert adapter._kv_rank_to_device(kv_rank_7) == 7
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_phx_adapter_close_stops_worker(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    adapter.close()
    assert not adapter._worker_thread.is_alive()


def test_phx_adapter_close_is_idempotent(tmp_path):
    adapter = make_posix_adapter(tmp_path)
    adapter.close()
    # A second close must not raise.
    adapter.close()


def test_phx_adapter_creates_base_path(tmp_path):
    """The adapter creates ``base_path`` on init if it does not exist."""
    base = tmp_path / "nested" / "kv_cache"
    assert not base.exists()
    config = PhxL2AdapterConfig(base_path=str(base))
    adapter = PhxL2Adapter(config)
    try:
        assert base.exists()
    finally:
        adapter.close()

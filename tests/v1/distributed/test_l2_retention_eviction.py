# SPDX-License-Identifier: Apache-2.0
"""L2 eviction must skip retained keys until their window expires.

Drives :class:`L2EvictionController` against a :class:`MockL2Adapter`
with a :class:`RetentionManager` injected, covering both the global-LRU
branch and the per-``cache_salt`` wipe branch.
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.retention_manager import RetentionManager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

OBJ_FLOATS = 128
OBJ_BYTES = OBJ_FLOATS * 4


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _key(chunk_id: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _memory_obj() -> TensorMemoryObj:
    raw = torch.empty(OBJ_FLOATS, dtype=torch.float32)
    raw.fill_(1.0)
    metadata = MemoryObjMetadata(
        shape=torch.Size([OBJ_FLOATS]),
        dtype=torch.float32,
        address=0,
        phy_size=OBJ_BYTES,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, metadata, parent_allocator=None)


def _wait_fd(fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    try:
        os.eventfd_read(fd)
    except BlockingIOError:
        pass
    return True


def _store_sync(adapter: MockL2Adapter, key: ObjectKey, obj: TensorMemoryObj):
    adapter.submit_store_task([key], [obj])
    assert _wait_fd(adapter.get_store_event_fd()), "store event timed out"
    adapter.pop_completed_store_tasks()


@pytest.fixture
def global_lru_setup():
    """Tiny-capacity adapter + global LRU policy + retention. Eviction
    ratio 1.0 so an over-watermark pass evicts every eligible key --
    whatever survives did so because retention vetoed it."""
    adapter = MockL2Adapter(
        MockL2AdapterConfig(
            max_size_gb=(8 * OBJ_BYTES) / (1024**3),
            mock_bandwidth_gb=10.0,
        )
    )
    policy = LRUEvictionPolicy()
    adapter.register_listener(L2EvictionPolicy(policy))

    state = L2AdapterEvictionState(
        adapter_id=0,
        adapter=adapter,
        eviction_config=EvictionConfig(
            eviction_policy="LRU",
            trigger_watermark=0.8,
            eviction_ratio=1.0,
        ),
    )
    state.eviction_policy = policy

    clock = FakeClock()
    retention = RetentionManager(max_retained_bytes=100 * OBJ_BYTES, clock=clock)
    controller = L2EvictionController([state], retention_manager=retention)
    yield adapter, controller, state, retention, clock
    adapter.close()


def test_retained_key_survives_over_watermark_cycle(global_lru_setup):
    adapter, controller, state, retention, _ = global_lru_setup

    for i in range(8):
        _store_sync(adapter, _key(i), _memory_obj())
    retention.note_stored([_key(0)], [OBJ_BYTES], ttl_sec=300)
    assert adapter.get_usage().usage_fraction >= 0.8

    controller._check_and_evict(state)

    assert _key(0) in adapter._memory_objects, "retained key was evicted"
    assert _key(1) not in adapter._memory_objects, "unretained keys must go"
    assert adapter.get_usage().total_bytes_used == OBJ_BYTES


def test_expired_key_rejoins_lru_pool(global_lru_setup):
    adapter, controller, state, retention, clock = global_lru_setup

    for i in range(8):
        _store_sync(adapter, _key(i), _memory_obj())
    retention.note_stored([_key(0)], [OBJ_BYTES], ttl_sec=300)

    clock.advance(301)
    # Expiry alone makes the key eligible; sweep only reconciles budget.
    controller._check_and_evict(state)
    assert _key(0) not in adapter._memory_objects

    assert retention.sweep() == 1
    assert retention.report_status()["retained_bytes"] == 0


def test_salt_wipe_respects_retention():
    """The unregistered-salt wipe (allowlist rule) must not delete keys
    inside an open retention window."""
    adapter = MockL2Adapter(
        MockL2AdapterConfig(max_size_gb=0.001, mock_bandwidth_gb=10.0)
    )
    policy = IsolatedLRUEvictionPolicy()
    adapter.register_listener(L2EvictionPolicy(policy))
    state = L2AdapterEvictionState(
        adapter_id=0,
        adapter=adapter,
        eviction_config=EvictionConfig(
            eviction_policy="IsolatedLRU",
            trigger_watermark=0.8,
            eviction_ratio=1.0,
        ),
    )
    state.eviction_policy = policy

    clock = FakeClock()
    retention = RetentionManager(max_retained_bytes=100 * OBJ_BYTES, clock=clock)
    controller = L2EvictionController(
        [state], quota_manager=QuotaManager(), retention_manager=retention
    )
    try:
        for i in range(4):
            _store_sync(adapter, _key(i, "alice"), _memory_obj())
        retention.note_stored([_key(0, "alice")], [OBJ_BYTES], ttl_sec=300)

        # alice has no quota entry -> effective limit 0 -> full wipe.
        controller._check_and_evict(state)

        assert _key(0, "alice") in adapter._memory_objects
        assert _key(1, "alice") not in adapter._memory_objects

        clock.advance(301)
        controller._check_and_evict(state)
        assert _key(0, "alice") not in adapter._memory_objects
    finally:
        adapter.close()


def test_report_status_retention_section(global_lru_setup):
    adapter, controller, state, retention, _ = global_lru_setup

    retention.note_stored([_key(0)], [OBJ_BYTES], ttl_sec=300)
    status = controller.report_status()
    assert status["retention"]["retained_keys"] == 1
    assert status["retention"]["retained_bytes"] == OBJ_BYTES

    bare = L2EvictionController([state])
    assert "retention" not in bare.report_status()

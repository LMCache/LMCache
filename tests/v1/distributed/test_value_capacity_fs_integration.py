# SPDX-License-Identifier: Apache-2.0
"""Real FS payload replacement through the native owner and store controller."""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.factory import create_l2_adapter_from_registry
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from tests.v1.distributed.test_local_value_policy import demand, key
from tests.v1.distributed.test_local_value_policy import runtime as local_runtime

runtime = local_runtime


def eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "native ownership transition timed out"
        time.sleep(0.01)


def native_result(query):
    deadline = time.monotonic() + 5
    while True:
        result = query()
        if result is not None:
            return result
        assert time.monotonic() < deadline, "native task did not complete"
        time.sleep(0.01)


@pytest.mark.parametrize("value_model", ["expected_reuses", "reuse_probability"])
def test_real_fs_replaces_low_value_payload_without_corrupting_retained_kv(
    runtime, tmp_path, monkeypatch, value_model
):
    pytest.importorskip("lmcache.lmcache_fs")
    bus, now, build = runtime
    policy = build(
        l2_capacity_policy="value",
        value_model=value_model,
        horizon_s=1.0,
        half_life_s=10.0,
        prefix_write_l2_ms=(0.1, 0.2),
    )
    layout = MemoryLayoutDesc(shapes=[torch.Size([2, 256, 8])], dtypes=[torch.bfloat16])
    block_bytes = 2 * 256 * 8 * 2
    manager = DramL1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=32 * 1024 * 1024,
                use_lazy=False,
                shm_name="",
            )
        )
    )
    config = FSNativeL2AdapterConfig(
        base_path=str(tmp_path / "cache"),
        max_capacity_gb=2 * block_bytes / 1024**3,
        num_workers=2,
        num_read_workers=0,
        require_empty=True,
    )
    adapter = create_l2_adapter_from_registry(config)
    adapter.register_listener(policy)
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_controllers.store_controller.get_event_bus",
        lambda: bus,
    )
    controller = StoreController(
        manager, [adapter], [L2AdapterDescriptor(0, config)], policy
    )

    def write(chunk, value):
        result = manager.reserve_write([key(chunk)], [False], layout)
        err, obj = result[key(chunk)]
        assert err == L1Error.SUCCESS and obj is not None
        obj.get_tensor(0).fill_(value)
        manager.finish_write([key(chunk)])
        eventually(lambda: key(chunk) in adapter.get_capacity_snapshot().resident_sizes)

    controller.start()
    try:
        for instant in [100.0, 101.0]:
            now[0] = instant
            demand(bus, policy, [b"cold"])
        write(b"cold", 11)
        for index in range(10):
            now[0] = 110 + index / 100
            demand(bus, policy, [b"hot"])
        write(b"hot", 17)
        for instant in [111.0, 112.0, 113.0]:
            now[0] = instant
            demand(bus, policy, [b"incoming"])
        write(b"incoming", 23)
        snapshot = adapter.get_capacity_snapshot()
        assert set(snapshot.resident_sizes) == {key(b"hot"), key(b"incoming")}
        assert (
            sum(snapshot.resident_sizes.values()) + snapshot.pending_bytes
            <= 2 * block_bytes
        )

        # Load actual retained FS bytes into independently reserved L1 buffers.
        for chunk, expected in [(b"hot", 17), (b"incoming", 23)]:
            read_key = key(chunk)
            lookup = adapter.submit_lookup_and_lock_task([read_key], {0: layout})
            found = native_result(
                lambda task=lookup: adapter.query_lookup_and_lock_result(task)
            )
            assert found.popcount() == 1
            output_key = key(b"restored-" + chunk)
            err, output = manager.reserve_write([output_key], [True], layout)[
                output_key
            ]
            assert err == L1Error.SUCCESS and output is not None
            output.get_tensor(0).zero_()
            load = adapter.submit_load_task([read_key], [output])
            loaded = native_result(lambda task=load: adapter.query_load_result(task))
            assert loaded.popcount() == 1
            assert torch.all(output.get_tensor(0) == expected)
            adapter.submit_unlock([read_key])
            manager.finish_write([output_key])
        assert bus.subscriber_exception_counts() == {}
    finally:
        controller.stop()
        adapter.close()
        manager.close()

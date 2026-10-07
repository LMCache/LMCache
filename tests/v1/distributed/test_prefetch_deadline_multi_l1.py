# SPDX-License-Identifier: Apache-2.0
"""Deadline fallback preserves multi-L1 affinity and read-lock ownership."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
import json
import uuid

# Third Party
from pytest_mock import MockerFixture
import pytest

# First Party
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from tests.v1.distributed.test_prefetch_controller import make_l1_config
from tests.v1.distributed.test_prefetch_deadline import (
    FakeClock,
    GatedLoadMockAdapter,
    in_flight_count,
    make_layout,
    make_object_key,
    result_ready,
    store_keys_in_l2,
    submit,
    wait_until,
)


@pytest.fixture
def managers() -> Iterator[list[L1Manager]]:
    """Use distinct real L1 pools for the default and archive affinities."""
    config = make_l1_config(8 * 1024 * 1024)
    pools = [
        L1Manager(
            replace(
                config,
                tag=tag,
                memory_config=replace(
                    config.memory_config,
                    shm_name=uuid.uuid4().hex[:12],
                ),
            )
        )
        for tag in ("_default", "archive")
    ]
    try:
        yield pools
    finally:
        for manager in pools:
            manager.close()


def _cli_args() -> list[str]:
    return [
        "--eviction-policy",
        "noop",
        *[
            value
            for tag in ("_default", "archive")
            for value in (
                "--l1-manager",
                json.dumps({"type": "DRAM", "tag": tag, "size_gb": 1}),
            )
        ],
    ]


@pytest.mark.parametrize("timeout", [None, 0.25])
def test_multi_l1_cli_preserves_deadline(timeout: float | None) -> None:
    args = _cli_args()
    if timeout is not None:
        args += ["--l2-prefetch-load-timeout", str(timeout)]
    config = parse_args(args)
    assert config.prefetch_load_timeout == timeout
    assert [manager.tag for manager in config.l1_manager_configs] == [
        "_default",
        "archive",
    ]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_multi_l1_cli_rejects_invalid_deadline(timeout: str) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        parse_args(_cli_args() + ["--l2-prefetch-load-timeout", timeout])


@pytest.mark.parametrize("policy, kept", [("prefix", [0]), ("full", [0, 2])])
def test_partial_deadline_keeps_affinity_and_drains_without_store(
    managers: list[L1Manager], policy: str, kept: list[int], mocker: MockerFixture
) -> None:
    configs = [MockL2AdapterConfig(0.05, 10.0) for _ in managers]
    for config, manager in zip(configs, managers, strict=True):
        config.affinity_tag = manager.config.tag
    adapters = [GatedLoadMockAdapter(config) for config in configs]
    slow, fast = adapters
    descriptors = [
        L2AdapterDescriptor(index=index, config=config)
        for index, config in enumerate(configs)
    ]
    stores = [
        StoreController(manager, [adapter], [descriptor], DefaultStorePolicy())
        for manager, adapter, descriptor in zip(
            managers, adapters, descriptors, strict=True
        )
    ]
    clock = FakeClock()
    controller = PrefetchController(
        l1_managers=managers,
        l1_manager_descriptors=[
            L1ManagerDescriptor(index=manager.l1_manager_id, config=manager.config)
            for manager in managers
        ],
        l2_adapters=list(adapters),
        adapter_descriptors=descriptors,
        policy=RetainPrefetchPolicy(),
        l2_load_timeout=0.5,
        clock=clock,
    )
    for store in stores:
        store.start()
    controller.start()
    try:
        layout = make_layout()
        keys = [make_object_key(index) for index in range(3)]
        store_keys_in_l2(slow, [keys[1]], layout)
        store_keys_in_l2(fast, [keys[0], keys[2]], layout)
        store_calls = [mocker.spy(adapter, "submit_store_task") for adapter in adapters]
        fast.release_loads()
        request_id = submit(controller, keys, layout, policy)
        assert slow.load_entered.wait(10)
        assert fast.load_result_consumed.wait(10)
        clock.advance(1)
        assert wait_until(lambda: result_ready(controller, request_id))
        result = controller.query_prefetch_result(request_id)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == kept
        assert result.l1_owners == {
            keys[index]: managers[1].l1_manager_id for index in kept
        }
        assert managers[0].report_status()["staging_object_count"] == 1
        assert managers[1].report_status()["read_locked_count"] == len(kept)
        assert all(managers[0].get_object_state(keys[index]) is None for index in kept)

        # A second caller holds one of the same keys while the timed-out
        # request drains. Drain cleanup must not consume either caller's lock.
        managers[1].reserve_read([keys[0]])
        slow.release_loads()
        assert wait_until(lambda: in_flight_count(controller) == 0)
        assert controller.query_prefetch_result(request_id) is None
        assert controller.report_status()["deadline_timeout_count"] == 1
        assert all(adapter.debug_locked_key_count() == 0 for adapter in adapters)
        assert all(
            manager.report_status()["staging_object_count"] == 0 for manager in managers
        )
        assert managers[0].get_object_state(keys[1]) is not None
        assert managers[0].report_status()["read_locked_count"] == 0
        assert managers[1].report_status()["read_locked_count"] == len(kept)
        managers[1].finish_read([keys[index] for index in kept])
        assert managers[1].report_status()["read_locked_count"] == 1
        managers[1].finish_read([keys[0]])
        assert managers[1].report_status()["read_locked_count"] == 0
        for recorder, store in zip(store_calls, stores, strict=True):
            recorder.assert_not_called()
            assert store.report_status()["pending_keys_count"] == 0
        assert all(manager.memcheck() for manager in managers)
    finally:
        slow.release_loads()
        fast.release_loads()
        controller.stop()
        for store in stores:
            store.stop()
        for adapter in adapters:
            adapter.close()


def test_queued_deadline_reports_owners_from_both_l1s(
    managers: list[L1Manager],
) -> None:
    config = MockL2AdapterConfig(0.05, 10.0)
    adapter = GatedLoadMockAdapter(config)
    clock = FakeClock()
    controller = PrefetchController(
        l1_managers=managers,
        l1_manager_descriptors=[
            L1ManagerDescriptor(index=manager.l1_manager_id, config=manager.config)
            for manager in managers
        ],
        l2_adapters=[adapter],
        adapter_descriptors=[L2AdapterDescriptor(index=0, config=config)],
        policy=RetainPrefetchPolicy(),
        max_in_flight=1,
        l2_load_timeout=0.5,
        clock=clock,
    )
    controller.start()
    try:
        layout = make_layout()
        keys = [make_object_key(index) for index in range(2)]
        for manager, key in zip(managers, keys, strict=True):
            manager.reserve_write([key], [False], layout)
            manager.finish_write([key])
        busy_key = make_object_key(100)
        store_keys_in_l2(adapter, [busy_key], layout)
        busy_request = submit(controller, [busy_key], layout)
        assert adapter.load_entered.wait(10)
        queued_request = submit(controller, keys, layout)
        assert wait_until(lambda: controller.report_status()["pending_queue_size"] == 1)
        clock.advance(1)
        assert wait_until(lambda: result_ready(controller, queued_request))
        result = controller.query_prefetch_result(queued_request)
        assert result is not None and result.l1_hit_count == 2
        assert result.l2_hit_count == 0
        assert result.l1_owners == {
            key: manager.l1_manager_id
            for manager, key in zip(managers, keys, strict=True)
        }
        assert all(
            manager.report_status()["read_locked_count"] == 1 for manager in managers
        )
        adapter.release_loads()
        assert wait_until(lambda: in_flight_count(controller) == 0)
        assert all(
            manager.report_status()["read_locked_count"] == 1 for manager in managers
        )
        assert controller.query_prefetch_result(busy_request) is not None
        for manager, key in zip(managers, keys, strict=True):
            manager.finish_read([key])
            assert manager.report_status()["read_locked_count"] == 0
        assert controller.query_prefetch_result(queued_request) is None
        assert adapter.debug_locked_key_count() == 0
    finally:
        adapter.release_loads()
        controller.stop()
        adapter.close()

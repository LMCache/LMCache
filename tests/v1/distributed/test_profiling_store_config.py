# SPDX-License-Identifier: Apache-2.0
"""Orthogonal store profiling at the parser and constructor boundaries."""

# Standard
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import Mock
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.config import StorageManagerConfig, parse_args
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy.arc import ARCEvictionPolicy
from lmcache.v1.distributed.eviction_policy.local_value import (
    LocalValueConfig,
    LocalValuePolicy,
    model_revision,
)
from lmcache.v1.distributed.eviction_policy.noop import NoOpEvictionPolicy
from lmcache.v1.distributed.storage_controllers.store_policy import StorePolicy
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
import lmcache.v1.distributed.storage_manager as sm


def arguments(policy: str = "ARC", path: str = "", json_l1: bool = False) -> list[str]:
    l1 = (
        ["--l1-manager", json.dumps({"type": "DRAM", "tag": "cpu", "size_gb": 0.01})]
        if json_l1
        else ["--l1-size-gb", "0.01"]
    )
    return (
        l1
        + ["--eviction-policy", policy]
        + (["--profiling-store-config", path] if path else [])
    )


@pytest.mark.parametrize("json_l1", [False, True])
def test_store_config_is_independent_in_all_parser_paths(json_l1: bool) -> None:
    parsed = parse_args(
        arguments(path="store.json", json_l1=json_l1)
        + ["--profiling-config", "l1.json"]
    )
    assert parsed.profiling_store_config == "store.json"
    assert parsed.eviction_config.eviction_policy == "ARC"
    assert parsed.eviction_config.profiling_config == "l1.json"
    assert asdict(parsed)["profiling_store_config"] == "store.json"
    assert parse_args(arguments(json_l1=json_l1)).profiling_store_config == ""


@pytest.fixture
def construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[SimpleNamespace]:
    bus = EventBus(EventBusConfig(enabled=True))
    subscribers = []
    register = bus.register_subscriber

    def subscribe(policy: Any) -> None:
        subscribers.append(policy)
        register(policy)

    monkeypatch.setattr(bus, "register_subscriber", subscribe)
    monkeypatch.setattr(sm, "get_event_bus", lambda: bus)
    monkeypatch.setattr(
        "lmcache.v1.mp_observability.event_bus.get_event_bus", lambda: bus
    )
    monkeypatch.setattr(
        "lmcache.v1.mp_observability.event_bus.is_observability_enabled", lambda: True
    )
    # The policy/capability interfaces are supplied by the other owners.
    monkeypatch.setattr(StorePolicy, "requires_value_capacity", False, raising=False)
    manager = Mock(l1_manager_id=0)
    manager.get_memory_tags.return_value = MemoryTags(
        MemorySegment.DRAM, MemoryResidency.LOCAL
    )
    adapter = Mock(supports_value_capacity=False, supports_global_eviction=True)
    policies, states = [], []
    stores = []

    def l1_controller(manager: Mock, config: Any, policy: Any) -> Mock:
        policies.append(policy)
        manager.register_listener(L1EvictionPolicy(policy))
        return Mock()

    def store_controller(
        manager: Mock, adapters: list, descriptors: list, policy: StorePolicy
    ) -> Mock:
        stores.append(policy)
        assert not subscribers
        return Mock()

    def l2_controller(adapter_states: list, **kwargs: Any) -> Mock:
        states.extend(adapter_states)
        return Mock()

    def managers(configs: list) -> list[Mock]:
        manager.config = configs[0]
        return [manager]

    monkeypatch.setattr(sm, "create_l1_managers", managers)
    monkeypatch.setattr(
        StorageManager,
        "_build_l2_adapter",
        lambda self, ac: (0, adapter, SimpleNamespace(index=0, config=ac)),
    )
    monkeypatch.setattr(sm, "L1EvictionController", l1_controller)
    monkeypatch.setattr(sm, "L2EvictionController", l2_controller)
    monkeypatch.setattr(sm, "StoreController", store_controller)
    monkeypatch.setattr(sm, "PrefetchController", Mock())
    monkeypatch.setattr(sm, "PeriodicEventNotifier", Mock())
    monkeypatch.setattr(sm, "register_gauge", Mock())
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    config = LocalValueConfig(
        model_name=str(model),
        chunk_size=256,
        prefix_recompute_ms=(10.0,),
        prefix_fetch_l1_ms=(1.0,),
        prefix_fetch_l2_ms=(4.0,),
        prefix_write_l2_ms=(2.0,),
        model_revision=model_revision(model),
        dtypes=("torch.bfloat16",),
        shapes=((2, 256, 8),),
    )
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"schema_version": 2, **asdict(config)}))
    yield SimpleNamespace(
        bus=bus,
        subscribers=subscribers,
        manager=manager,
        adapter=adapter,
        policies=policies,
        stores=stores,
        states=states,
        path=str(path),
    )
    bus.stop()


def configured(path: str = "", policy: str = "ARC") -> StorageManagerConfig:
    return parse_args(
        arguments(policy, path)
        + [
            "--l2-adapter",
            json.dumps(
                {
                    "type": "mock",
                    "max_size_gb": 0.01,
                    "mock_bandwidth_gb": 10.0,
                    "eviction": {"eviction_policy": "LRU"},
                }
            ),
        ]
    )


@pytest.mark.parametrize("shared", [False, True])
def test_commit_subscribes_one_store_policy_and_preserves_l1(
    construction: SimpleNamespace, shared: bool
) -> None:
    env = construction
    config = configured(
        "" if shared else env.path, "ProfiledValue" if shared else "ARC"
    )
    if shared:
        config.eviction_config.profiling_config = env.path
    subject = StorageManager(config)
    try:
        assert isinstance(env.stores[0], LocalValuePolicy)
        assert env.subscribers == env.stores
        assert env.bus.has_subscribers(EventType.MP_LOOKUP)
        if shared:
            assert env.policies[0] is env.stores[0]
        else:
            assert isinstance(env.policies[0], ARCEvictionPolicy)
            assert env.policies[0] is not env.stores[0]
    finally:
        subject.close()


def test_arc_alone_has_no_profiling_subscriber(construction: SimpleNamespace) -> None:
    env = construction
    subject = StorageManager(configured())
    try:
        assert isinstance(env.policies[0], ARCEvictionPolicy)
        assert not isinstance(env.stores[0], LocalValuePolicy)
        assert env.subscribers == []
        assert not isinstance(env.states[0].eviction_policy, NoOpEvictionPolicy)
    finally:
        subject.close()


def test_ambiguous_shared_and_standalone_rejected_before_allocating(
    construction: SimpleNamespace,
) -> None:
    env = construction
    with pytest.raises(ValueError, match="profiling-store-config"):
        StorageManager(configured(env.path, "ProfiledValue"))
    env.manager.close.assert_not_called()
    assert env.policies == env.subscribers == []


@pytest.mark.parametrize("invalid", ["no_l2", "two_l2", "two_l1", "store"])
def test_standalone_fixed_topology_rejected_before_allocating(
    construction: SimpleNamespace, invalid: str
) -> None:
    env = construction
    config = configured(env.path)
    if invalid == "no_l2":
        config.l2_adapter_config.adapters.clear()
    elif invalid == "two_l2":
        config.l2_adapter_config.adapters *= 2
    elif invalid == "two_l1":
        config.l1_manager_configs *= 2
    else:
        config.store_policy = "skip_l1"
    with pytest.raises(ValueError, match="one L1"):
        StorageManager(config)
    assert env.policies == env.subscribers == []


def test_failed_constructor_does_not_subscribe_and_closes_owners(
    construction: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = construction
    monkeypatch.setattr(
        sm, "PrefetchController", Mock(side_effect=RuntimeError("prefetch"))
    )
    with pytest.raises(RuntimeError, match="prefetch"):
        StorageManager(configured(env.path))
    assert env.subscribers == []
    env.adapter.close.assert_called_once()
    env.manager.close.assert_called_once()


@pytest.mark.parametrize("failure", ["missing", "malformed", "disabled"])
def test_standalone_calibration_failure_closes_l1_without_subscription(
    construction: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    env = construction
    if failure == "disabled":
        monkeypatch.setattr(
            "lmcache.v1.mp_observability.event_bus.is_observability_enabled",
            lambda: False,
        )
        error = ValueError
    elif failure == "malformed":
        Path(env.path).write_text("not json")
        error = ValueError
    else:
        env.path += ".missing"
        error = FileNotFoundError
    with pytest.raises(error):
        StorageManager(configured(env.path))
    env.manager.close.assert_called_once()
    assert env.subscribers == []
    env.adapter.close.assert_not_called()


def test_standalone_l1_lifecycle_preserves_original_admission_scale(
    construction: SimpleNamespace,
) -> None:
    env = construction
    config = configured(env.path)
    subject = StorageManager(config)
    reference = LocalValuePolicy(
        LocalValueConfig.load(env.path), env.bus, register=False
    )
    try:
        standalone = env.stores[0]
        costs = LocalValueConfig.load(env.path)
        chunks = [b"hot"] + [bytes([i]) for i in range(8)]
        keys = [
            ObjectKey(
                chunk_hash=c,
                model_name=costs.model_name,
                kv_rank=ObjectKey.ComputeKVRank(1, 0, 1, 0),
            )
            for c in chunks
        ]
        for chunk in [b"hot", *chunks]:
            event = Event(
                event_type=EventType.MP_LOOKUP,
                metadata={
                    "model_name": costs.model_name,
                    "chunk_size": 256,
                    "chunk_hashes": [chunk],
                    "dtypes": ["torch.bfloat16"],
                    "shapes": [[2, 256, 8]],
                    "cache_salt": "",
                    "world_size": 1,
                    "tags": {},
                },
            )
            for policy in (reference, standalone):
                policy.get_subscriptions()[EventType.MP_LOOKUP](event)
        listeners = [
            call.args[0] for call in env.manager.register_listener.call_args_list
        ]
        assert len(env.policies) == 1
        for listener in listeners:
            listener.on_l1_keys_write_finished(keys)
            listener.on_l1_keys_accessed(keys)
        reference.on_keys_created(keys)
        descriptors = [
            L2AdapterDescriptor(index=0, config=config.l2_adapter_config.adapters[0])
        ]
        expected = reference.select_store_targets([keys[0]], descriptors)
        assert expected == {0: [keys[0]]}, "fixture must distinguish resident scale"
        assert standalone.select_store_targets([keys[0]], descriptors) == expected
        for listener in listeners:
            listener.on_l1_keys_deleted_by_manager(keys[1:])
        reference.on_keys_removed(keys[1:])
        assert reference.select_store_targets([keys[0]], descriptors) == {}
        assert standalone.select_store_targets([keys[0]], descriptors) == {}
    finally:
        reference.shutdown()
        subject.close()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_value_capacity_capability_and_no_competing_lru(
    construction: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
    supported: bool,
) -> None:
    env = construction
    monkeypatch.setattr(
        LocalValuePolicy, "requires_value_capacity", True, raising=False
    )
    env.adapter.supports_value_capacity = supported
    config = configured(
        "" if shared else env.path, "ProfiledValue" if shared else "ARC"
    )
    if shared:
        config.eviction_config.profiling_config = env.path
    if not supported:
        with pytest.raises(ValueError, match="value.capacity"):
            StorageManager(config)
        assert env.subscribers == []
        env.adapter.close.assert_called_once()
        env.manager.close.assert_called_once()
        return
    subject = StorageManager(config)
    try:
        assert isinstance(env.states[0].eviction_policy, NoOpEvictionPolicy)
        assert (
            config.l2_adapter_config.adapters[0].eviction_config.eviction_policy
            == "LRU"
        )
    finally:
        subject.close()

# SPDX-License-Identifier: Apache-2.0
"""Public demand/eviction contracts, including distinct-distance negatives."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig, parse_args
from lmcache.v1.distributed.eviction_policy.factory import CreateEvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.profiled import (
    ProfileConfig,
    ProfiledEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import EvictionDestination
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig


@pytest.fixture
def bus():
    value = EventBus(EventBusConfig(enabled=True))
    value.start()
    yield value
    value.stop()


def key(chunk: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk,
        model_name="model",
        kv_rank=ObjectKey.ComputeKVRank(1, 0, 1, 0),
    )


def observe(bus: EventBus, policy: ProfiledEvictionPolicy, chunks: list[bytes]) -> None:
    target = policy.statistics()["requests"] + 1
    bus.publish(
        Event(
            event_type=EventType.MP_LOOKUP,
            metadata={
                "model_name": "model",
                "chunk_size": 256,
                "chunk_hashes": chunks,
            },
        )
    )
    deadline = time.monotonic() + 3
    while policy.statistics()["requests"] < target:
        if time.monotonic() >= deadline:
            pytest.fail("real EventBus did not deliver demand observation")
        time.sleep(0.001)


def delivered(policy: ProfiledEvictionPolicy, requests: int) -> None:
    deadline = time.monotonic() + 3
    while policy.statistics()["requests"] < requests:
        assert time.monotonic() < deadline, "subscriber did not drain"
        time.sleep(0.001)


def test_distinct_reuse_distance_is_not_access_gap(bus):
    policy = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0,)), bus)
    observe(bus, policy, [b"A"])
    for _ in range(8):
        observe(bus, policy, [b"B"])
    observe(bus, policy, [b"A"])
    assert policy.profile(b"A")["reuse_distance"] == 1
    assert policy.profile(b"A")["frequency"] == 2


def test_frequent_old_key_survives_scan_but_lru_does_not(bus):
    policy = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0,)), bus)
    lru = LRUEvictionPolicy()
    for _ in range(12):
        observe(bus, policy, [b"hot"])
    for chunk in (b"hot", b"scan1", b"scan2"):
        policy.on_keys_created([key(chunk)])
        lru.on_keys_created([key(chunk)])
        if chunk != b"hot":
            observe(bus, policy, [chunk])
    assert lru.get_eviction_actions(0.2)[0].keys == [key(b"hot")]
    assert policy.get_eviction_actions(0.2)[0].keys == [key(b"scan1")]


def test_measured_cost_changes_retention_decision(bus):
    # Equal observed frequencies, but different whole-prefix costs/lengths.
    full = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0, 40.0)), bus)
    ablated = ProfiledEvictionPolicy(
        ProfileConfig("model", 256, (1.0, 40.0), mode="no-cost"), bus
    )
    # Both subscribers receive the same demand, so wait on the later one too.
    observe(bus, full, [b"expensive0", b"expensive1"])
    observe(bus, ablated, [b"cheap"])
    delivered(full, 2)
    delivered(ablated, 2)
    for policy in (full, ablated):
        policy.on_keys_created([key(b"expensive0"), key(b"expensive1")])
        policy.on_keys_created([key(b"cheap")])
    assert full.get_eviction_actions(0.2)[0].keys == [key(b"cheap")]
    assert ablated.get_eviction_actions(0.2)[0].keys == [key(b"expensive1")]


@pytest.mark.parametrize("mode", ["no-frequency", "no-distance"])
def test_each_demand_signal_changes_actual_selection(bus, mode):
    full = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0,)), bus)
    ablated = ProfiledEvictionPolicy(
        ProfileConfig("model", 256, (1.0,), mode=mode), bus
    )
    sequence = (
        [b"A"] * 12 + [b"B"]
        if mode == "no-frequency"
        else [b"B"] + [str(i).encode() for i in range(10)] + [b"B", b"A", b"A"]
    )
    for chunk in sequence:
        observe(bus, full, [chunk])
    delivered(ablated, len(sequence))
    for policy in (full, ablated):
        policy.on_keys_created([key(b"A")])
        policy.on_keys_created([key(b"B")])
    assert full.get_eviction_actions(0.5)[0].keys == [key(b"B")]
    assert ablated.get_eviction_actions(0.5)[0].keys == [key(b"A")]


def test_prefix_suffix_first_and_orphaned_suffix_has_no_value(bus):
    policy = ProfiledEvictionPolicy(ProfileConfig("model", 256, (2.0, 4.0)), bus)
    observe(bus, policy, [b"A", b"B"])
    policy.on_keys_created([key(b"A"), key(b"B")])
    assert policy.get_eviction_actions(0.5)[0].keys == [key(b"B")]
    policy.on_keys_removed([key(b"A")])
    assert policy.get_eviction_actions(1)[0].keys == [key(b"B")]
    assert policy.profile(b"A") is not None  # key-only history is not residency


def test_selection_does_not_remove_residents_and_filter_is_reentrant(bus):
    policy = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0,)), bus)
    policy.on_keys_reserved([key(b"A"), key(b"B")])

    def eligible(candidate):
        policy.statistics()  # Must not deadlock on a callback under policy lock.
        return candidate != key(b"B")

    action = policy.get_eviction_actions(1, eligible)[0]
    assert action.keys == [key(b"A")]
    assert policy.get_eviction_actions(1)[0].keys == [key(b"B"), key(b"A")]
    policy.on_keys_removed(action.keys)
    assert policy.get_eviction_actions(1)[0].keys == [key(b"B")]


def test_observe_mode_preserves_lru_actions(bus):
    policy = ProfiledEvictionPolicy(
        ProfileConfig("model", 256, (1.0,), mode="observe"), bus
    )
    lru = LRUEvictionPolicy()
    for current in (policy, lru):
        current.on_keys_reserved([key(b"A"), key(b"B")])
        current.on_keys_created([key(b"A"), key(b"B")])
        current.on_keys_touched([key(b"B")])
        current.register_eviction_destination(EvictionDestination.DISCARD)
    for _ in range(20):
        observe(bus, policy, [b"A"])
    for ratio in (0, 0.5, 1):
        assert policy.get_eviction_actions(ratio) == lru.get_eviction_actions(ratio)


def test_history_bound_and_forgetting_make_key_cold(bus):
    policy = ProfiledEvictionPolicy(
        ProfileConfig("model", 256, (1.0,), history_keys=2), bus
    )
    for chunk in (b"A", b"B", b"C", b"A"):
        observe(bus, policy, [chunk])
        assert policy.statistics()["history_keys"] <= 2
    assert policy.profile(b"A")["frequency"] == 1
    assert policy.profile(b"B") is None
    policy.shutdown()
    assert policy.statistics()["history_keys"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_size", True),
        ("history_keys", 0),
        ("prefix_recompute_ms", (float("nan"),)),
        ("prefix_recompute_ms", (True,)),
        ("prefix_recompute_ms", (0.0,)),
        ("mode", "oracle"),
        ("model_name", 1),
    ],
)
def test_calibration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(ProfileConfig("model", 256, (1.0,)), **{field: value})


def test_factory_requires_measured_config_and_live_observability():
    with pytest.raises(ValueError, match="requires"):
        CreateEvictionPolicy(EvictionConfig("Profiled"))
    assert isinstance(CreateEvictionPolicy(EvictionConfig("LRU")), LRUEvictionPolicy)


def test_load_calibration_round_trip_and_unknown_schema(tmp_path):
    source = tmp_path / "calibration.json"
    record = {
        "schema_version": 1,
        "model_name": "model",
        "chunk_size": 256,
        "prefix_recompute_ms": [1.0, 4.0],
    }
    source.write_text(json.dumps(record))
    assert ProfileConfig.load(str(source)).prefix_recompute_ms == (1.0, 4.0)
    record["schema_version"] = True
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        ProfileConfig.load(str(source))


def test_cli_populates_profile_config_and_default_stays_lru(tmp_path):
    path = tmp_path / "profile.json"
    config = parse_args(
        [
            "--l1-size-gb",
            "0.5",
            "--eviction-policy",
            "Profiled",
            "--profiling-config",
            str(path),
        ]
    )
    assert config.eviction_config.eviction_policy == "Profiled"
    assert config.eviction_config.profiling_config == str(path)
    stock = parse_args(["--l1-size-gb", "0.5", "--eviction-policy", "LRU"])
    assert stock.eviction_config.profiling_config == ""


def test_invalid_demand_and_inconsistent_ancestry_do_not_change_profile(bus):
    policy = ProfiledEvictionPolicy(ProfileConfig("model", 256, (1.0, 2.0)), bus)
    observe(bus, policy, [b"A", b"B"])
    for metadata in (
        {"model_name": "other", "chunk_size": 256, "chunk_hashes": [b"A"]},
        {"model_name": "model", "chunk_size": 512, "chunk_hashes": [b"A"]},
        {"model_name": "model", "chunk_size": 256, "chunk_hashes": [b"A", b"A"]},
        {"model_name": "model", "chunk_size": 256, "chunk_hashes": [b"B"]},
    ):
        bus.publish(Event(event_type=EventType.MP_LOOKUP, metadata=metadata))
    # This valid event is a barrier behind the rejected events on the drain thread.
    observe(bus, policy, [b"C"])
    assert policy.statistics()["requests"] == 2
    assert policy.profile(b"A")["frequency"] == 1
    assert policy.profile(b"B")["position"] == 1


def test_event_updates_overlap_residency_and_reentrant_selection(bus):
    policy = ProfiledEvictionPolicy(
        ProfileConfig("model", 256, (1.0,), history_keys=32), bus
    )

    def churn():
        for index in range(300):
            current = key(str(index % 40).encode())
            policy.on_keys_created([current])
            policy.on_keys_touched([current])
            policy.get_eviction_actions(0.5, lambda _: bool(policy.statistics()))
            policy.on_keys_removed([current])

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(churn)
        for index in range(100):
            observe(bus, policy, [str(index % 40).encode()])
        future.result(timeout=10)
    assert policy.statistics()["requests"] == 100
    assert policy.statistics()["history_keys"] == 32
    assert bus.subscriber_exception_counts() == {}
    assert policy.get_eviction_actions(1) == []

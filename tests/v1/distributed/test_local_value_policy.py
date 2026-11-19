# SPDX-License-Identifier: Apache-2.0
"""Local cost decisions must reach the real policy entry points."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
import importlib
import json
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.distributed.eviction_policy.factory import CreateEvictionPolicy
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.reconfiguration import L2ReconfigureError
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig


def make_config(**changes):
    module = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    config = module.LocalValueConfig(
        model_name="model",
        chunk_size=256,
        prefix_recompute_ms=(10.0, 20.0),
        prefix_fetch_l1_ms=(1.0, 2.0),
        prefix_fetch_l2_ms=(4.0, 8.0),
        prefix_write_l2_ms=(2.0, 4.0),
        model_revision="0" * 64,
        dtypes=("torch.bfloat16",),
        shapes=((2, 256, 8),),
    )
    return replace(config, **changes)


def key(value):
    return ObjectKey(
        chunk_hash=value,
        model_name="model",
        kv_rank=ObjectKey.ComputeKVRank(1, 0, 1, 0),
    )


@pytest.fixture
def runtime():
    module = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    bus = EventBus(EventBusConfig(enabled=True))
    bus.start()
    now = [100.0]
    policies = []

    def build(**changes):
        policy = module.LocalValuePolicy(
            make_config(**changes), bus, clock=lambda: now[0]
        )
        policies.append(policy)
        return policy

    yield bus, now, build
    bus.stop()


def demand(bus, policy, chunks):
    count = policy.statistics()["requests"]
    bus.publish(
        Event(
            event_type=EventType.MP_LOOKUP,
            metadata={
                "model_name": "model",
                "chunk_size": 256,
                "chunk_hashes": chunks,
                "dtypes": ["torch.bfloat16"],
                "shapes": [[2, 256, 8]],
                "cache_salt": "",
                "world_size": 1,
                "tags": {},
            },
        )
    )
    deadline = time.monotonic() + 3
    while policy.statistics()["requests"] == count:
        assert time.monotonic() < deadline
        time.sleep(0.001)


def adapters():
    return [L2AdapterDescriptor(index=0, config=MockL2AdapterConfig(0.01, 10.0))]


def capacity(residents, *, pending=0, unavailable=(), limit=2):
    # First Party
    from lmcache.v1.distributed.l2_adapters.base import CapacitySnapshot

    return CapacitySnapshot(
        resident_sizes={key(chunk): 1 for chunk in residents},
        unavailable=frozenset(key(chunk) for chunk in unavailable),
        pending_bytes=pending,
        capacity_bytes=limit,
    )


def test_capacity_uses_lowest_resident_value_not_a_fixed_charge(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value")
    for _ in range(3):
        demand(bus, policy, [b"incoming"])
    demand(bus, policy, [b"cold"])
    for _ in range(8):
        demand(bus, policy, [b"hot"])
    plan = policy.plan_capacity_admission(
        [key(b"incoming")], 1, capacity([b"cold", b"hot"])
    )
    assert plan.admitted_keys == (key(b"incoming"),)
    assert plan.victim_keys == (key(b"cold"),)
    stronger = policy.plan_capacity_admission(
        [key(b"incoming")], 1, capacity([b"hot"], limit=1)
    )
    assert stronger.admitted_keys == stronger.victim_keys == ()


def test_probability_capacity_replaces_positive_value_not_only_cold_blocks(runtime):
    bus, now, build = runtime
    policy = build(
        l2_capacity_policy="value",
        value_model="reuse_probability",
        horizon_s=1.0,
        half_life_s=10.0,
        prefix_write_l2_ms=(0.1, 0.2),
    )
    for instant in [100.0, 101.0]:
        now[0] = instant
        demand(bus, policy, [b"cold"])
    for index in range(10):
        now[0] = 110 + index / 100
        demand(bus, policy, [b"hot"])
    for instant in [111.0, 112.0, 113.0]:
        now[0] = instant
        demand(bus, policy, [b"incoming"])
    plan = policy.plan_capacity_admission(
        [key(b"incoming")], 1, capacity([b"cold", b"hot"])
    )
    assert plan.admitted_keys == (key(b"incoming"),)
    assert plan.victim_keys == (key(b"cold"),)
    # Same incoming value must not displace a high-value resident instead.
    assert (
        policy.plan_capacity_admission(
            [key(b"incoming")], 1, capacity([b"hot"], limit=1)
        ).admitted_keys
        == ()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"l2_capacity_policy": "unknown"},
        {"l2_capacity_policy": "value", "mode": "observe"},
    ],
)
def test_capacity_configuration_rejects_unsupported_modes(changes):
    with pytest.raises(ValueError):
        make_config(**changes)


def test_capacity_batch_does_not_overbook_or_replace_its_own_admissions(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value")
    for _ in range(5):
        demand(bus, policy, [b"A"])
        demand(bus, policy, [b"B"])
    plan = policy.plan_capacity_admission(
        [key(b"A"), key(b"B")], 1, capacity([], limit=1)
    )
    assert len(plan.admitted_keys) == 1
    assert plan.victim_keys == ()


def test_store_statistics_distinguish_fallback_rejection_and_real_ranking(runtime):
    bus, _, build = runtime
    policy = build()
    demand(bus, policy, [b"cold"])
    policy.select_store_targets([key(b"cold"), key(b"unknown")], adapters())
    stats = policy.store_statistics()
    assert stats["store_evaluated_keys"] == 1
    assert stats["store_admitted_keys"] == 0
    assert stats["store_rejected_keys"] == 1
    assert stats["store_fallback_keys"] == 1
    assert stats["ranked_decisions"] == 0
    policy.on_keys_created([key(b"cold")])
    policy.get_eviction_actions(1.0)
    assert policy.store_statistics()["ranked_decisions"] == 1
    policy.on_keys_created([key(b"unknown")])
    policy.get_eviction_actions(1.0)
    assert policy.store_statistics()["ranked_decisions"] == 1


def test_capacity_charges_pending_bytes_and_never_evicts_busy_keys(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value")
    demand(bus, policy, [b"cold"])
    for _ in range(4):
        demand(bus, policy, [b"incoming"])
    plan = policy.plan_capacity_admission(
        [key(b"incoming")],
        1,
        capacity([b"cold"], pending=1, unavailable=[b"cold", b"pending"]),
    )
    assert plan.admitted_keys == plan.victim_keys == ()


def test_capacity_forgotten_pending_child_still_protects_resident_parent(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value", history_keys=2)
    demand(bus, policy, [b"root", b"pending-child"])
    demand(bus, policy, [b"root"])
    for _ in range(5):
        demand(bus, policy, [b"incoming"])
    assert policy.profile(b"pending-child") is None
    plan = policy.plan_capacity_admission(
        [key(b"incoming")],
        1,
        capacity([b"root"], pending=1, unavailable=[b"pending-child"]),
    )
    assert plan.admitted_keys == plan.victim_keys == ()


def test_capacity_replacement_preserves_resident_prefix_and_deduplicates(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value")
    demand(bus, policy, [b"root", b"suffix"])
    for _ in range(6):
        demand(bus, policy, [b"incoming"])
    plan = policy.plan_capacity_admission(
        [key(b"incoming"), key(b"incoming")],
        1,
        capacity([b"root", b"suffix"]),
    )
    assert plan.admitted_keys == (key(b"incoming"),)
    assert plan.victim_keys == (key(b"suffix"),)


def test_capacity_sunk_write_is_not_charged_to_resident(runtime):
    bus, _, build = runtime
    policy = build(l2_capacity_policy="value", prefix_write_l2_ms=(100.0, 200.0))
    for _ in range(3):
        demand(bus, policy, [b"resident"])
    for _ in range(18):
        demand(bus, policy, [b"incoming"])
    assert policy.plan_capacity_admission(
        [key(b"incoming")], 1, capacity([], limit=1)
    ).admitted_keys == (key(b"incoming"),)
    plan = policy.plan_capacity_admission(
        [key(b"incoming")], 1, capacity([b"resident"], limit=1)
    )
    assert plan.admitted_keys == plan.victim_keys == ()


def test_capacity_unknown_stale_and_duplicate_resident_never_claim_space(runtime):
    bus, now, build = runtime
    policy = build(l2_capacity_policy="value")
    for _ in range(4):
        demand(bus, policy, [b"hot"])
    assert (
        policy.plan_capacity_admission(
            [key(b"hot")], 1, capacity([b"hot"])
        ).admitted_keys
        == ()
    )
    assert (
        policy.plan_capacity_admission([key(b"unknown")], 1, capacity([])).admitted_keys
        == ()
    )
    now[0] += policy.config_ttl_s + 1
    plan = policy.plan_capacity_admission([key(b"hot")], 1, capacity([]))
    assert plan.admitted_keys == plan.victim_keys == ()


def test_identity_tags_are_unsupported_and_use_default_store(runtime):
    bus, _, build = runtime
    policy = build()
    demand(bus, policy, [b"cold"])
    plain = key(b"cold")
    tagged = replace(plain, identity=replace(plain.identity, tags={"tenant": "other"}))
    assert policy.select_store_targets([plain], adapters()) == {}
    assert policy.select_store_targets([tagged], adapters()) == {0: [tagged]}


@pytest.mark.parametrize("request_configs", [{"lmcache.tag.tenant": "other"}, None])
def test_actual_lookup_emitter_preserves_tag_scope(runtime, request_configs):
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    bus, _, build = runtime
    policy = build()
    drained = threading.Event()
    seen = []

    def observe(event):
        seen.append(event)
        if len(seen) == 5:
            drained.set()

    bus.subscribe(EventType.MP_LOOKUP, observe)

    def stop_after_emission(*args):
        raise RuntimeError("stop-after-emission")

    subject = object.__new__(LookupModule)
    subject._ctx = SimpleNamespace(
        event_bus=bus,
        chunk_size=256,
        layout_desc_registry=SimpleNamespace(
            find=lambda *args: SimpleNamespace(
                dtypes=[torch.bfloat16], shapes=[[2, 256, 8]]
            ),
            find_attn_desc=stop_after_emission,
        ),
        token_hasher=SimpleNamespace(
            compute_packed_chunk_hashes=lambda *args, **kwargs: [b"cold"]
        ),
    )
    request = SimpleNamespace(
        model_name="model",
        world_size=1,
        request_id="scope",
        cache_salt="",
        token_bytes=b"\x01\x00\x00\x00",
        end=1,
        num_tokens=1,
        request_configs=request_configs,
    )
    for _ in range(5):
        with pytest.raises(RuntimeError, match="stop-after-emission"):
            subject.lookup(request, 1)
    assert drained.wait(3)
    expected_tags = {"tenant": "other"} if request_configs else {}
    assert seen[0].metadata["tags"] == expected_tags
    if request_configs:
        assert policy.statistics()["requests"] == 0
        assert policy.profile(b"cold") is None
    else:
        assert policy.statistics()["requests"] == 5
        assert policy.profile(b"cold")["frequency"] > 4


def test_failed_profiled_manager_construction_does_not_subscribe(runtime, monkeypatch):
    # First Party
    from lmcache.v1.distributed.eviction_policy.profiled import ProfileConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    bus, _, _ = runtime
    config = parse_args(
        [
            "--l1-size-gb",
            "0.01",
            "--eviction-policy",
            "Profiled",
            "--profiling-config",
            "unused.json",
        ]
    )
    closed = []
    manager = SimpleNamespace(
        config=config.l1_manager_configs[0],
        l1_manager_id=0,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.create_l1_managers",
        lambda _: [manager],
    )
    for name in [
        "lmcache.v1.distributed.storage_manager.get_event_bus",
        "lmcache.v1.mp_observability.event_bus.get_event_bus",
    ]:
        monkeypatch.setattr(name, lambda: bus)
    monkeypatch.setattr(
        "lmcache.v1.mp_observability.event_bus.is_observability_enabled", lambda: True
    )
    monkeypatch.setattr(
        ProfileConfig, "load", lambda _: ProfileConfig("model", 256, (10.0,))
    )

    def fail_controller(*args, **kwargs):
        raise RuntimeError("injected-controller-failure")

    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.L1EvictionController", fail_controller
    )
    with pytest.raises(RuntimeError, match="injected-controller-failure"):
        StorageManager(config)
    assert closed == [True]
    assert not bus.has_subscribers(EventType.MP_LOOKUP)


def test_cost_model_distinguishes_first_copy_and_existing_disk_copy():
    config = make_config()
    assert config.first_copy_gain_ms(0, 2.0) == pytest.approx(10.0)
    assert config.retention_gain_ms(0, 2.0, disk_backed=True) == pytest.approx(6.0)
    assert config.retention_gain_ms(0, 2.0, disk_backed=False) == pytest.approx(18.0)
    # Marginal costs telescope; no whole-prefix cost credited to every chunk.
    assert config.first_copy_gain_ms(1, 2.0) == pytest.approx(10.0)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), True])
def test_recompute_fraction_rejects_invalid_inputs(fraction):
    with pytest.raises(ValueError, match="recompute_fraction"):
        make_config(recompute_fraction=fraction)


def test_unknown_value_model_is_rejected():
    with pytest.raises(ValueError, match="value_model"):
        make_config(value_model="oracle")


def test_probability_rejects_count_domain_no_cost_control():
    with pytest.raises(ValueError, match="no-cost"):
        make_config(value_model="reuse_probability", mode="no-cost")


@pytest.mark.parametrize(
    "half_life,horizon,write",
    [(1.0, 1e-310, 4.8), (1e-310, 1e-310, 4.8), (1e308, 1e-15, 4.2)],
)
def test_probability_scales_extreme_rate_before_it_can_overflow(
    runtime, half_life, horizon, write
):
    bus, now, build = runtime
    now[0] = 0
    policy = build(
        value_model="reuse_probability",
        horizon_s=horizon,
        half_life_s=half_life,
        prefix_write_l2_ms=(write, 2 * write),
    )
    demand(bus, policy, [b"hot"])
    now[0] = horizon
    demand(bus, policy, [b"hot"])
    # Probability is about .632 (or .75 when age is one half-life),
    # not inflated by overflow or quantized subnormal exposure.
    assert policy.select_store_targets([key(b"hot")], adapters()) == {}


@pytest.mark.parametrize("write,admit", [(1.8, True), (2.4, False)])
def test_probability_uses_decayed_time_exposure_not_plain_age(runtime, write, admit):
    bus, now, build = runtime
    policy = build(
        value_model="reuse_probability",
        horizon_s=10.0,
        half_life_s=10.0,
        prefix_write_l2_ms=(write, 2 * write),
    )
    demand(bus, policy, [b"hot"])
    now[0] += 10
    demand(bus, policy, [b"hot"])
    now[0] += 10
    # Repeat mass .5 / exposure (1-.25)/(ln(2)/10), P about .370.
    # Plain-age division incorrectly predicts P about .221.
    expected = {0: [key(b"hot")]} if admit else {}
    assert policy.select_store_targets([key(b"hot")], adapters()) == expected


def test_recompute_fraction_changes_only_recompute_alternatives():
    config = make_config(recompute_fraction=0.5)
    assert config.first_copy_gain_ms(0, 0.8) == pytest.approx(-1.2)
    assert config.retention_gain_ms(0, 0.8, disk_backed=False) == pytest.approx(3.2)
    assert config.retention_gain_ms(0, 0.8, disk_backed=True) == pytest.approx(2.4)


def test_probability_cannot_turn_one_hit_savings_into_unbounded_reuses(runtime):
    bus, _, build = runtime
    policy = build(value_model="reuse_probability", prefix_write_l2_ms=(8.0, 16.0))
    for _ in range(20):
        demand(bus, policy, [b"hot"])
    # Even P=1 saves only 10-4ms, less than the 8ms first-copy write charge.
    assert policy.select_store_targets([key(b"hot")], adapters()) == {}


def test_probability_horizon_and_idle_decay_change_actual_admission(runtime):
    bus, now, build = runtime
    short = build(
        value_model="reuse_probability",
        horizon_s=0.1,
        prefix_write_l2_ms=(0.3, 0.6),
    )
    long = build(
        value_model="reuse_probability",
        horizon_s=1.0,
        prefix_write_l2_ms=(0.3, 0.6),
    )
    demand(bus, long, [b"hot"])
    assert long.select_store_targets([key(b"hot")], adapters()) == {}
    now[0] += 10
    demand(bus, long, [b"hot"])
    assert short.select_store_targets([key(b"hot")], adapters()) == {}
    assert long.select_store_targets([key(b"hot")], adapters()) == {0: [key(b"hot")]}
    now[0] += 120
    assert long.select_store_targets([key(b"hot")], adapters()) == {}


def test_recompute_fraction_reverses_real_admission(runtime):
    bus, _, build = runtime
    full = build(value_model="reuse_probability")
    partial = build(value_model="reuse_probability", recompute_fraction=0.5)
    for _ in range(5):
        demand(bus, partial, [b"hot"])
    assert full.select_store_targets([key(b"hot")], adapters()) == {0: [key(b"hot")]}
    assert partial.select_store_targets([key(b"hot")], adapters()) == {}


def test_recompute_fraction_changes_eviction_but_not_disk_replica_value(runtime):
    bus, _, build = runtime
    full = build(value_model="reuse_probability")
    partial = build(value_model="reuse_probability", recompute_fraction=1.0)
    for chunk in (b"A", b"B"):
        for _ in range(5):
            demand(bus, partial, [chunk])
    for policy in (full, partial):
        policy.on_keys_created([key(b"A"), key(b"B")])
        policy.on_l2_keys_stored([key(b"B")], [1024])
    assert full.get_eviction_actions(0.5)[0].keys == [key(b"B")]
    assert partial.get_eviction_actions(0.5)[0].keys == [key(b"A")]


def test_store_entry_point_rejects_negative_net_gain(runtime):
    bus, _, build = runtime
    policy = build(prefix_fetch_l2_ms=(12.0, 24.0))
    demand(bus, policy, [b"cold"])
    assert policy.select_store_targets([key(b"cold")], adapters()) == {}
    assert policy.select_l1_deletions([key(b"cold")]) == []


def test_unknown_stale_and_observe_defer_to_original_store(runtime):
    bus, now, build = runtime
    policy = build(prefix_fetch_l2_ms=(12.0, 24.0))
    assert policy.select_store_targets([key(b"unknown")], adapters()) == {
        0: [key(b"unknown")]
    }
    demand(bus, policy, [b"cold"])
    now[0] += policy.config_ttl_s + 1
    assert policy.select_store_targets([key(b"cold")], adapters()) == {
        0: [key(b"cold")]
    }
    observe = build(mode="observe", prefix_fetch_l2_ms=(12.0, 24.0))
    demand(bus, observe, [b"cold"])
    assert observe.select_store_targets([key(b"cold")], adapters()) == {
        0: [key(b"cold")]
    }


def test_frequency_and_distance_affect_admission(runtime):
    bus, _, build = runtime
    policy = build(prefix_write_l2_ms=(8.0, 16.0))
    for _ in range(5):
        demand(bus, policy, [b"hot"])
    assert policy.select_store_targets([key(b"hot")], adapters()) == {0: [key(b"hot")]}
    for index in range(30):
        demand(bus, policy, [str(index).encode()])
    assert policy.select_store_targets([key(b"hot")], adapters()) == {}


def test_real_l2_callbacks_change_retention_not_ownership(runtime):
    bus, _, build = runtime
    policy = build()
    for _ in range(3):
        demand(bus, policy, [b"A"])
        demand(bus, policy, [b"B"])
    policy.on_keys_created([key(b"A")])
    policy.on_keys_created([key(b"B")])
    policy.on_l2_keys_stored([key(b"B")], [1024])
    assert policy.get_eviction_actions(0.5)[0].keys == [key(b"B")]
    assert policy.get_eviction_actions(0.5, lambda _: False) == []
    # Merely proposing eviction never removes resident tracking.
    assert policy.get_eviction_actions(0.5)[0].keys == [key(b"B")]
    policy.on_l2_keys_deleted([key(b"B")])
    assert policy.get_eviction_actions(0.5)[0].keys == [key(b"A")]


@pytest.mark.parametrize(
    "changes",
    [
        {"prefix_fetch_l1_ms": (1.0,)},
        {"prefix_fetch_l2_ms": (4.0, float("nan"))},
        {"prefix_fetch_l2_ms": (4.0, 3.0)},
        {"half_life_s": 0.0},
        {"config_ttl_s": True},
    ],
)
def test_invalid_cost_tables_are_rejected(changes):
    with pytest.raises(ValueError):
        make_config(**changes)


def test_local_selector_and_injected_controller_use_one_policy(
    runtime, tmp_path, monkeypatch
):
    bus, _, build = runtime
    record = {
        "schema_version": 2,
        "model_name": "model",
        "chunk_size": 256,
        "prefix_recompute_ms": [10.0, 20.0],
        "prefix_fetch_l1_ms": [1.0, 2.0],
        "prefix_fetch_l2_ms": [4.0, 8.0],
        "prefix_write_l2_ms": [2.0, 4.0],
        "model_revision": "0" * 64,
        "dtypes": ["torch.bfloat16"],
        "shapes": [[2, 256, 8]],
    }
    path = tmp_path / "costs.json"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    local = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    record["model_name"] = str(model)
    record["model_revision"] = local.model_revision(model)
    path.write_text(json.dumps(record))
    parsed = parse_args(
        [
            "--l1-size-gb",
            "0.5",
            "--eviction-policy",
            "ProfiledValue",
            "--profiling-config",
            str(path),
        ]
    )
    module = importlib.import_module("lmcache.v1.mp_observability.event_bus")
    monkeypatch.setattr(module, "get_event_bus", lambda: bus)
    monkeypatch.setattr(module, "is_observability_enabled", lambda: True)
    assert not bus.has_subscribers(EventType.MP_LOOKUP)
    CreateEvictionPolicy(parsed.eviction_config, defer_subscription=True)
    assert not bus.has_subscribers(EventType.MP_LOOKUP)
    policy = CreateEvictionPolicy(parsed.eviction_config)
    listeners = []
    manager = SimpleNamespace(register_listener=listeners.append)
    L1EvictionController(manager, parsed.eviction_config, policy=policy)
    assert listeners[0].policy is policy
    assert policy.select_l1_deletions([]) == []
    path.write_text(json.dumps({**record, "schema_version": True}))
    with pytest.raises(ValueError):
        CreateEvictionPolicy(parsed.eviction_config)
    path.write_text(json.dumps(record))
    (model / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="model bytes"):
        CreateEvictionPolicy(parsed.eviction_config)


def test_schema_two_rejects_unknown_fields(runtime, tmp_path):
    module = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"schema_version": 2, "future_oracle": True}))
    with pytest.raises(ValueError):
        module.LocalValueConfig.load(str(path))


def test_suffix_not_admitted_when_parent_copy_is_missing(runtime):
    bus, _, build = runtime
    policy = build()
    for _ in range(5):
        demand(bus, policy, [b"parent", b"child"])
    assert policy.select_store_targets([key(b"child")], adapters()) == {}
    policy.on_l2_keys_stored([key(b"parent")], [1024])
    assert policy.select_store_targets([key(b"child")], adapters()) == {
        0: [key(b"child")]
    }


def test_recency_decay_reduces_observed_reuse(runtime):
    bus, now, build = runtime
    policy = build()
    for _ in range(4):
        demand(bus, policy, [b"hot"])
    assert policy.select_store_targets([key(b"hot")], adapters()) == {0: [key(b"hot")]}
    now[0] += 120
    assert policy.select_store_targets([key(b"hot")], adapters()) == {}


def test_two_hit_control_has_no_cost_ranking(runtime):
    bus, now, build = runtime
    policy = build(mode="two-hit")
    demand(bus, policy, [b"A"])
    assert policy.select_store_targets([key(b"A")], adapters()) == {}
    now[0] += 1
    demand(bus, policy, [b"A"])
    assert policy.select_store_targets([key(b"A")], adapters()) == {0: [key(b"A")]}


def test_cost_snapshot_rejects_mutable_arrays():
    with pytest.raises(ValueError):
        make_config(prefix_fetch_l1_ms=[1.0, 2.0])


def test_nested_index_referenced_model_bytes_are_bound(tmp_path):
    local = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights").mkdir()
    shard = tmp_path / "weights" / "shard.safetensors"
    shard.write_bytes(b"original")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"layer.weight": "weights/shard.safetensors"},
            }
        )
    )
    before = local.model_revision(tmp_path)
    shard.write_bytes(b"changed")
    assert local.model_revision(tmp_path) != before


def test_model_file_symlinks_remain_portable_and_hash_actual_bytes(tmp_path):
    local = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    blob = tmp_path / "blob.safetensors"
    blob.write_bytes(b"first")
    (model / "model.safetensors").symlink_to(blob)
    before = local.model_revision(model)
    blob.write_bytes(b"second")
    assert local.model_revision(model) != before


def test_model_index_cannot_reference_unbound_external_file(tmp_path):
    local = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"layer": "../outside.safetensors"},
            }
        )
    )
    with pytest.raises(ValueError, match="unbound"):
        local.model_revision(tmp_path)


def test_unknown_resident_preserves_stock_lru_fallback(runtime):
    bus, _, build = runtime
    policy = build()
    for _ in range(5):
        demand(bus, policy, [b"known"])
    known = key(b"known")
    unsupported = replace(key(b"other"), cache_salt="unsupported")
    policy.on_keys_created([known])
    policy.on_keys_created([unsupported])
    assert policy.get_eviction_actions(0.5)[0].keys == [known]


@pytest.mark.parametrize(
    "scope", [{"cache_salt": "other"}, {"world_size": 2}, {"tags": {"tenant": "other"}}]
)
def test_foreign_scope_cannot_pollute_empty_tp1_demand(runtime, scope):
    bus, _, build = runtime
    policy = build()
    # Standard
    import threading

    barrier = threading.Event()
    bus.subscribe(EventType.MP_LOOKUP, lambda _: barrier.set())
    bus.publish(
        Event(
            event_type=EventType.MP_LOOKUP,
            metadata={
                "model_name": "model",
                "chunk_size": 256,
                "chunk_hashes": [b"foreign"],
                "dtypes": ["torch.bfloat16"],
                "shapes": [[2, 256, 8]],
                "cache_salt": "",
                "world_size": 1,
                "tags": {},
                **scope,
            },
        )
    )
    assert barrier.wait(3)
    assert policy.statistics()["requests"] == 0


def test_deleted_root_and_partial_store_do_not_admit_unreachable_suffix(runtime):
    bus, _, build = runtime
    policy = build(
        prefix_recompute_ms=(10.0, 20.0, 30.0),
        prefix_fetch_l1_ms=(1.0, 2.0, 3.0),
        prefix_fetch_l2_ms=(4.0, 8.0, 12.0),
        prefix_write_l2_ms=(2.0, 4.0, 6.0),
    )
    for _ in range(6):
        demand(bus, policy, [b"root", b"middle", b"child"])
    policy.on_l2_keys_stored([key(b"root"), key(b"middle")], [1, 1])
    policy.on_l2_keys_deleted([key(b"root")])
    assert policy.select_store_targets([key(b"child")], adapters()) == {}


def test_layout_mismatch_cannot_seed_cost_decisions(runtime):
    bus, _, build = runtime
    policy = build(prefix_fetch_l2_ms=(12.0, 24.0))
    for _ in range(3):
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP,
                metadata={
                    "model_name": "model",
                    "chunk_size": 256,
                    "chunk_hashes": [b"bad"],
                    "dtypes": ["torch.float32"],
                    "shapes": [[999, 999]],
                },
            )
        )
    time.sleep(0.02)
    assert policy.statistics()["requests"] == 0
    assert policy.select_store_targets([key(b"bad")], adapters()) == {0: [key(b"bad")]}


def test_invalid_topology_rejected_before_policy_subscription(runtime, monkeypatch):
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

    bus, _, _ = runtime
    config = parse_args(
        [
            "--l1-size-gb",
            "0.5",
            "--eviction-policy",
            "ProfiledValue",
            "--profiling-config",
            "must-not-be-read.json",
        ]
    )
    calls = []

    def factory(_):
        calls.append(1)
        raise AssertionError("policy must not be created for invalid topology")

    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.CreateEvictionPolicy", factory
    )
    with pytest.raises(ValueError, match="one L1"):
        StorageManager(config)
    assert calls == []


@pytest.mark.parametrize(
    "method,args",
    [
        ("add_l2_adapter", (None,)),
        ("delete_l2_adapter", (0,)),
        ("reconfigure_l2_adapter", (0, "resize", {})),
    ],
)
def test_fixed_topology_public_calls_reject_before_native_access(method, args):
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

    # Fault boundary: no native owners exist. Removing the public-entry guard
    # must fail here by accessing nonexistent owners, not silently mutate a store.
    subject = object.__new__(StorageManager)
    object.__setattr__(subject, "_fixed_l2_topology", True)
    with pytest.raises(L2ReconfigureError, match="fixed calibrated"):
        getattr(subject, method)(*args)


@pytest.mark.parametrize("value_model", ["expected_reuses", "reuse_probability"])
def test_native_storage_manager_executes_admission_and_releases_locks(
    runtime, tmp_path, monkeypatch, value_model
):
    # Standard
    from dataclasses import asdict
    import threading

    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2Adapter
    from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
    from lmcache.v1.platform import torch_dev

    if not torch_dev.is_available():
        pytest.skip("native pinned-memory/controller execution requires an accelerator")

    bus, _, _ = runtime
    local = importlib.import_module(
        "lmcache.v1.distributed.eviction_policy.local_value"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    costs = make_config(
        model_name=str(model),
        model_revision=local.model_revision(model),
        value_model=value_model,
    )
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"schema_version": 2, **asdict(costs)}))
    config = parse_args(
        [
            "--l1-size-gb",
            "0.03",
            "--no-l1-use-lazy",
            "--eviction-policy",
            "ProfiledValue",
            "--profiling-config",
            str(path),
            "--l2-adapter",
            json.dumps(
                {"type": "mock", "max_size_gb": 0.01, "mock_bandwidth_gb": 10.0}
            ),
        ]
    )
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.get_event_bus", lambda: bus
    )
    monkeypatch.setattr(
        "lmcache.v1.mp_observability.event_bus.get_event_bus", lambda: bus
    )
    monkeypatch.setattr(
        "lmcache.v1.mp_observability.event_bus.is_observability_enabled", lambda: True
    )
    manager = StorageManager(config)
    try:
        barrier = threading.Event()
        bus.subscribe(
            EventType.MP_LOOKUP,
            lambda event: (
                barrier.set() if event.metadata.get("request_id") == "barrier" else None
            ),
        )
        hot_hash = ObjectKey.IntHash2Bytes(1)
        cold_hash = ObjectKey.IntHash2Bytes(2)
        for chunk in [hot_hash] * 5 + [cold_hash]:
            bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    metadata={
                        "model_name": str(model),
                        "chunk_size": 256,
                        "chunk_hashes": [chunk],
                        "dtypes": ["torch.bfloat16"],
                        "shapes": [[2, 256, 8]],
                        "cache_salt": "",
                        "world_size": 1,
                        "tags": {},
                    },
                )
            )
        bus.publish(
            Event(event_type=EventType.MP_LOOKUP, metadata={"request_id": "barrier"})
        )
        assert barrier.wait(3), "native policy observation did not drain"
        hot = replace(key(hot_hash), model_name=str(model))
        cold = replace(key(cold_hash), model_name=str(model))
        objects = manager.reserve_write(
            [hot, cold],
            MemoryLayoutDesc(
                shapes=[torch.Size([2, 256, 8])],
                dtypes=[torch.bfloat16],
            ),
        )
        assert set(objects) == {hot, cold}
        for obj in objects.values():
            tensor = obj.get_tensor(0)
            assert tensor is not None
            tensor.fill_(17)
        manager.finish_write([hot, cold], l1_owners_of(objects))
        adapter = manager.l2_adapters()[0][1]
        assert isinstance(adapter, MockL2Adapter)
        deadline = time.monotonic() + 5
        while not adapter.debug_has_key(hot):
            assert time.monotonic() < deadline, "admitted key never reached native L2"
            time.sleep(0.01)
        assert not adapter.debug_has_key(cold), "rejected key was physically stored"
        # Finished native stores must not retain their L1 read locks.
        deleted = 0
        while deleted < 2:
            count, skipped = manager.delete_l1_keys([hot, cold])
            deleted += count
            if deleted == 2:
                assert skipped == 0
                break
            assert time.monotonic() < deadline, "store read locks were not released"
            time.sleep(0.01)
        with pytest.raises(L2ReconfigureError, match="fixed calibrated"):
            manager.delete_l2_adapter(0)
        assert bus.subscriber_exception_counts() == {}
    finally:
        manager.close()

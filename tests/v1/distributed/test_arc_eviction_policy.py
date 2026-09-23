# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP Adaptive Replacement Cache policy."""

# Standard
import argparse
import json

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import (
    ARCEvictionPolicy,
    CreateEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import EvictionDestination
from lmcache.v1.distributed.l2_adapters.config import (
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="model",
        kv_rank=0,
    )


def _complete_eviction(policy: ARCEvictionPolicy, ratio: float) -> list[ObjectKey]:
    actions = policy.get_eviction_actions(ratio)
    assert len(actions) == 1
    policy.on_keys_removed(actions[0].keys)
    return actions[0].keys


class TestARCTracking:
    def test_touch_promotes_recent_key_and_protects_frequent_keys(self) -> None:
        policy = ARCEvictionPolicy()
        keys = [_key(i) for i in range(1, 5)]
        for key in keys:
            policy.on_keys_created([key])
        policy.on_keys_touched(keys[:2])

        actions = policy.get_eviction_actions(0.5)

        assert actions[0].keys == keys[2:]

    def test_batch_order_keeps_later_suffixes_at_the_lru_end(self) -> None:
        policy = ARCEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]
        policy.on_keys_created(keys)

        actions = policy.get_eviction_actions(0.34)

        assert actions[0].keys == [keys[2]]

    def test_key_eligible_filter_skips_locked_key(self) -> None:
        policy = ARCEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]
        for key in keys:
            policy.on_keys_created([key])

        actions = policy.get_eviction_actions(
            0.34, key_eligible_filter=lambda key: key != keys[0]
        )

        assert actions[0].keys == [keys[1]]

    def test_ratio_and_destination_match_mp_policy_contract(self) -> None:
        policy = ARCEvictionPolicy(default_destination=EvictionDestination.L2_CACHE)
        policy.on_keys_created([_key(1)])
        assert policy.get_eviction_actions(0.0) == []
        assert (
            policy.get_eviction_actions(0.01)[0].destination
            == EvictionDestination.L2_CACHE
        )


class TestARCGhostFeedback:
    def test_adaptation_uses_fractional_ghost_ratio(self) -> None:
        policy = ARCEvictionPolicy()
        capacity_keys = [_key(i) for i in range(1, 6)]
        for key in capacity_keys:
            policy.on_keys_created([key])
        policy.on_keys_removed(capacity_keys)

        frequent_ghosts = [_key(i) for i in range(6, 9)]
        for key in frequent_ghosts:
            policy.on_keys_created([key])
            policy.on_keys_touched([key])
            assert _complete_eviction(policy, 1.0) == [key]

        recent_ghosts = [_key(9), _key(10)]
        for key in recent_ghosts:
            policy.on_keys_created([key])
            assert _complete_eviction(policy, 1.0) == [key]

        policy.on_keys_created([recent_ghosts[0]])

        assert policy.get_debug_state()["target_t1_size"] == 1.5

    def test_recent_and_frequent_ghosts_adjust_the_next_victim(self) -> None:
        policy = ARCEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1])
        policy.on_keys_created([key2])

        assert _complete_eviction(policy, 0.5) == [key1]
        assert policy.get_debug_state()["b1"] == [key1]

        policy.on_keys_created([key1])
        assert policy.get_debug_state()["target_t1_size"] == 1.0
        assert _complete_eviction(policy, 0.5) == [key2]
        assert policy.get_debug_state()["b1"] == [key2]

        assert _complete_eviction(policy, 1.0) == [key1]
        assert policy.get_debug_state()["b2"] == [key1]

        policy.on_keys_created([key1])
        assert policy.get_debug_state()["target_t1_size"] == 0.0
        key3 = _key(3)
        policy.on_keys_created([key3])
        assert policy.get_eviction_actions(0.5)[0].keys == [key3]

    def test_equal_t1_target_selects_recent_list_like_vllm(self) -> None:
        policy = ARCEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1])
        policy.on_keys_created([key2])
        assert _complete_eviction(policy, 0.5) == [key1]

        policy.on_keys_created([key1])
        state = policy.get_debug_state()
        assert state["target_t1_size"] == 1.0
        assert state["t1"] == [key2]

        assert policy.get_eviction_actions(0.5)[0].keys == [key2]

    def test_explicit_removal_does_not_create_ghost_feedback(self) -> None:
        policy = ARCEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1])
        policy.on_keys_created([key2])

        policy.on_keys_removed([key1])
        policy.on_keys_created([key1])

        state = policy.get_debug_state()
        assert state["b1"] == []
        assert state["b2"] == []
        assert state["target_t1_size"] == 0.0
        assert policy.get_eviction_actions(0.5)[0].keys == [key2]

    def test_selecting_without_completed_removal_keeps_key_resident(self) -> None:
        policy = ARCEvictionPolicy()
        key = _key(1)
        policy.on_keys_created([key])

        assert policy.get_eviction_actions(1.0)[0].keys == [key]

        assert policy.get_num_tracked_keys() == 1
        assert policy.get_debug_state()["b1"] == []


class TestARCConfiguration:
    def test_factory_creates_arc_policy(self) -> None:
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="ARC"))
        assert isinstance(policy, ARCEvictionPolicy)

    def test_l1_cli_accepts_arc(self) -> None:
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        args = parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "ARC"])

        config = parse_args_to_config(args)

        assert config.eviction_config.eviction_policy == "ARC"

    def test_l2_adapter_config_accepts_arc(self) -> None:
        parser = argparse.ArgumentParser()
        add_l2_adapters_args(parser)
        spec = {
            "type": "mock",
            "max_size_gb": 1,
            "mock_bandwidth_gb": 10,
            "eviction": {"eviction_policy": "ARC"},
        }
        args = parser.parse_args(["--l2-adapter", json.dumps(spec)])

        config = parse_args_to_l2_adapters_config(args)

        assert config.adapters[0].eviction_config is not None
        assert config.adapters[0].eviction_config.eviction_policy == "ARC"


class TestARCMPListeners:
    def test_l1_listener_drives_arc_lifecycle(self) -> None:
        policy = ARCEvictionPolicy()
        listener = L1EvictionPolicy(policy)
        key1, key2 = _key(1), _key(2)

        listener.on_l1_keys_write_finished([key1, key2])
        listener.on_l1_keys_accessed([key1])
        action = policy.get_eviction_actions(0.5)[0]
        listener.on_l1_keys_deleted_by_manager(action.keys)

        assert action.keys == [key2]
        assert policy.get_debug_state()["b1"] == [key2]

    def test_l2_listener_drives_arc_lifecycle(self) -> None:
        policy = ARCEvictionPolicy()
        listener = L2EvictionPolicy(policy)
        key1, key2 = _key(1), _key(2)

        listener.on_l2_keys_stored([key1, key2], [1, 1])
        listener.on_l2_keys_accessed([key1])
        action = policy.get_eviction_actions(0.5)[0]
        listener.on_l2_keys_deleted(action.keys)

        assert action.keys == [key2]
        assert policy.get_debug_state()["b1"] == [key2]

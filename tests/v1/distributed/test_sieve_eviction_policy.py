# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP SIEVE eviction policy."""

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
    CreateEvictionPolicy,
    SieveEvictionPolicy,
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


class TestSieveTracking:
    def test_batch_order_evicts_later_suffix_first(self) -> None:
        policy = SieveEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]

        policy.on_keys_created(keys)

        assert policy.get_eviction_actions(0.34)[0].keys == [keys[2]]

    def test_touch_grants_second_chance_without_reordering(self) -> None:
        policy = SieveEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]
        for key in keys:
            policy.on_keys_created([key])
        policy.on_keys_touched([keys[0]])

        actions = policy.get_eviction_actions(0.34)

        assert actions[0].keys == [keys[1]]

    def test_second_chance_bit_is_cleared_by_the_hand(self) -> None:
        policy = SieveEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1])
        policy.on_keys_created([key2])
        policy.on_keys_touched([key1, key2])

        actions = policy.get_eviction_actions(0.5)

        assert actions[0].keys == [key1]

    def test_key_eligible_filter_skips_locked_key(self) -> None:
        policy = SieveEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]
        for key in keys:
            policy.on_keys_created([key])

        actions = policy.get_eviction_actions(
            0.34, key_eligible_filter=lambda key: key != keys[0]
        )

        assert actions[0].keys == [keys[1]]

    def test_removing_hand_advances_to_next_oldest_key(self) -> None:
        policy = SieveEvictionPolicy()
        keys = [_key(i) for i in range(1, 4)]
        for key in keys:
            policy.on_keys_created([key])

        policy.on_keys_removed([keys[0]])

        assert policy.get_eviction_actions(0.5)[0].keys == [keys[1]]

    def test_selection_waits_for_removal_confirmation(self) -> None:
        policy = SieveEvictionPolicy()
        key = _key(1)
        policy.on_keys_created([key])

        assert policy.get_eviction_actions(1.0)[0].keys == [key]
        assert policy.get_num_tracked_keys() == 1

        policy.on_keys_removed([key])
        assert policy.get_num_tracked_keys() == 0
        assert policy.get_eviction_actions(1.0) == []

    def test_ratio_and_destination_match_mp_policy_contract(self) -> None:
        policy = SieveEvictionPolicy(default_destination=EvictionDestination.L2_CACHE)
        policy.on_keys_created([_key(1)])

        assert policy.get_eviction_actions(0.0) == []
        assert (
            policy.get_eviction_actions(0.01)[0].destination
            == EvictionDestination.L2_CACHE
        )


class TestSieveConfiguration:
    def test_factory_creates_sieve_policy(self) -> None:
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="SIEVE"))

        assert isinstance(policy, SieveEvictionPolicy)

    def test_l1_cli_accepts_sieve(self) -> None:
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        args = parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "SIEVE"])

        config = parse_args_to_config(args)

        assert config.eviction_config.eviction_policy == "SIEVE"

    def test_l2_adapter_config_accepts_sieve(self) -> None:
        parser = argparse.ArgumentParser()
        add_l2_adapters_args(parser)
        spec = {
            "type": "mock",
            "max_size_gb": 1,
            "mock_bandwidth_gb": 10,
            "eviction": {"eviction_policy": "SIEVE"},
        }
        args = parser.parse_args(["--l2-adapter", json.dumps(spec)])

        config = parse_args_to_l2_adapters_config(args)

        assert config.adapters[0].eviction_config is not None
        assert config.adapters[0].eviction_config.eviction_policy == "SIEVE"


class TestSieveMPListeners:
    def test_l1_listener_does_not_treat_write_completion_as_a_hit(self) -> None:
        policy = SieveEvictionPolicy()
        listener = L1EvictionPolicy(policy)
        key1, key2 = _key(1), _key(2)

        listener.on_l1_keys_reserved_write([key1, key2])
        listener.on_l1_keys_write_finished([key1, key2])

        assert policy.get_eviction_actions(0.5)[0].keys == [key2]

    def test_l2_listener_drives_sieve_lifecycle(self) -> None:
        policy = SieveEvictionPolicy()
        listener = L2EvictionPolicy(policy)
        key1, key2 = _key(1), _key(2)

        listener.on_l2_keys_stored([key1, key2], [1, 1])
        listener.on_l2_keys_accessed([key2])
        action = policy.get_eviction_actions(0.5)[0]
        listener.on_l2_keys_deleted(action.keys)

        assert action.keys == [key1]
        assert policy.get_num_tracked_keys() == 1

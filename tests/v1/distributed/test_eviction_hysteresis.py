# SPDX-License-Identifier: Apache-2.0
"""Eviction hysteresis (``target_watermark``) in the L1 and L2 eviction
controllers, and the ``EvictionConfig`` range checks.

The controllers are driven synchronously with a fake L1 manager / L2 adapter
and a policy that records the ratio it is asked to evict, so the tests pin
the ratio each branch computes without depending on a real policy's
rounding.
"""

# Standard
from types import MappingProxyType
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
    hysteresis_eviction_ratio,
)


class _RecordingPolicy:
    """Stands in for an eviction policy: records each requested ratio."""

    def __init__(self, support_isolation: bool = False) -> None:
        self.support_isolation = support_isolation
        self.calls: list[tuple[float, Any]] = []

    def get_eviction_actions(self, ratio, cache_salt=None, **kwargs):
        self.calls.append((ratio, cache_salt))
        return []


class _FakeL1Manager:
    def __init__(self, used: int, total: int) -> None:
        self.used, self.total = used, total

    def register_listener(self, listener) -> None:
        pass

    def get_memory_usage(self) -> tuple[int, int]:
        return self.used, self.total

    def is_key_evictable(self, key) -> bool:
        return True


class _FakeL2Adapter:
    def __init__(self, usage: AdapterUsage) -> None:
        self.usage = usage

    def register_listener(self, listener) -> None:
        pass

    def get_usage(self) -> AdapterUsage:
        return self.usage


def _config(trigger: float, target: float, ratio: float, policy="LRU"):
    return EvictionConfig(
        eviction_policy=policy,
        trigger_watermark=trigger,
        target_watermark=target,
        eviction_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# The shared ratio
# ---------------------------------------------------------------------------


def test_hysteresis_ratio() -> None:
    assert hysteresis_eviction_ratio(1.0, 0.5, 0.1) == pytest.approx(0.5)
    assert hysteresis_eviction_ratio(0.9, 0.6, 0.1) == pytest.approx(1 / 3)
    # Off: the configured eviction_ratio is used unchanged.
    assert hysteresis_eviction_ratio(0.9, 0.0, 0.1) == 0.1
    # Already under the target: nothing to free, never negative.
    assert hysteresis_eviction_ratio(0.4, 0.5, 0.1) == 0.0


# ---------------------------------------------------------------------------
# L1EvictionController
# ---------------------------------------------------------------------------


def _l1_controller(used: int, total: int, config: EvictionConfig):
    ctrl = L1EvictionController(_FakeL1Manager(used, total), config)  # type: ignore[arg-type]
    policy = _RecordingPolicy()
    ctrl._eviction_policy = policy  # type: ignore[assignment]
    return ctrl, policy


def test_l1_hysteresis_evicts_down_to_target() -> None:
    ctrl, policy = _l1_controller(900, 1000, _config(0.8, 0.5, 0.1))
    ctrl._check_and_evict()
    assert [r for r, _ in policy.calls] == [pytest.approx(0.4 / 0.9)]


def test_l1_without_target_uses_eviction_ratio() -> None:
    ctrl, policy = _l1_controller(900, 1000, _config(0.8, 0.0, 0.1))
    ctrl._check_and_evict()
    assert [r for r, _ in policy.calls] == [0.1]


def test_l1_below_trigger_does_not_evict() -> None:
    ctrl, policy = _l1_controller(700, 1000, _config(0.8, 0.5, 0.1))
    ctrl._check_and_evict()
    assert policy.calls == []


# ---------------------------------------------------------------------------
# L2EvictionController, aggregate path
# ---------------------------------------------------------------------------


def _l2_setup(usage: AdapterUsage, config: EvictionConfig, isolation=False):
    state = L2AdapterEvictionState(0, _FakeL2Adapter(usage), config)  # type: ignore[arg-type]
    policy = _RecordingPolicy(support_isolation=isolation)
    state.eviction_policy = policy  # type: ignore[assignment]
    qm = QuotaManager() if isolation else None
    return L2EvictionController([state], quota_manager=qm), state, policy, qm


def test_l2_global_hysteresis_evicts_down_to_target() -> None:
    usage = AdapterUsage(total_bytes_used=950, total_capacity_bytes=1000)
    ctrl, state, policy, _ = _l2_setup(usage, _config(0.9, 0.6, 0.2))
    ctrl._check_and_evict(state)
    assert [r for r, _ in policy.calls] == [pytest.approx((0.95 - 0.6) / 0.95)]


def test_l2_global_without_target_uses_eviction_ratio() -> None:
    usage = AdapterUsage(total_bytes_used=950, total_capacity_bytes=1000)
    ctrl, state, policy, _ = _l2_setup(usage, _config(0.9, 0.0, 0.2))
    ctrl._check_and_evict(state)
    assert [r for r, _ in policy.calls] == [0.2]


def test_l2_global_below_trigger_or_unbounded_does_not_evict() -> None:
    below = AdapterUsage(total_bytes_used=500, total_capacity_bytes=1000)
    ctrl, state, policy, _ = _l2_setup(below, _config(0.9, 0.6, 0.2))
    ctrl._check_and_evict(state)
    unbounded = AdapterUsage(total_bytes_used=500, total_capacity_bytes=0)
    ctrl2, state2, policy2, _ = _l2_setup(unbounded, _config(0.9, 0.6, 0.2))
    ctrl2._check_and_evict(state2)
    assert policy.calls == [] and policy2.calls == []


# ---------------------------------------------------------------------------
# L2EvictionController, per-cache_salt path
# ---------------------------------------------------------------------------


def test_l2_per_salt_hysteresis_and_unregistered_wipe() -> None:
    usage = AdapterUsage(
        total_bytes_used=1100,
        total_capacity_bytes=10_000,
        bytes_by_cache_salt=MappingProxyType({"alice": 1000, "stranger": 100}),
    )
    ctrl, state, policy, qm = _l2_setup(
        usage, _config(0.9, 0.5, 0.1, policy="IsolatedLRU"), isolation=True
    )
    assert qm is not None
    qm.set_quota("alice", 1000)

    ctrl._check_and_evict(state)

    # alice: 1000 B over 0.9 * 1000, evicted down to 0.5 * 1000 → ratio 0.5.
    # stranger has no quota (limit 0) and is wiped whatever the target.
    assert sorted(policy.calls, key=lambda c: c[1]) == [
        (pytest.approx(0.5), "alice"),
        (1.0, "stranger"),
    ]


def test_l2_per_salt_without_target_uses_eviction_ratio() -> None:
    usage = AdapterUsage(
        total_bytes_used=1000,
        total_capacity_bytes=10_000,
        bytes_by_cache_salt=MappingProxyType({"alice": 1000}),
    )
    ctrl, state, policy, qm = _l2_setup(
        usage, _config(0.9, 0.0, 0.1, policy="IsolatedLRU"), isolation=True
    )
    assert qm is not None
    qm.set_quota("alice", 1000)
    ctrl._check_and_evict(state)
    assert policy.calls == [(0.1, "alice")]


# ---------------------------------------------------------------------------
# EvictionConfig range checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trigger_watermark": 1.5},
        {"trigger_watermark": -0.1},
        {"trigger_watermark": True},
        {"eviction_ratio": 1.2},
        {"eviction_ratio": -0.5},
        {"target_watermark": -0.1},
        {"trigger_watermark": 0.8, "target_watermark": 0.8},
        {"trigger_watermark": 0.8, "target_watermark": 0.9},
    ],
)
def test_eviction_config_rejects_out_of_range(kwargs) -> None:
    with pytest.raises(ValueError):
        EvictionConfig(eviction_policy="LRU", **kwargs)


def test_eviction_config_accepts_valid_target() -> None:
    assert EvictionConfig("LRU", trigger_watermark=0.8).target_watermark == 0.0
    cfg = EvictionConfig("LRU", trigger_watermark=0.8, target_watermark=0.5)
    assert cfg.target_watermark == 0.5

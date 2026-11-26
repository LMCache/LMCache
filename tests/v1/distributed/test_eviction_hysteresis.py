# SPDX-License-Identifier: Apache-2.0
"""Eviction hysteresis (``target_watermark``) in the eviction controllers.

The controllers are driven synchronously with a fake L1 manager / L2 adapter
and a policy that records the ratio it is asked to evict, so the tests pin
the ratio each branch computes without depending on a real policy's
rounding.
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
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

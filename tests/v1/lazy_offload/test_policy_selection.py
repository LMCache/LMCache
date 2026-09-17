# SPDX-License-Identifier: Apache-2.0
"""What ``create_offload_policy`` builds, and what it refuses to build.

Drain behaviour lives in ``test_fifo_policy.py`` and
``test_eviction_aware_policy.py``; here only the selection and the config
validation done at construction time.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy import (
    POLICY_CONFIG_KEY,
    LazyOffloadMode,
    create_offload_policy,
)
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    EvictionAwarePolicyConfig,
    EvictionAwareStoreQueue,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import FIFOOffloadPolicy


class TestPolicySelection:
    """Which policy ``POLICY_CONFIG_KEY`` maps to."""

    def test_default_is_eviction_aware(self) -> None:
        policy = create_offload_policy({}, MagicMock())
        assert isinstance(policy, EvictionAwareStoreQueue)

    def test_fifo_is_selectable(self) -> None:
        policy = create_offload_policy(
            {POLICY_CONFIG_KEY: LazyOffloadMode.FIFO.value}, MagicMock()
        )
        assert isinstance(policy, FIFOOffloadPolicy)

    def test_unknown_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown offload policy"):
            create_offload_policy({POLICY_CONFIG_KEY: "UNKNOWN"}, MagicMock())


class TestEvictionAwareConfig:
    """Defaults and range checks on the eviction-aware tunables."""

    def test_defaults_are_the_calibrated_values(self) -> None:
        config = EvictionAwarePolicyConfig()
        assert (config.horizon_steps, config.max_deferral_seconds) == (2.5, 0.0)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("horizon_steps", 0),
            ("max_drain_per_step", 0),
            ("max_deferral_seconds", -1.0),
        ],
    )
    def test_out_of_range_tunables_are_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match=field):
            EvictionAwarePolicyConfig(**{field: value})

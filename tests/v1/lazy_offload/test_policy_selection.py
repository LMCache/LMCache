# SPDX-License-Identifier: Apache-2.0
"""What ``create_offload_policy`` builds, and what it refuses to build.

Drain behaviour lives in ``test_fifo_policy.py`` and
``test_eviction_aware_policy.py``; here only the selection.
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

# SPDX-License-Identifier: Apache-2.0
"""Tests for event-driven prefetch completion waiting."""

# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)


def test_wait_for_prefetch_result_does_not_consume_result() -> None:
    """Completion wait wakes on the event notifier and preserves the result."""
    ctrl = PrefetchController(
        l1_manager=cast(L1Manager, object()),
        l2_adapters=[],
        adapter_descriptors=[],
        policy=DefaultPrefetchPolicy(),
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size((1,))],
        dtypes=[torch.float32],
    )

    request_id = ctrl.submit_prefetch_request([], layout)
    assert not ctrl.wait_for_prefetch_result(request_id, timeout=0.01)
    assert ctrl.query_prefetch_result(request_id) is None

    ctrl.start()
    try:
        assert ctrl.wait_for_prefetch_result(request_id, timeout=1.0)
        assert ctrl.query_prefetch_result(request_id) == 0
        assert ctrl.query_prefetch_result(request_id) is None
    finally:
        ctrl.stop()


def test_wait_for_unknown_prefetch_result_returns_immediately() -> None:
    """Unknown request IDs are treated as already unblocked."""
    ctrl = PrefetchController(
        l1_manager=cast(L1Manager, object()),
        l2_adapters=[],
        adapter_descriptors=[],
        policy=DefaultPrefetchPolicy(),
    )
    ctrl.start()
    try:
        assert ctrl.wait_for_prefetch_result(123, timeout=0.01)
    finally:
        ctrl.stop()

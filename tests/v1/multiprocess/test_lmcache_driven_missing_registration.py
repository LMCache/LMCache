# SPDX-License-Identifier: Apache-2.0
"""Missing-registration responses for LMCache-driven GPU transfers."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)


@pytest.mark.parametrize("method_name", ["store", "retrieve"])
def test_missing_registration_returns_terminal_false(method_name: str) -> None:
    """An absent context returns the producer event instead of raising.

    The MQ blocking-handler exception path does not send an error response.
    Returning a normal response therefore ensures the caller's future reaches
    a terminal state during the restart-before-registration window.
    """
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module.get_and_touch_context_entry = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    producer_event = b"worker-producer-event"
    key = SimpleNamespace(request_id="request", cache_salt="", worker_id=0)

    result = getattr(module, method_name)(
        key,
        42,
        [[0]],
        producer_event,
    )

    assert result == (producer_event, False)
    module.get_and_touch_context_entry.assert_called_once_with(42)

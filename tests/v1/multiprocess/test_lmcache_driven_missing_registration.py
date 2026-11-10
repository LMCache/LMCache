# SPDX-License-Identifier: Apache-2.0
"""Missing-registration responses for LMCache-driven GPU transfers."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)


def _cache_key(
    *,
    world_size: int,
    worker_id: int | None,
    request_id: str,
    start: int = 0,
    end: int = 4,
) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=world_size,
        worker_id=worker_id,
        token_ids=(1, 2, 3, 4),
        start=start,
        end=end,
        request_id=request_id,
    )


@pytest.mark.parametrize("method_name", ["store", "retrieve"])
def test_missing_registration_returns_terminal_false(method_name: str) -> None:
    """An absent context returns an event-free response instead of raising.

    The MQ blocking-handler exception path does not send an error response.
    Returning a normal response therefore ensures the caller's future reaches
    a terminal state during the restart-before-registration window. The empty
    handle indicates that the server submitted no device work.
    """
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module.get_and_touch_context_entry = MagicMock(  # type: ignore[method-assign]
        return_value=None
    )
    module._ctx = MagicMock()
    producer_event = b"worker-producer-event"
    key = _cache_key(world_size=1, worker_id=0, request_id="request")

    result = getattr(module, method_name)(
        key,
        42,
        [[0]],
        producer_event,
    )

    assert result == (b"", False)
    module.get_and_touch_context_entry.assert_called_once_with(42)

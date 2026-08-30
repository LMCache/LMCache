# SPDX-License-Identifier: Apache-2.0
"""Preemption synchronization tests for the vLLM multiprocess adapter."""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)


def _healthy_adapter() -> tuple[LMCacheMPWorkerAdapter, MagicMock]:
    adapter = object.__new__(LMCacheMPWorkerAdapter)
    adapter._health_event = threading.Event()
    adapter._health_event.set()
    transfer_ctx = MagicMock(name="transfer_ctx")
    adapter.transfer_ctx = transfer_ctx
    return adapter, transfer_ctx


def test_handle_preemptions_delegates_without_device_wide_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport owns its completion primitive during preemption.

    A worker-side device synchronize cannot complete server-process CUDA work
    and serializes unrelated inference work when preemptions are frequent.
    """
    adapter, transfer_ctx = _healthy_adapter()
    synchronize = MagicMock(name="synchronize")
    monkeypatch.setattr(adapter_mod.torch_dev, "synchronize", synchronize)

    adapter.handle_preemptions(True)

    transfer_ctx.flush_inflight_stores.assert_called_once_with()
    synchronize.assert_not_called()


def test_handle_preemptions_false_is_noop() -> None:
    adapter, transfer_ctx = _healthy_adapter()

    adapter.handle_preemptions(False)

    transfer_ctx.flush_inflight_stores.assert_not_called()

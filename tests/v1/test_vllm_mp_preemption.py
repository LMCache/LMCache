# SPDX-License-Identifier: Apache-2.0
"""Preemption synchronization tests for the vLLM multiprocess adapter."""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
)


def _healthy_adapter() -> tuple[LMCacheMPWorkerAdapter, MagicMock]:
    transfer_ctx = MagicMock(name="transfer_ctx")
    adapter = cast(
        LMCacheMPWorkerAdapter,
        SimpleNamespace(is_healthy=True, transfer_ctx=transfer_ctx),
    )
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

    LMCacheMPWorkerAdapter.handle_preemptions(adapter, True)

    transfer_ctx.flush_inflight_stores.assert_called_once_with()
    synchronize.assert_not_called()


def test_handle_preemptions_false_is_noop() -> None:
    adapter, transfer_ctx = _healthy_adapter()

    LMCacheMPWorkerAdapter.handle_preemptions(adapter, False)

    transfer_ctx.flush_inflight_stores.assert_not_called()

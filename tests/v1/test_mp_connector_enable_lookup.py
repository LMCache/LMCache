# SPDX-License-Identifier: Apache-2.0
"""Lookup gating via ``lmcache.mp.enable_lookup`` through public MP APIs.

When ``enable_lookup`` is false the connector must not submit any LOOKUP --
neither from ``get_num_new_matched_tokens`` nor from the eager-prefetch
``on_new_request`` path -- while the store path stays functional: request
trackers are still created so ``update_state_after_alloc`` and stores work
unchanged. Network/GPU I/O is stubbed.
"""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import (
    vllm_multi_process_adapter as adapter_mod,  # noqa: E402
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

pytestmark = pytest.mark.no_shared_allocator


def _config(enable_lookup: bool, eager_prefetch: bool) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lmcache.mp.enable_lookup": enable_lookup,
                    "lmcache.mp.eager_prefetch": eager_prefetch,
                },
            ),
        ),
    )


def _request(request_id: str = "request") -> Request:
    return cast(
        Request,
        SimpleNamespace(
            request_id=request_id,
            cache_salt="",
            all_token_ids=list(range(12)),
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
            resumable=False,
        ),
    )


@pytest.fixture
def mock_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub transport, telemetry and GPU resources, keeping scheduler logic real."""
    scheduler = MagicMock(lmcache_tokens_per_chunk=4)
    scheduler.check_lookup_result.return_value = 0
    transfer = MagicMock()
    transfer.create_recorded_event.return_value = None
    telemetry = MagicMock()
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 4)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())
    monkeypatch.setattr(adapter_mod, "set_ipc_policy", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda *a, **kw: transfer
    )
    monkeypatch.setattr(
        adapter_mod.RequestTelemetryFactory, "create", lambda **kw: telemetry
    )
    return SimpleNamespace(scheduler=scheduler, transfer=transfer, telemetry=telemetry)


@pytest.mark.parametrize("enable_lookup", [False, True])
def test_enable_lookup_gates_every_lookup_path(
    enable_lookup: bool, mock_io: SimpleNamespace
) -> None:
    """False skips LOOKUPs on both submission paths; True submits them.

    The scheduler may poll get_num_new_matched_tokens repeatedly, so both
    polls must stay gated, and on_new_request (eager prefetch) must not offer
    a second, ungated submission path.
    """
    scheduler = LMCacheMPConnector(
        _config(enable_lookup, eager_prefetch=True), KVConnectorRole.SCHEDULER
    )
    try:
        request = _request()

        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        # The tracker is still created, so update_state_after_alloc and the
        # store path keep working for gated requests.
        assert request.request_id in scheduler.request_trackers

        scheduler.on_new_request(request)

        if enable_lookup:
            assert mock_io.scheduler.maybe_submit_lookup_request.called
        else:
            mock_io.scheduler.maybe_submit_lookup_request.assert_not_called()
    finally:
        scheduler.shutdown()

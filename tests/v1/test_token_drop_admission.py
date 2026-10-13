# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

vllm = pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestState,
)


def request(td: bool):
    extra = {"lmcache.token_drop": {"algorithm": "rkv", "config": {}}} if td else {}
    return SimpleNamespace(
        request_id="td" if td else "normal",
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": extra}),
        cache_salt=None,
        all_token_ids=[1, 2],
        prompt_token_ids=[1, 2],
        num_prompt_tokens=2,
        skip_reading_prefix_cache=False,
        resumable=False,
        status=None,
    )


def connector():
    instance = LMCacheMPConnector.__new__(LMCacheMPConnector)
    instance._role = KVConnectorRole.SCHEDULER
    instance._eager_prefetch = True
    instance._reserve_last_token_for_lookup = False
    instance.lazy_offload = True
    instance.request_trackers = {}
    instance.scheduler_adapter = MagicMock()
    instance._lazy_offload_manager = MagicMock()
    return instance


@pytest.mark.skipif(
    vllm.__version__.split("+", 1)[0] != "0.25.1",
    reason="Token dropping is pinned to vLLM 0.25.1",
)
def test_td_admission_skips_lmcache_and_apc():
    c = connector()
    td = request(True)
    with patch(
        "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
        return_value=(None, None),
    ):
        c.on_new_request(td)
        assert c.get_num_new_matched_tokens(td, 0) == (0, False)
    assert td.skip_reading_prefix_cache is True
    assert c.request_trackers["td"].state == LMCacheMPRequestState.BYPASS_LMCACHE
    c.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
    c._lazy_offload_manager.on_request_arrived.assert_not_called()


@pytest.mark.skipif(
    vllm.__version__.split("+", 1)[0] != "0.25.1",
    reason="Token dropping is pinned to vLLM 0.25.1",
)
def test_td_cannot_execute_until_worker_lifecycle_lands():
    c = connector()
    td = request(True)
    with patch(
        "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
        return_value=(None, None),
    ):
        c.on_new_request(td)
    output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="td")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )
    with pytest.raises(NotImplementedError, match="not enabled yet"):
        c.build_connector_meta(output)


def test_normal_request_still_uses_lookup():
    c = connector()
    normal = request(False)
    with patch(
        "lmcache.integration.vllm.lmcache_mp_metadata.extract_mm_features",
        return_value=(None, None),
    ):
        c.on_new_request(normal)
    assert normal.skip_reading_prefix_cache is False
    c.scheduler_adapter.maybe_submit_lookup_request.assert_called_once()
    c._lazy_offload_manager.on_request_arrived.assert_called_once()

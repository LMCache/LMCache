# SPDX-License-Identifier: Apache-2.0
"""The scheduler connector declares prefix-replay windows loaded only when
the lookup served them and vLLM's own hit leaves the window to the load."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
)
from vllm.v1.request import RequestStatus  # noqa: E402
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
)

CHUNK = 256
REPLAY_TOKENS = 128
GLOBAL_GROUPS = (7,)
REPLAY_GROUPS = (0, 1, 2)


@pytest.fixture(autouse=True)
def _loaded_groups_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a vLLM without the loaded-groups hook its later default.

    vLLM releases before the hook never call the override, so the tests stand
    in vLLM's default (the prefix-cacheable groups) to exercise it there too.
    """
    if not hasattr(KVConnectorBase_V1, "get_loaded_kv_cache_group_ids"):
        monkeypatch.setattr(
            KVConnectorBase_V1,
            "get_loaded_kv_cache_group_ids",
            lambda self, request: self._kv_cache_config.prefix_cacheable_group_ids,
            raising=False,
        )


def _request(num_tokens: int) -> SimpleNamespace:
    tokens = list(range(num_tokens))
    return SimpleNamespace(
        request_id="req-0",
        resumable=False,
        status=RequestStatus.WAITING,
        cache_salt="",
        prompt_token_ids=tokens,
        num_prompt_tokens=num_tokens,
        all_token_ids=ConstantList(tokens),
        mm_features=[],
        sampling_params=SimpleNamespace(extra_args=None),
        block_hashes=[],
    )


def _connector(hit_tokens: int, window_served: bool) -> LMCacheMPConnector:
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK
    adapter.check_lookup_result.return_value = hit_tokens
    adapter.query_lookup_window.return_value = window_served
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = False
    connector._hit_alignment_tokens = CHUNK
    connector._connector_stats = LMCacheMPConnectorStats()
    connector.lazy_offload = False
    connector._prefix_replay_tokens = REPLAY_TOKENS
    connector._prefix_replay_group_ids = REPLAY_GROUPS
    connector._kv_cache_config = SimpleNamespace(
        prefix_cacheable_group_ids=GLOBAL_GROUPS
    )
    return connector


@pytest.mark.parametrize(
    "window_served,vllm_hit,expected",
    [
        pytest.param(True, 0, GLOBAL_GROUPS + REPLAY_GROUPS, id="window-loaded"),
        pytest.param(False, 0, GLOBAL_GROUPS, id="window-missing"),
        # vLLM's own hit reaches into the window: retrieve would skip it.
        pytest.param(True, 2 * CHUNK, GLOBAL_GROUPS, id="apc-covers-window"),
    ],
)
def test_loaded_groups_follow_window_lookup(window_served, vllm_hit, expected):
    request = _request(4 * CHUNK)
    connector = _connector(hit_tokens=2 * CHUNK, window_served=window_served)

    matched, _load_async = connector.get_num_new_matched_tokens(request, vllm_hit)

    assert matched == 2 * CHUNK - vllm_hit
    assert connector.get_loaded_kv_cache_group_ids(request) == tuple(sorted(expected))


def test_windows_not_stored_skips_the_window_query():
    request = _request(4 * CHUNK)
    connector = _connector(hit_tokens=2 * CHUNK, window_served=True)
    connector._prefix_replay_group_ids = ()

    connector.get_num_new_matched_tokens(request, 0)

    connector.scheduler_adapter.query_lookup_window.assert_not_called()
    assert connector.get_loaded_kv_cache_group_ids(request) == GLOBAL_GROUPS


def test_unknown_request_gets_vllm_default():
    connector = _connector(hit_tokens=0, window_served=False)
    assert connector.get_loaded_kv_cache_group_ids(_request(CHUNK)) == GLOBAL_GROUPS


@pytest.mark.parametrize(
    "window_served,matched",
    [
        # vLLM does not floor a hit whose window is loaded, so only the last
        # token is recomputed.
        pytest.param(True, 4 * CHUNK - 1, id="window-loaded"),
        # Bounded replay floors the hit to a block, so a whole chunk goes back.
        pytest.param(False, 3 * CHUNK, id="window-missing"),
    ],
)
def test_full_prompt_hit_gives_back_a_chunk_only_without_window(window_served, matched):
    request = _request(4 * CHUNK)
    connector = _connector(hit_tokens=4 * CHUNK, window_served=window_served)

    assert connector.get_num_new_matched_tokens(request, 0)[0] == matched


@pytest.mark.parametrize(
    "num_external_tokens,expected,window_loads",
    [
        pytest.param(2 * CHUNK, GLOBAL_GROUPS + REPLAY_GROUPS, 1, id="loads"),
        # No retrieve, so no window reaches vLLM's blocks.
        pytest.param(0, GLOBAL_GROUPS, 0, id="no-load"),
    ],
)
def test_window_counts_as_loaded_only_when_the_request_loads(
    num_external_tokens, expected, window_loads
):
    request = _request(4 * CHUNK)
    connector = _connector(hit_tokens=2 * CHUNK, window_served=True)
    connector.get_num_new_matched_tokens(request, 0)
    blocks = SimpleNamespace(get_block_ids=lambda: ())

    connector.update_state_after_alloc(request, blocks, num_external_tokens)

    assert connector.get_loaded_kv_cache_group_ids(request) == tuple(sorted(expected))
    stats = connector.get_kv_connector_stats()
    reduced = stats.reduce() if stats is not None else {}
    assert reduced.get("LMCache MP window loads", 0) == window_loads

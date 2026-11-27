# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for PR #3: vLLM MP connector lookup_id lifecycle."""

# Standard
from unittest.mock import MagicMock, patch

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.mp_lookup_state import LookupState
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.common.custom_types import READ_LIFETIME_V1, IPCCacheServerKey
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.transport.futures import MessagingFuture


def _done_future(value):
    fut: MessagingFuture = MessagingFuture()
    fut.set_result(value)
    return fut


def _make_scheduler_adapter(
    experimental: list[str], extra_config: dict | None = None
) -> tuple[LMCacheMPSchedulerAdapter, MagicMock]:
    client = MagicMock()
    client.get_chunk_size.return_value = _done_future(16)
    client.get_experimental.return_value = _done_future(experimental)
    client.lookup.return_value = _done_future(None)
    client.query_prefetch_status.return_value = _done_future(16)
    client.end_session.return_value = _done_future(None)
    with (
        patch(
            "lmcache.integration.vllm.vllm_multi_process_adapter."
            "RequestClientFactory.create",
            return_value=client,
        ),
        patch(
            "lmcache.integration.vllm.vllm_multi_process_adapter."
            "_uses_lmcache_driven_transfer",
            return_value=True,
        ),
    ):
        adapter = LMCacheMPSchedulerAdapter(
            server_urls=["grpc://127.0.0.1:5555"],
            context=MagicMock(),
            model_name="m",
            vllm_block_size=16,
            parallel_strategy=ParallelStrategy(
                vllm_world_size=1,
                vllm_worker_id=0,
                tp_size=1,
                pp_size=1,
                n_servers=1,
            ),
            extra_config=extra_config,
        )
    adapter._ensure_heartbeat_started = MagicMock()  # type: ignore[method-assign]
    return adapter, client


def test_scheduler_adapter_negotiates_read_lifetime_v1_and_isolates_lookups() -> None:
    """Scheduler enables read_lifetime_v1, assigns unique lookup_ids, and cleans up."""
    adapter, client = _make_scheduler_adapter([READ_LIFETIME_V1])
    assert adapter._read_lifetime_v1 is True

    tokens = pack_token_ids(list(range(32)))
    assert adapter.maybe_submit_lookup_request("req-1", tokens) is True
    lid1 = adapter.get_active_lookup_id("req-1")
    assert lid1 != ""
    # Lookup key and query_prefetch_status carry lid1.
    sent_key = client.lookup.call_args[0][0]
    assert sent_key.lookup_id == lid1
    client.query_prefetch_status.assert_called_with("req-1", lid1)

    # Tearing down lid1 sends end_session("req-1", lid1) and allows a fresh lid2.
    adapter.end_lookup("req-1", lid1)
    client.end_session.assert_called_with("req-1", lid1)

    assert adapter.maybe_submit_lookup_request("req-1", tokens) is True
    lid2 = adapter.get_active_lookup_id("req-1")
    assert lid2 != "" and lid2 != lid1

    # Request-level end_session tears down remaining lid2 AND request-level session.
    client.end_session.reset_mock()
    adapter.end_session("req-1")
    calls = [c.args for c in client.end_session.call_args_list]
    assert ("req-1", lid2) in calls
    assert ("req-1",) in calls


def test_scheduler_adapter_falls_back_cleanly_on_old_server_or_kill_switch() -> None:
    """Older server without capability or kill-switch=False keeps lookup_id empty."""
    old_adapter, old_client = _make_scheduler_adapter([])
    assert old_adapter._read_lifetime_v1 is False
    tokens = pack_token_ids(list(range(16)))
    assert old_adapter.maybe_submit_lookup_request("req-old", tokens) is True
    assert old_adapter.get_active_lookup_id("req-old") == ""
    assert old_client.lookup.call_args[0][0].lookup_id == ""
    old_client.query_prefetch_status.assert_called_with("req-old")

    disabled, _ = _make_scheduler_adapter(
        [READ_LIFETIME_V1], extra_config={"lmcache.mp.read_lifetime_v1": False}
    )
    assert disabled._read_lifetime_v1 is False


def test_metadata_and_worker_adapter_propagate_lookup_id() -> None:
    """GetRetrieveMetadata and worker retrieve submission propagate lookup_id."""
    req = MagicMock()
    req.request_id = "req-w1"
    req.cache_salt = ""
    req.lora_request = None
    req.all_token_ids = list(range(32))
    req.num_prompt_tokens = 32
    req.sampling_params = None
    req.mm_features = []

    tracker = LMCacheMPRequestTracker(req)
    tracker.lookup_id = "lkp-xyz"
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_lmcache_hit_tokens = 32
    meta = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, 16, [16])
    assert meta is not None
    assert meta.lookup_id == "lkp-xyz"

    worker = object.__new__(LMCacheMPWorkerAdapter)
    worker.model_name = "m"
    worker.parallel_strategy = ParallelStrategy(
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )
    worker._health_event = MagicMock(is_set=MagicMock(return_value=True))
    worker._ensure_heartbeat_started = MagicMock()  # type: ignore[method-assign]
    worker.kv_caches = {}
    worker.blocks_in_chunk = 1
    worker.retrieve_futures = {}
    worker.retrieve_events = {}
    worker._block_ids_per_group = lambda op: op.block_ids  # type: ignore[method-assign]
    transfer_ctx = MagicMock()
    worker._require_transfer_ctx = lambda _op: transfer_ctx  # type: ignore[method-assign]

    op = LoadStoreOp(
        token_bytes=pack_token_ids(list(range(32))),
        block_ids=[[0, 1]],
        start=0,
        end=32,
    )
    worker.batched_submit_retrieve_requests(
        ["req-w1"], [op], None, lookup_ids=["lkp-xyz"]
    )
    submitted_key = transfer_ctx.submit_retrieve.call_args[0][1]
    assert submitted_key.lookup_id == "lkp-xyz"


def test_lookup_state_exposes_lookup_id_and_preserves_it_on_tail_free() -> None:
    """LookupState forwards lookup_id on prefetch status and mismatch tail free."""
    c1, c2 = MagicMock(), MagicMock()
    c1.lookup.return_value = _done_future(None)
    c2.lookup.return_value = _done_future(None)
    c1.query_prefetch_status.return_value = _done_future(32)
    c2.query_prefetch_status.return_value = _done_future(16)
    key = IPCCacheServerKey.from_token_ids(
        "m",
        1,
        None,
        list(range(32)),
        start=0,
        end=32,
        request_id="req-ls",
        lookup_id="lkp-ls",
    )
    state = LookupState(
        key,
        {"u1": c1, "u2": c2},
        tp_size=1,
        tokens_per_chunk=16,
        timeout=5.0,
        blocking_lookup=False,
    )
    assert state.lookup_id == "lkp-ls"
    state.submit()
    assert state.get_result() == 16
    tail_key = c1.free_lookup_locks.call_args[0][0]
    assert tail_key.lookup_id == "lkp-ls"

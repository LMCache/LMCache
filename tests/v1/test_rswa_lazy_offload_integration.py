# SPDX-License-Identifier: Apache-2.0
"""Prompt-only cache boundaries across the public lazy-offload lifecycle.

These CPU contract tests use real vLLM requests and block pools. They do not
claim that Unlimited-OCR can enable lazy offload: its model configuration
disables the prefix caching that lazy offload requires.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
)

POLICIES = ("FIFO", "EVICTION_AWARE")


def _config(lazy: bool, policy: str = "FIFO") -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="baidu/Unlimited-OCR", rswa_window=128),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": lazy,
                "lmcache.mp.lazy_offload_policy": policy,
            },
        ),
        parallel_config=SimpleNamespace(
            world_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )


def _request(prompt: int = 512, *, resumable: bool = False) -> Request:
    request = Request(
        request_id="same-id",
        prompt_token_ids=list(range(prompt)),
        sampling_params=SamplingParams(max_tokens=1024),
        pooling_params=None,
        resumable=resumable,
    )
    request.append_output_token_ids(list(range(prompt, prompt + 512)))
    return request


def _step(tokens: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        total_num_scheduled_tokens=tokens,
        num_scheduled_tokens={},
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )


def _lazy_connector(
    policy: str,
) -> tuple[LMCacheMPConnector, LazyOffloadManager, BlockPool, MagicMock]:
    """Seed callback state without constructing networked adapters.

    The real base initializes role ownership; the manager, policies, requests
    and block pool remain real. Private connector state is seeded only here
    because the production constructor handshakes with a separate server.
    """
    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = 256
    adapter.check_lookup_result.return_value = 0
    receipts: dict[str, int] = {}

    def completed(request_id: str, count: int) -> bool:
        receipts[request_id] = receipts.get(request_id, 0) + count
        if receipts[request_id] == 2:
            del receipts[request_id]
            return True
        return False

    adapter.update_pending_store_count.side_effect = completed
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    KVConnectorBase_V1.__init__(
        connector, _config(True, policy), KVConnectorRole.SCHEDULER, None
    )  # type: ignore[arg-type]
    connector._prompt_only_cache = True
    connector._hit_alignment_tokens = 16
    connector._group_tokens_per_block = [16]
    connector._eager_prefetch = True
    connector._connector_stats = LMCacheMPConnectorStats()
    connector.lazy_offload = True
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": policy,
            "lmcache.mp.lazy_offload_threshold": 1,
        },
        [16],
        adapter,
    )
    connector._lazy_offload_manager = manager
    pool = BlockPool(num_gpu_blocks=128, enable_caching=True, hash_block_size=16)
    connector.bind_gpu_block_pool(pool)
    return connector, manager, pool, adapter


def _queue_prompt(
    connector: LMCacheMPConnector,
    manager: LazyOffloadManager,
    pool: BlockPool,
    request: Request,
) -> list[KVCacheBlock]:
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    blocks = pool.get_new_blocks(64)
    for block in blocks:
        block.set_block_hash(
            make_block_hash_with_group_id(
                BlockHash(f"block-{block.block_id}".encode()), 0
            ),
            num_tokens=16,
        )
    connector.update_state_after_alloc(
        request,
        SimpleNamespace(get_block_ids=lambda: ([b.block_id for b in blocks],)),
        num_external_tokens=0,
    )  # type: ignore[arg-type]
    tracker = connector.request_trackers[request.request_id]
    for computed in (256, 1024):
        tracker.num_scheduled_tokens = computed
        meta = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, 256, [16])
        assert meta is not None
        manager.add_store_candidate(meta)
    assert LMCacheMPRequestMetadata.GetStoreMetadata(tracker, 256, [16]) is None
    return blocks


def _receipt(
    connector: LMCacheMPConnector, request_id: str, *, failed: bool = False
) -> None:
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={request_id: 1},
                failed_store_requests={request_id} if failed else set(),
            )
        )
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("failed", [False, True])
def test_prompt_stores_wait_for_every_worker_before_releasing_session(
    policy: str, failed: bool
) -> None:
    """Coalescing excludes decode KV; failed stores still release their pins."""
    connector, manager, pool, adapter = _lazy_connector(policy)
    request = _request()
    blocks = _queue_prompt(connector, manager, pool, request)

    assert connector.request_finished(request, []) == (False, None)
    pool.free_blocks(blocks)  # vLLM releases request ownership after the callback.
    adapter.end_session.assert_not_called()
    assert not connector.has_pending_push_work()  # Buffered is not in flight.
    assert len(connector.build_connector_meta(_step(0))) == 0
    metadata = connector.build_connector_meta(_step())
    assert len(metadata.requests) == 1
    store = metadata.requests[0]
    assert (store.op.start, store.op.end) == (0, 512)
    assert store.op.token_ids == list(range(512))
    assert store.op.flat_block_ids == [b.block_id for b in blocks[:32]]
    assert [b.ref_cnt for b in blocks] == [1] * 32 + [0] * 32
    assert connector.has_pending_push_work()

    _receipt(connector, request.request_id, failed=failed)
    adapter.end_session.assert_not_called()
    assert connector.has_pending_push_work()
    _receipt(connector, request.request_id)
    adapter.end_session.assert_called_once_with(request.request_id)
    assert not connector.has_pending_push_work()
    assert all(block.ref_cnt == 0 for block in blocks)
    _receipt(connector, request.request_id)  # A stale receipt cannot end twice.
    adapter.end_session.assert_called_once_with(request.request_id)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("finish_before_receipt", [False, True])
@pytest.mark.parametrize("already_in_flight", [False, True])
def test_bypass_reusing_a_deferred_id_does_not_take_over_its_session(
    policy: str, finish_before_receipt: bool, already_in_flight: bool
) -> None:
    """The old cached request settles independently of the new cache bypass."""
    connector, manager, pool, adapter = _lazy_connector(policy)
    old = _request()
    blocks = _queue_prompt(connector, manager, pool, old)
    connector.request_finished(old, [])
    pool.free_blocks(blocks)
    if already_in_flight:
        assert len(connector.build_connector_meta(_step()).requests) == 1

    bypass = _request(prompt=128, resumable=True)
    adapter.reset_mock()
    connector.on_new_request(bypass)
    assert connector.get_num_new_matched_tokens(bypass, 0) == (0, False)
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.end_session.assert_not_called()
    if finish_before_receipt:
        assert connector.request_finished(bypass, []) == (False, None)
    if not already_in_flight:
        assert len(connector.build_connector_meta(_step()).requests) == 1
    _receipt(connector, old.request_id)
    adapter.end_session.assert_not_called()
    _receipt(connector, old.request_id, failed=True)
    adapter.end_session.assert_called_once_with(old.request_id)
    if not finish_before_receipt:
        assert connector.request_finished(bypass, []) == (False, None)
    adapter.end_session.assert_called_once_with(old.request_id)
    assert all(block.ref_cnt == 0 for block in blocks)
    assert not connector.has_pending_push_work()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("reuse_finished_id", [False, True])
def test_new_tracker_restarts_the_prompt_without_reusing_stale_stores(
    policy: str, reuse_finished_id: bool
) -> None:
    """Preemption drops buffered work; id reuse isolates late failed receipts."""
    connector, manager, pool, adapter = _lazy_connector(policy)
    request = _request()
    old_blocks = _queue_prompt(connector, manager, pool, request)
    if reuse_finished_id:
        connector.request_finished(request, [])
    pool.free_blocks(old_blocks)

    if reuse_finished_id:
        assert len(connector.build_connector_meta(_step()).requests) == 1
        request = _request()
    else:
        request.status = RequestStatus.PREEMPTED
    new_blocks = _queue_prompt(connector, manager, pool, request)
    request.status = RequestStatus.RUNNING
    assert connector.request_finished(request, []) == (False, None)
    pool.free_blocks(new_blocks)
    adapter.end_session.assert_not_called()
    if reuse_finished_id:
        assert len(connector.build_connector_meta(_step())) == 0
        _receipt(connector, request.request_id, failed=True)
        adapter.end_session.assert_not_called()
        _receipt(connector, request.request_id)
        adapter.end_session.assert_not_called()
        assert all(block.ref_cnt == 0 for block in old_blocks)

    metadata = connector.build_connector_meta(_step())
    assert len(metadata.requests) == 1
    store = metadata.requests[0]
    assert (store.op.start, store.op.end) == (0, 512)
    assert store.op.token_ids == list(range(512))
    assert store.op.flat_block_ids == [b.block_id for b in new_blocks[:32]]
    _receipt(connector, request.request_id)
    adapter.end_session.assert_not_called()
    _receipt(connector, request.request_id)
    adapter.end_session.assert_called_once_with(request.request_id)
    assert all(block.ref_cnt == 0 for block in new_blocks)
    assert not connector.has_pending_push_work()


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("resumable", [False, True])
def test_zero_cap_request_has_no_lazy_session_even_after_preemption(
    policy: str, resumable: bool
) -> None:
    """Empty prompts and resumable prompts never create a lazy manager slot."""
    connector, manager, _pool, adapter = _lazy_connector(policy)
    request = _request(prompt=128 if resumable else 0, resumable=resumable)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([],)), 0
    )  # type: ignore[arg-type]
    request.status = RequestStatus.PREEMPTED
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.request_finished(request, []) == (False, None)
    assert manager.on_request_arrived(request.request_id).sessions_to_end == []
    adapter.maybe_submit_lookup_request.assert_not_called()
    adapter.end_session.assert_not_called()
    assert not connector.has_pending_push_work()


class _ReachedAdapter(Exception):
    """Sentinel showing startup validation accepted the configuration."""


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("lazy", [False, True])
def test_prefix_cache_disabled_configuration_requires_immediate_offload(
    monkeypatch: pytest.MonkeyPatch,
    role: KVConnectorRole,
    policy: str,
    lazy: bool,
) -> None:
    """Both policies reject APC-off before constructing either network adapter."""
    # First Party
    import lmcache.integration.vllm.lmcache_mp_connector as connector_module

    config = _config(lazy, policy)
    for name in (
        "validate_mamba_step_alignment",
        "validate_kv_cache_groups",
        "validate_dcp_support",
    ):
        monkeypatch.setattr(connector_module, name, MagicMock())
    monkeypatch.setattr(
        connector_module, "get_group_tokens_per_block", lambda *_args: [16]
    )
    monkeypatch.setattr(
        connector_module, "get_vllm_scheduler_block_size", lambda *_args: 16
    )
    monkeypatch.setattr(
        connector_module,
        "build_parallel_strategy_from_vllm_config",
        lambda *_args: SimpleNamespace(dcp_size=1, vllm_world_size=1, vllm_worker_id=0),
    )
    monkeypatch.setattr(connector_module.zmq.Context, "instance", MagicMock())
    scheduler = MagicMock(side_effect=_ReachedAdapter)
    worker = MagicMock(side_effect=_ReachedAdapter)
    monkeypatch.setattr(connector_module, "LMCacheMPSchedulerAdapter", scheduler)
    monkeypatch.setattr(connector_module, "LMCacheMPWorkerAdapter", worker)
    if lazy:
        with pytest.raises(ValueError, match="requires vLLM prefix caching"):
            LMCacheMPConnector(config, role)  # type: ignore[arg-type]
        scheduler.assert_not_called()
        worker.assert_not_called()
    else:
        with pytest.raises(_ReachedAdapter):
            LMCacheMPConnector(config, role)  # type: ignore[arg-type]
        assert scheduler.call_count + worker.call_count == 1
    assert config.cache_config.enable_prefix_caching is False

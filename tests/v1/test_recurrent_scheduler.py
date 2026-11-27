# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import recurrent_scheduler as scheduler_module
from lmcache.integration.vllm.recurrent_scheduler import RecurrentHistoryScheduler
from lmcache.v1.transport.recurrent_history import HistoryResponse


def start_history(history, adapter, request, tracker):
    adapter._mq_timeout = 5
    adapter.read_recurrent_history.side_effect = lambda future: future.value
    history.begin_background(request, tracker)
    history.settle(request.request_id)
    history.poll()
    history.poll()


@pytest.mark.parametrize("missing", [None, 0, 1, 2])
def test_complete_local_history_skips_rpc_and_rechecks_after_eviction(
    missing: int | None,
) -> None:
    pool = MagicMock(hash_block_size=16)
    blocks = [[SimpleNamespace(block_id=2 * i + j) for j in range(2)] for i in range(3)]
    pool.get_cached_block.side_effect = [
        None if i == missing else block for i, block in enumerate(blocks)
    ]
    adapter = MagicMock(model_name="model", world_size=2)
    adapter.submit_recurrent_history.return_value = SimpleNamespace(
        query=lambda: True, value=HistoryResponse()
    )
    history = RecurrentHistoryScheduler(pool, adapter, [1, 2])
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b", b"c"])
    tracker = MagicMock(num_lmcache_hit_tokens=64, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 256

    start_history(history, adapter, request, tracker)

    if missing is None:
        adapter.submit_recurrent_history.assert_not_called()
        tracker.packed_token_ids.assert_not_called()
    else:
        adapter.submit_recurrent_history.assert_called_once()
        sent = adapter.submit_recurrent_history.call_args.args[0]
        assert sent.action == "prepare"
        assert sent.already_local == [16 * (i + 1) for i in range(3) if i != missing]
        assert sent.key.end == 64
    pool.touch.assert_not_called()
    pool.get_new_blocks.assert_not_called()
    pool.free_blocks.assert_not_called()
    assert not history.pending
    assert history.enabled
    assert tracker.request_configs is None

    # A later request must observe eviction instead of reusing the previous scan.
    adapter.submit_recurrent_history.reset_mock()
    request.request_id = "next-request"
    pool.get_cached_block.side_effect = [blocks[0], None, blocks[2]]
    start_history(history, adapter, request, tracker)
    adapter.submit_recurrent_history.assert_called_once()
    assert adapter.submit_recurrent_history.call_args.args[0].already_local == [16, 48]


def test_unfull_primary_restore_skips_history_without_disabling_it() -> None:
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    adapter = MagicMock(model_name="model", world_size=2)
    adapter.submit_recurrent_history.return_value = SimpleNamespace(
        query=lambda: True, value=HistoryResponse()
    )
    history = RecurrentHistoryScheduler(pool, adapter, [1])
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b"])
    tracker = MagicMock(num_lmcache_hit_tokens=24, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 128

    start_history(history, adapter, request, tracker)

    adapter.submit_recurrent_history.assert_not_called()
    assert history.enabled
    assert tracker.request_configs is None

    tracker.num_lmcache_hit_tokens = 32
    start_history(history, adapter, request, tracker)

    adapter.submit_recurrent_history.assert_called_once()


def test_cancel_timeout_retries_without_recycling_unfenced_destinations(
    monkeypatch,
) -> None:
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    pool.get_num_free_blocks.return_value = 8
    pool.get_new_blocks.return_value = [
        SimpleNamespace(block_id=1),
        SimpleNamespace(block_id=2),
    ]
    plan = HistoryResponse(lease="lease", positions=[16], groups=[1, 2])
    adapter = MagicMock(model_name="model", world_size=2)
    adapter.submit_recurrent_history.side_effect = [
        SimpleNamespace(query=lambda: True, value=plan),
        TimeoutError("control timeout"),
        SimpleNamespace(
            query=lambda: True, value=HistoryResponse(lease="lease", settled=True)
        ),
        SimpleNamespace(
            query=lambda: True, value=HistoryResponse(lease="lease", settled=True)
        ),
    ]
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b"])
    request.is_finished.return_value = False
    tracker = MagicMock(num_lmcache_hit_tokens=32, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 128
    history = RecurrentHistoryScheduler(pool, adapter, [1, 2])

    start_history(history, adapter, request, tracker)
    history.settle("request", cancel=True)
    pool.free_blocks.assert_not_called()
    assert history.pending["request"].canceled

    clock = [0.0]
    monkeypatch.setattr(
        scheduler_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    history.poll()
    clock[0] += 0.06
    history.poll()
    history.poll()
    assert not history.pending
    assert pool.free_blocks.call_count == 2
    pool.cache_full_blocks.assert_not_called()


def test_receipt_for_another_lease_cannot_release_private_blocks() -> None:
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    pool.get_num_free_blocks.return_value = 8
    pool.get_new_blocks.return_value = [SimpleNamespace(block_id=1)]
    adapter = MagicMock(model_name="model", world_size=2)
    adapter.submit_recurrent_history.side_effect = [
        SimpleNamespace(
            query=lambda: True,
            value=HistoryResponse(lease="mine", positions=[16], groups=[1]),
        ),
        SimpleNamespace(
            query=lambda: True,
            value=HistoryResponse(lease="other", settled=True, successful=True),
        ),
    ]
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b"])
    request.is_finished.return_value = False
    tracker = MagicMock(num_lmcache_hit_tokens=32, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 128
    history = RecurrentHistoryScheduler(pool, adapter, [1])

    start_history(history, adapter, request, tracker)
    history.settle("request")
    history.worker_finished("request", "mine", 0)
    history.worker_finished("request", "mine", 1)
    history.poll()
    history.poll()

    assert history.pending
    pool.free_blocks.assert_not_called()
    pool.cache_full_blocks.assert_not_called()


def test_abort_before_primary_receipt_drops_unsubmitted_history():
    history = RecurrentHistoryScheduler(MagicMock(hash_block_size=16), MagicMock(), [1])
    history.begin_background(
        SimpleNamespace(request_id="A"), SimpleNamespace(num_lmcache_hit_tokens=32)
    )
    history.adapter.submit_recurrent_history.assert_not_called()
    history.finish_request("A", normal_finish=False)
    assert not history.has_work()


def test_coverage_before_primary_and_partial_overlap():
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    adapter = MagicMock(model_name="model", world_size=2)
    adapter._mq_timeout = 10
    adapter.read_recurrent_history.side_effect = lambda future: future.value
    sent = []

    def submit(op):
        sent.append(op)
        positions = [
            pos for pos in range(16, op.key.end, 16) if pos not in op.already_local
        ]
        return SimpleNamespace(
            query=lambda: True,
            value=HistoryResponse(
                lease=op.key.request_id, groups=[1], positions=positions
            ),
        )

    adapter.submit_recurrent_history.side_effect = submit
    h = RecurrentHistoryScheduler(pool, adapter, [1])

    def start(rid, hashes, salt=""):
        request = SimpleNamespace(request_id=rid, block_hashes=hashes)
        tracker = SimpleNamespace(
            num_lmcache_hit_tokens=len(hashes) * 16,
            cache_salt=salt,
            request_configs=None,
            packed_token_ids=lambda: bytes(64 * len(hashes)),
        )
        h.begin_background(request, tracker)
        h.poll()
        h.poll()

    start("A", [b"a", b"b", b"c"])
    assert h.plans["A"].positions == [16, 32]
    assert h.take_background_tasks() == []  # primary still in progress
    start("B", [b"a", b"b", b"c"])
    assert [op.key.request_id for op in sent] == ["A"]
    assert "B" not in h.background and "B" not in h.pending
    start("partial", [b"a", b"b", b"x", b"y"])
    assert sent[-1].already_local == [16, 32] and h.plans["partial"].positions == [48]
    start("other-prefix", [b"w", b"z"])
    assert sent[-1].already_local == []
    start("other-salt", [b"a", b"b", b"c"], "isolated")
    assert sent[-1].already_local == []
    h.settle("A", cancel=True)
    assert not any(owner == "A" for owner in h.owners.values())
    assert "partial" in h.owners.values()


def test_two_overlapping_async_responses_only_one_owner():
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    adapter = MagicMock(model_name="m", world_size=1)
    adapter._mq_timeout = 5

    def submit(op):
        return SimpleNamespace(
            query=lambda: True,
            value=HistoryResponse(
                lease=op.key.request_id if op.key else op.lease,
                positions=[16],
                groups=[1],
            ),
        )

    adapter.submit_recurrent_history.side_effect = submit
    adapter.read_recurrent_history.side_effect = lambda f: f.value
    h = RecurrentHistoryScheduler(pool, adapter, [1])
    for rid in ("A", "B"):
        h.begin_background(
            SimpleNamespace(request_id=rid, block_hashes=[b"x", b"y"]),
            SimpleNamespace(
                num_lmcache_hit_tokens=32,
                cache_salt="",
                request_configs=None,
                packed_token_ids=lambda: bytes(128),
            ),
        )
    h.poll()
    h.poll()
    h.poll()
    assert set(h.owners.values()) == {"A"} and set(h.plans) == {"A"}
    assert "B" not in h.background and not h.dispatch and not h.pending
    assert any(
        c.args[0].action == "cancel" and c.args[0].lease == "B"
        for c in adapter.submit_recurrent_history.call_args_list
    )


def test_prepare_timeout_allows_another_primary_request(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        scheduler_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    adapter = MagicMock(model_name="model", world_size=2)
    adapter._mq_timeout = 3
    adapter.submit_recurrent_history.return_value.query.return_value = False
    scheduler = RecurrentHistoryScheduler(pool, adapter, [1])
    tracker = SimpleNamespace(
        num_lmcache_hit_tokens=32,
        request_configs=None,
        cache_salt="",
        packed_token_ids=lambda: bytes(128),
    )
    scheduler.begin_background(
        SimpleNamespace(request_id="A", block_hashes=[b"a", b"b"]), tracker
    )
    scheduler.poll()
    assert "A" in scheduler.planning
    scheduler.begin_background(
        SimpleNamespace(request_id="B", block_hashes=[b"x", b"y"]), tracker
    )
    scheduler.settle("B")
    scheduler.poll()
    assert "B" in scheduler.primary_done and "A" in scheduler.planning
    now[0] = 3.01
    scheduler.poll()
    assert not scheduler.planning and not scheduler.enabled
    adapter.read_recurrent_history.assert_not_called()
    assert scheduler.take_background_tasks() == []


@pytest.mark.parametrize("action", ["status", "acknowledge"])
def test_unready_control_retries_only_after_backoff(monkeypatch, action):
    now = [0.0]
    monkeypatch.setattr(
        scheduler_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    adapter = MagicMock()
    adapter._mq_timeout = 3
    adapter.submit_recurrent_history.return_value.query.return_value = False
    scheduler = RecurrentHistoryScheduler(MagicMock(hash_block_size=16), adapter, [1])
    for instant, submissions in ((0, 1), (1, 1), (3.01, 1), (3.02, 1), (3.07, 2)):
        now[0] = instant
        assert scheduler._advance("lease", action) is None
        assert adapter.submit_recurrent_history.call_count == submissions
        adapter.read_recurrent_history.assert_not_called()

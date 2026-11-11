# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.integration.vllm.recurrent_scheduler import RecurrentHistoryScheduler
from lmcache.v1.multiprocess.protocols.recurrent_history import HistoryResponse


def test_cancel_timeout_retries_without_recycling_unfenced_destinations() -> None:
    pool = MagicMock(hash_block_size=16)
    pool.get_cached_block.return_value = None
    pool.get_num_free_blocks.return_value = 8
    pool.get_new_blocks.return_value = [
        SimpleNamespace(block_id=1),
        SimpleNamespace(block_id=2),
    ]
    plan = HistoryResponse(lease="lease", positions=[16], groups=[1, 2])
    adapter = MagicMock(model_name="model", world_size=2)
    adapter.recurrent_history.side_effect = [
        plan,
        TimeoutError("control timeout"),
        HistoryResponse(lease="lease", settled=True),
        HistoryResponse(lease="lease", settled=True),
    ]
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b"])
    request.is_finished.return_value = False
    tracker = MagicMock(num_lmcache_hit_tokens=32, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 128
    history = RecurrentHistoryScheduler(pool, adapter, [1, 2])

    history.prepare(request, tracker)
    history.settle("request", cancel=True)
    pool.free_blocks.assert_not_called()
    assert history.pending["request"].canceled

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
    adapter.recurrent_history.side_effect = [
        HistoryResponse(lease="mine", positions=[16], groups=[1]),
        HistoryResponse(lease="other", settled=True, successful=True),
    ]
    request = MagicMock(request_id="request", block_hashes=[b"a", b"b"])
    request.is_finished.return_value = False
    tracker = MagicMock(num_lmcache_hit_tokens=32, cache_salt="", request_configs=None)
    tracker.packed_token_ids.return_value = b"\x00" * 128
    history = RecurrentHistoryScheduler(pool, adapter, [1])

    history.prepare(request, tracker)
    history.settle("request")

    assert history.pending
    pool.free_blocks.assert_not_called()
    pool.cache_full_blocks.assert_not_called()

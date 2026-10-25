# SPDX-License-Identifier: Apache-2.0
"""Scheduler and worker queues for split metadata lookup."""

# Standard
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.tier_pipeline import (
    PHASE_MISS,
    PHASE_PENDING,
    PHASE_READY,
    TierAdmission,
    TierLoad,
    TierWorkerQueue,
    admission_tokens,
    completion_kind,
    offer_waiting_lookups,
)


def _load(num_chunks: int = 1) -> TierLoad:
    return TierLoad(
        num_chunks=num_chunks,
        tiers=[],
        cum_chunk_lengths_total=[0, 4],
        tier_expected_chunks=[num_chunks],
        keys_per_chunk=1,
    )


def test_config_defaults_off():
    config = LMCacheEngineConfig()
    assert config.split_metadata_lookup is False
    assert completion_kind(False) == ""
    assert completion_kind(True) == "preloaded"


def test_lookup_response_kind_defaults_empty():
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/inprocess/v1/lookup_client/async_lookup_message.py"
    )
    spec = spec_from_file_location("async_lookup_message_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    msg = module.LookupResponseMsg(lookup_id="r", num_hit_tokens=4)
    assert msg.kind == ""


def test_admission_pending_is_not_a_miss_and_ready_is_positive():
    admission = TierAdmission()
    assert admission_tokens(admission.phase("r"), 0) is None

    assert admission.offer("r") is True
    assert admission.offer("r") is False
    assert admission.phase("r") == PHASE_PENDING
    assert admission_tokens(admission.phase("r"), 0) is None

    # Metadata hit only. HBM admission still waits.
    admission.note("r", "lookup", 128)
    assert admission.phase("r") == PHASE_PENDING
    tokens = admission_tokens(
        admission.phase("r"), admission.preloaded_tokens("r")
    )
    assert tokens is None

    admission.note("r", "reading", 0)
    assert admission.phase("r") == PHASE_PENDING

    admission.note("r", "preloaded", 128)
    assert admission.phase("r") == PHASE_READY
    tokens = admission_tokens(
        admission.phase("r"), admission.preloaded_tokens("r")
    )
    assert tokens == 128

    # A later ACK does not change a finished hit.
    admission.note("r", "lookup", 0)
    assert admission.phase("r") == PHASE_READY


def test_admission_lookup_zero_is_a_miss():
    admission = TierAdmission()
    admission.offer("r")
    admission.note("r", "lookup", 0)
    assert admission.phase("r") == PHASE_MISS
    assert admission_tokens(admission.phase("r"), 0) == 0
    assert admission.inflight() == 0


def test_legacy_empty_kind_means_the_body_is_in_host():
    admission = TierAdmission()
    admission.offer("r")
    admission.note("r", "", 32)
    assert admission.phase("r") == PHASE_READY
    assert admission.preloaded_tokens("r") == 32


def test_worker_miss_does_not_keep_a_get_and_unfit_stays_queued():
    queue = TierWorkerQueue()
    accepted, replaced = queue.note_lookup("miss", 0, None)
    assert accepted is True
    assert replaced is None
    assert queue.take_next_fitting(lambda _job: True) is None
    assert queue.has_waiting() is False

    job = _load()
    queue.note_lookup("hit", 4, job)
    assert queue.take_next_fitting(lambda _job: False) is None
    assert queue.has_waiting() is True
    assert queue.drop_if_waiting("hit") is job
    assert queue.has_waiting() is False


def test_worker_claims_the_oldest_prefix_that_fits():
    queue = TierWorkerQueue()
    large = _load(4)
    small = _load(1)
    queue.note_lookup("large", 16, large)
    queue.note_lookup("small", 4, small)

    claimed = queue.take_next_fitting(lambda job: job.num_chunks <= 1)
    assert claimed is not None
    assert claimed[0] == "small"
    assert queue.has_waiting() is True
    assert queue.drop_if_waiting("large") is large
    assert queue.is_reading("small") is True


class _Request:
    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.all_token_ids = tokens
        self.num_preemptions = 0
        self.priority = 0


class _Client:
    def __init__(self):
        self.tier_admission = TierAdmission()
        self.sent: list[str] = []

    def submit_lookup(self, token_ids, lookup_id, request_configs=None):
        del token_ids, request_configs
        if not self.tier_admission.offer(lookup_id):
            return False
        self.sent.append(lookup_id)
        return True


def test_offer_starts_lookup_before_hbm_and_does_not_pop_the_queue():
    waiting = [_Request("a", [1, 2]), _Request("b", [3, 4]), _Request("c", [5])]
    scheduler = SimpleNamespace(
        waiting=waiting, skipped_waiting=[], scheduler_config=None
    )
    client = _Client()

    def prepare(request):
        return list(request.all_token_ids), None

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen={})
    assert client.sent == ["a", "b"]
    assert waiting[0].request_id == "a"
    assert len(waiting) == 3
    assert client.tier_admission.phase("c") == "absent"

    offer_waiting_lookups(scheduler, client, prepare, window=2, preempt_seen={})
    assert client.sent == ["a", "b"]


def _storage_manager_cls():
    pytest.importorskip("lmcache.lmcache_native")
    from lmcache.v1.storage_backend.storage_manager import StorageManager

    return StorageManager


def test_split_flag_requires_async_loading():
    storage_manager = _storage_manager_cls()
    enabled = SimpleNamespace(
        split_metadata_lookup=True, enable_async_loading=True
    )
    sync_only = SimpleNamespace(
        split_metadata_lookup=True, enable_async_loading=False
    )
    off = SimpleNamespace(split_metadata_lookup=False, enable_async_loading=True)
    assert storage_manager._split_metadata_lookup_enabled(
        SimpleNamespace(config=enabled)
    )
    assert not storage_manager._split_metadata_lookup_enabled(
        SimpleNamespace(config=sync_only)
    )
    assert not storage_manager._split_metadata_lookup_enabled(
        SimpleNamespace(config=off)
    )


class _Backend:
    def __init__(self, hits: int):
        self.hits = hits
        self.gets = 0
        self.unpinned: list = []

    async def batched_async_contains(self, lookup_id, keys, pin):
        del lookup_id, pin
        return self.hits if self.hits <= len(keys) else len(keys)

    def unpin(self, key):
        self.unpinned.append(key)

    def batched_get_non_blocking(self, lookup_id, keys, transfer_spec=None):
        del lookup_id, keys, transfer_spec
        self.gets += 1
        raise AssertionError("GET must not start for this test")


class _Ack:
    def __init__(self):
        self.responses = []

    def send_response_to_scheduler(self, lookup_id, num_hit_tokens, kind=""):
        self.responses.append((lookup_id, num_hit_tokens, kind))


def _manager(split: bool, backend: _Backend):
    storage_manager = _storage_manager_cls()
    manager = storage_manager.__new__(storage_manager)
    manager.config = SimpleNamespace(
        split_metadata_lookup=split,
        enable_async_loading=True,
    )
    manager.tier_queue = TierWorkerQueue()
    manager.async_lookup_server = _Ack()
    manager._tier_abort = set()
    manager.loop = SimpleNamespace(is_running=lambda: False)
    manager.get_active_storage_backends = lambda search_range=None: [("disk", backend)]
    return manager


def test_flag_off_miss_is_one_ack_and_does_not_get():
    backend = _Backend(0)
    manager = _manager(False, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 0, "")]
    assert backend.gets == 0


def test_flag_on_miss_acks_lookup_and_does_not_get():
    backend = _Backend(0)
    manager = _manager(True, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 0, "lookup")]
    assert backend.gets == 0
    assert manager.tier_queue.has_waiting() is False


def test_flag_on_hit_stays_queued_until_it_fits():
    backend = _Backend(2)
    manager = _manager(True, backend)
    asyncio.run(
        manager.async_lookup_and_prefetch(
            "r", keys=[object(), object()], cum_chunk_lengths=[0, 4, 8]
        )
    )
    assert manager.async_lookup_server.responses == [("r", 8, "lookup")]
    assert backend.gets == 0
    assert manager.tier_queue.has_waiting() is True
    assert manager.tier_queue.take_next_fitting(lambda _job: False) is None
    claimed = manager.tier_queue.take_next_fitting(lambda _job: True)
    assert claimed is not None
    assert claimed[0] == "r"
    assert claimed[1].num_chunks == 2

# SPDX-License-Identifier: Apache-2.0
"""Completion means the real file strategy has finished, even with an empty queue."""

# Standard
import json
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.record_strategies.base import AsyncRecorder
from lmcache.v1.lookup_client.record_strategies.file_hash import FileHashStrategy

pytestmark = pytest.mark.no_shared_allocator


def _file_strategy(tmp_path):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=4,
        extra_config={"chunk_statistics_file_output_dir": str(tmp_path)},
    )
    return FileHashStrategy(config, chunk_size=4)


def test_idle_smoke_can_finish_without_work(tmp_path):
    recorder = AsyncRecorder(_file_strategy(tmp_path))
    try:
        assert recorder.wait_for_completion(timeout=0.1) is True
    finally:
        recorder.close()
    assert not recorder.async_worker_thread.is_alive()


@pytest.mark.parametrize(
    "boundary,preprocess_in_caller",
    [("preprocess", False), ("record", False), ("record", True)],
)
def test_completion_waits_for_in_flight_file_work(
    tmp_path, monkeypatch, boundary, preprocess_in_caller
):
    strategy = _file_strategy(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = getattr(strategy, boundary)

    def gated_file_operation(*args):
        entered.set()
        assert release.wait(timeout=5), "test did not release the strategy operation"
        return original(*args)

    # Delay the strategy dependency while retaining actual hashing and JSONL I/O.
    monkeypatch.setattr(strategy, boundary, gated_file_operation)
    recorder = AsyncRecorder(strategy, preprocess_in_caller=preprocess_in_caller)
    try:
        recorder.record_async(list(range(8)), "last-in-flight")
        assert entered.wait(timeout=2)
        assert recorder.wait_for_completion(timeout=0.05) is False
        release.set()
        assert recorder.wait_for_completion(timeout=2) is True
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        entry = json.loads(files[0].read_text())
        assert entry["lookup_id"] == "last-in-flight"
        assert len(entry["chunk_hashes"]) == 2
        assert strategy.get_statistics()["total_chunks"] == 2
    finally:
        release.set()
        recorder.close()
    assert not recorder.async_worker_thread.is_alive()


@pytest.mark.parametrize("boundary", ["preprocess", "record"])
def test_failed_file_job_does_not_block_following_completion(
    tmp_path, monkeypatch, boundary
):
    strategy = _file_strategy(tmp_path)
    original = getattr(strategy, boundary)
    first_call = True
    completed_good_job = threading.Event()

    def fail_once_then_do_real_work(*args):
        nonlocal first_call
        if first_call:
            first_call = False
            raise OSError("injected file-strategy failure")
        return original(*args)

    original_record = strategy.record

    def record_and_notify(*args):
        original_record(*args)
        completed_good_job.set()

    monkeypatch.setattr(strategy, "record", record_and_notify)
    if boundary == "record":
        original = strategy.record
    monkeypatch.setattr(strategy, boundary, fail_once_then_do_real_work)
    recorder = AsyncRecorder(strategy)
    try:
        recorder.record_async(list(range(8)), "injected-failure")
        recorder.record_async(list(range(8)), "next-good-job")
        assert completed_good_job.wait(timeout=2)
        assert recorder.wait_for_completion(timeout=0.2) is True
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        entries = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert [entry["lookup_id"] for entry in entries] == ["next-good-job"]
        assert len(entries[0]["chunk_hashes"]) == 2
    finally:
        recorder.close()
    assert not recorder.async_worker_thread.is_alive()


def test_close_finishes_queued_file_work(tmp_path):
    recorder = AsyncRecorder(_file_strategy(tmp_path))
    try:
        recorder.record_async(list(range(8)), "first")
        recorder.record_async(list(range(8)), "second")
    finally:
        recorder.close()
    assert not recorder.async_worker_thread.is_alive()
    assert recorder.wait_for_completion(timeout=0) is True
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    entries = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [entry["lookup_id"] for entry in entries] == ["first", "second"]

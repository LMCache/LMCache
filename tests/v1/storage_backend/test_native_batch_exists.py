# SPDX-License-Identifier: Apache-2.0
"""Batch EXISTS error tolerance of the native ConnectorBase, exercised through
the real C++ filesystem connector."""

# Standard
from pathlib import Path
from typing import Any, Iterator
import select
import time

# Third Party
import pytest

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")

STORED_KEY = "model@0x0@0x0@0x1"
STORED_KEY_AFTER_FAILURE = "model@0x0@0x0@0x2"
MISSING_KEY = "model@0x0@0x0@0x3"
# A filename longer than NAME_MAX makes the filesystem existence check throw.
UNCHECKABLE_KEY = "m" * 300 + "@0x0@0x0@0x4"


@pytest.fixture
def fs_client(tmp_path: Path) -> Iterator[Any]:
    """Single-worker FS connector, so the whole batch runs as one tile."""
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", False, 0)
    yield client
    client.close()


def _wait_for_completion(
    client: Any, future_id: int, timeout_s: float = 5.0
) -> tuple[int, bool, str, list[bool] | None]:
    """Block until the completion for ``future_id`` arrives and return it."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        select.select([client.event_fd()], [], [], 0.1)
        for completion in client.drain_completions():
            if completion[0] == future_id:
                return completion
    raise TimeoutError(f"no completion for future {future_id}")


def test_batch_exists_reports_uncheckable_key_as_miss(fs_client: Any) -> None:
    """One key whose check fails must not fail or corrupt the rest of the batch."""
    stored = [STORED_KEY, STORED_KEY_AFTER_FAILURE]
    set_id = fs_client.submit_batch_set(
        stored, [memoryview(bytearray(64)) for _ in stored]
    )
    _, set_ok, set_error, _ = _wait_for_completion(fs_client, set_id)
    assert set_ok, set_error

    future_id = fs_client.submit_batch_exists(
        [STORED_KEY, UNCHECKABLE_KEY, STORED_KEY_AFTER_FAILURE, MISSING_KEY]
    )
    _, ok, error, per_key_results = _wait_for_completion(fs_client, future_id)

    assert ok, error
    # The hit after the failure proves the loop kept going past the exception.
    assert per_key_results == [True, False, True, False]

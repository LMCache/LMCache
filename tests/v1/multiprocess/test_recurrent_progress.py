# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.recurrent_progress import RecurrentCopyProgress


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_waits_for_all_worker_fences(tp) -> None:
    progress = RecurrentCopyProgress(set(range(tp)))
    for rank in range(tp):
        assert progress.start(rank)
    for rank in range(tp):
        assert progress.take_result() is None
        progress.finish(rank, succeeded=True)
    assert progress.take_result() is True
    assert progress.take_result() is None


def test_cancellation_waits_for_a_started_copy_and_skips_pending_workers() -> None:
    progress = RecurrentCopyProgress({0, 1})
    assert progress.start(0)
    progress.cancel()
    assert not progress.start(1)
    assert progress.take_result() is None

    progress.finish(0, succeeded=True)
    assert progress.take_result() is False

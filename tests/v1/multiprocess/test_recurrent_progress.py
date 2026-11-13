# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.v1.multiprocess.recurrent_progress import RecurrentCopyProgress


def test_waits_for_all_worker_fences() -> None:
    progress = RecurrentCopyProgress({0, 1})
    assert progress.start(0)
    assert progress.start(1)
    progress.finish(1, succeeded=True)
    assert progress.take_result() is None

    progress.finish(0, succeeded=True)
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

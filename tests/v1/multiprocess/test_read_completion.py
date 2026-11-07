# SPDX-License-Identifier: Apache-2.0
"""Device completion, rather than callback delivery or TTL, releases reads."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import read_completion as mod
from lmcache.v1.multiprocess.read_completion import ActiveReadCompletionTracker


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide controllable device events and drive polling synchronously."""
    device = MagicMock()
    device.record_completion_event.return_value.is_complete.return_value = False
    monkeypatch.setattr(mod, "platform_stream", device)
    monkeypatch.setattr(mod, "create_periodic_thread", MagicMock())
    return device


def test_unfinished_event_retains_handles_until_proven_complete(
    platform: MagicMock,
) -> None:
    """An early notification cannot free the CPU source of a queued transfer."""
    release = MagicMock()
    tracker = ActiveReadCompletionTracker(release)
    tracker.defer([(7, [1, 2])], "device", "stream")
    assert tracker.poll() == 0
    assert tracker.pending_count() == 1
    release.assert_not_called()
    platform.record_completion_event.return_value.is_complete.return_value = True
    assert tracker.poll() == 1
    release.assert_called_once_with([(7, [1, 2])])
    assert tracker.poll() == 0
    assert tracker.pending_count() == 0
    tracker.close()


@pytest.mark.parametrize("drain_fails", [False, True])
def test_record_failure_drains_before_release(
    platform: MagicMock, drain_fails: bool
) -> None:
    """Two device API failures retain the handles for an explicit recovery."""
    order: list[str] = []
    release = MagicMock(side_effect=lambda groups: order.append("release"))
    tracker = ActiveReadCompletionTracker(release)
    platform.record_completion_event.side_effect = RuntimeError("event failed")

    def drain(*args: object) -> None:
        order.append("drain")
        if drain_fails:
            raise RuntimeError("drain failed")

    platform.synchronize_stream.side_effect = drain
    if drain_fails:
        with pytest.raises(RuntimeError, match="drain failed"):
            tracker.defer([(7, [1])], "device", "stream")
        assert order == ["drain"]
        assert tracker.pending_count() == 1
        release.assert_not_called()
        platform.synchronize_stream.side_effect = lambda *args: order.append("drain")
    else:
        tracker.defer([(7, [1])], "device", "stream")
        assert order == ["drain", "release"]
        assert tracker.pending_count() == 0
    tracker.close()
    assert tracker.pending_count() == 0
    release.assert_called_once_with([(7, [1])])


def test_close_drains_pending_stream_before_storage_release(
    platform: MagicMock,
) -> None:
    """Shutdown can safely finish a batch even when its event is still pending."""
    order: list[str] = []
    release = MagicMock(side_effect=lambda groups: order.append("release"))
    tracker = ActiveReadCompletionTracker(release)
    tracker.defer([(7, [1])], "device", "stream")
    platform.synchronize_stream.side_effect = lambda *args: order.append("drain")
    tracker.close()
    assert order == ["drain", "release"]
    assert tracker.pending_count() == 0


def test_failed_query_keeps_batch_for_retry(platform: MagicMock) -> None:
    """An error querying progress provides no evidence that reuse is safe."""
    release = MagicMock()
    tracker = ActiveReadCompletionTracker(release)
    event = platform.record_completion_event.return_value
    tracker.defer([(7, [1])], "device", "stream")
    event.is_complete.side_effect = RuntimeError("query failed")
    with pytest.raises(RuntimeError, match="query failed"):
        tracker.poll()
    assert tracker.pending_count() == 1
    release.assert_not_called()
    event.is_complete.side_effect = None
    event.is_complete.return_value = True
    assert tracker.poll() == 1
    tracker.close()

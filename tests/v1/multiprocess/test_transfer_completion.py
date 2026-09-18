# SPDX-License-Identifier: Apache-2.0
"""Tests for stream-ordered release of imported events and deferred replies."""

# Standard
from concurrent.futures import Future
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import transfer_completion as tc

_TARGET = tc.TransferStreams(device="dev", stream="stream", cupy_stream="cupy")


class _RecordingBackend:
    """Event backend double that records imports and waits.

    Implements the whole ``EventIPCBackend`` protocol; only ``import_event``
    and ``wait_event`` are exercised here.
    """

    device_type = "fake"

    def __init__(self, fail_wait: bool = False) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail_wait = fail_wait

    def check_event_support(self, device: object) -> None:
        return None

    def create_event(self, device: object) -> object:
        return ("local", device)

    def export_event(self, event: object, device: object) -> bytes:
        return b"exported"

    def import_event(self, handle: bytes, device: object) -> object:
        self.calls.append(("import", handle, device))
        return ("imported", handle)

    def record_event(self, event: object, stream: object) -> None:
        return None

    def wait_event(self, event: object, stream: object) -> None:
        self.calls.append(("wait", event, stream))
        if self.fail_wait:
            raise RuntimeError("wait failed")

    def query_event(self, event: object) -> bool:
        return True

    def synchronize_event(self, event: object, device: object) -> None:
        return None


@pytest.fixture
def submitted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, str, Any]]:
    """Capture stream callbacks instead of enqueuing them on a device."""
    calls: list[tuple[Any, str, Any]] = []
    monkeypatch.setattr(
        tc,
        "submit_callback_to_stream",
        lambda stream, kind, payload: calls.append((stream, kind, payload)),
    )
    return calls


def _drain(
    completion: tc.TransferCompletion, calls: list[tuple[Any, str, Any]]
) -> None:
    """Run each captured callback through the handler registered for its kind."""
    handlers: dict[str, tuple[Any, Any]] = {}
    completion.register_host_funcs(
        lambda kind, handler, payload_type: handlers.__setitem__(
            kind, (handler, payload_type)
        )
    )
    for _stream, kind, payload in calls:
        handler, payload_type = handlers[kind]
        assert isinstance(payload, payload_type)
        handler(payload)
    calls.clear()


def test_import_is_held_until_the_stream_callback_fires(submitted) -> None:
    completion = tc.TransferCompletion()
    backend = _RecordingBackend()

    completion.wait_for_producer(backend, b"h", _TARGET)

    assert backend.calls == [
        ("import", b"h", "dev"),
        ("wait", ("imported", b"h"), "stream"),
    ]
    assert submitted == [("cupy", tc.RELEASE_IMPORTED_EVENT_KIND, b"h")]
    assert completion.held_import_count() == 1

    _drain(completion, submitted)
    assert completion.held_import_count() == 0


def test_same_handle_is_released_once_per_consumed_wait(submitted) -> None:
    completion = tc.TransferCompletion()
    backend = _RecordingBackend()

    completion.wait_for_producer(backend, b"h", _TARGET)
    completion.wait_for_producer(backend, b"h", _TARGET)
    assert completion.held_import_count() == 2

    completion.release_imported_event(b"h")
    assert completion.held_import_count() == 1
    completion.release_imported_event(b"h")
    assert completion.held_import_count() == 0


def test_failed_wait_holds_nothing(submitted) -> None:
    completion = tc.TransferCompletion()
    backend = _RecordingBackend(fail_wait=True)

    with pytest.raises(RuntimeError, match="wait failed"):
        completion.wait_for_producer(backend, b"h", _TARGET)

    assert completion.held_import_count() == 0
    assert submitted == []


def test_release_of_unknown_handle_is_ignored() -> None:
    tc.TransferCompletion().release_imported_event(b"never-imported")


def test_reply_resolves_only_when_the_stream_callback_fires(submitted) -> None:
    completion = tc.TransferCompletion()

    reply = completion.reply_when_done(_TARGET, True)

    assert isinstance(reply, Future)
    assert not reply.done()
    assert completion.pending_reply_count() == 1
    assert [(stream, kind) for stream, kind, _payload in submitted] == [
        ("cupy", tc.RESOLVE_DEFERRED_REPLY_KIND)
    ]

    _drain(completion, submitted)
    assert reply.result(timeout=0) == (b"", True)
    assert completion.pending_reply_count() == 0


def test_replies_resolve_independently_in_callback_order(submitted) -> None:
    completion = tc.TransferCompletion()
    first = completion.reply_when_done(_TARGET, True)
    second = completion.reply_when_done(_TARGET, False)

    _drain(completion, submitted[:1])
    assert first.result(timeout=0) == (b"", True)
    assert not second.done()

    _drain(completion, submitted[1:])
    assert second.result(timeout=0) == (b"", False)


def test_resolve_of_unknown_reply_is_ignored() -> None:
    tc.TransferCompletion().resolve_reply(99)

# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
import gc
import weakref

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.platform.torch_ops import completion_recorder

_STREAM = SimpleNamespace(ptr=0)


class _Payload:
    pass


@pytest.fixture
def ops(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Route the dispatcher through the torch fallback recorder."""
    fallback = SimpleNamespace(
        record_completion_on_stream=completion_recorder.record_completion_on_stream,
        drain_recorded_completions=completion_recorder.drain_recorded_completions,
    )
    monkeypatch.setattr(native_completion, "_device_ops", fallback)
    completion_recorder.drain_recorded_completions()
    return fallback


def test_dispatch_passes_payload_through(ops: SimpleNamespace) -> None:
    """The handler gets the submitted object, and a payload with no handler
    is released after the drain."""
    seen: list[object] = []
    payload = {ObjectKey(b"chunk", "model", 0, 0): 1}
    orphan = _Payload()
    orphan_ref = weakref.ref(orphan)

    dispatcher = native_completion.DeviceHostFuncDispatcher()
    dispatcher.register("finish_write", seen.append)
    native_completion.submit_callback_to_stream(_STREAM, "unregistered", orphan)
    native_completion.submit_callback_to_stream(_STREAM, "finish_write", payload)
    dispatcher.stop()

    assert len(seen) == 1
    assert seen[0] is payload
    del orphan
    gc.collect()
    assert orphan_ref() is None


def test_failed_record_releases_payload(ops: SimpleNamespace) -> None:
    """A recorder that raises leaves no payload parked."""

    def fail(*args: object) -> None:
        raise RuntimeError("host func launch failed")

    ops.record_completion_on_stream = fail
    payload = _Payload()
    payload_ref = weakref.ref(payload)

    with pytest.raises(RuntimeError):
        native_completion.submit_callback_to_stream(_STREAM, "finish_write", payload)
    del payload
    gc.collect()
    assert payload_ref() is None


def test_unknown_handle_does_not_stop_drain(ops: SimpleNamespace) -> None:
    """Entries the dispatcher never parked are dropped and later ones still
    reach their handler."""
    seen: list[object] = []
    payload = _Payload()

    dispatcher = native_completion.DeviceHostFuncDispatcher()
    dispatcher.register("finish_write", seen.append)
    completion_recorder.record_completion_on_stream(0, "finish_write", b"\xff" * 8)
    completion_recorder.record_completion_on_stream(0, "finish_write", b"bad")
    native_completion.submit_callback_to_stream(_STREAM, "finish_write", payload)
    dispatcher.stop()

    assert len(seen) == 1
    assert seen[0] is payload

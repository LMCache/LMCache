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


class _Payload:
    pass


def test_dispatch_passes_payload_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler gets the submitted object, and a payload with no handler
    is released after the drain."""
    monkeypatch.setattr(
        native_completion,
        "_device_ops",
        SimpleNamespace(
            record_completion_on_stream=completion_recorder.record_completion_on_stream,
            drain_recorded_completions=completion_recorder.drain_recorded_completions,
        ),
    )
    completion_recorder.drain_recorded_completions()
    stream = SimpleNamespace(ptr=0)
    seen: list[object] = []
    payload = {ObjectKey(b"chunk", "model", 0, 0): 1}
    orphan = _Payload()
    orphan_ref = weakref.ref(orphan)

    dispatcher = native_completion.DeviceHostFuncDispatcher()
    dispatcher.register("finish_write", seen.append)
    native_completion.submit_callback_to_stream(stream, "unregistered", orphan)
    native_completion.submit_callback_to_stream(stream, "finish_write", payload)
    dispatcher.stop()

    assert len(seen) == 1
    assert seen[0] is payload
    del orphan
    gc.collect()
    assert orphan_ref() is None

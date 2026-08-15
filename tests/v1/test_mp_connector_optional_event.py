# SPDX-License-Identifier: Apache-2.0
"""The MP connector submit paths pass ``None`` when the device has no event IPC."""

# Standard
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# First Party
from lmcache.integration.vllm import lmcache_mp_connector  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)


class _RecordingEvent:
    """Minimal stand-in for a CUDA-like interprocess event."""

    def __init__(self, interprocess: bool = False):
        self.interprocess = interprocess
        self.recorded = False

    def record(self) -> None:
        self.recorded = True


def _connector(
    monkeypatch, direction: Literal["STORE", "RETRIEVE"]
) -> tuple[object, MagicMock]:
    """Build a connector with just the attributes the submit paths read."""
    conn = lmcache_mp_connector.LMCacheMPConnector.__new__(
        lmcache_mp_connector.LMCacheMPConnector
    )
    metadata = LMCacheMPConnectorMetadata()
    metadata.add_request_metadata(
        LMCacheMPRequestMetadata(
            request_id="req-0",
            direction=direction,
            op=LoadStoreOp(token_ids=[], block_ids=[[0]]),
        )
    )
    adapter = MagicMock()
    monkeypatch.setattr(
        conn, "_get_connector_metadata", lambda: metadata, raising=False
    )
    monkeypatch.setattr(conn, "worker_adapter", adapter, raising=False)
    monkeypatch.setattr(conn, "dispatcher", None, raising=False)
    return conn, adapter


def _set_event_ipc_backend(monkeypatch, backend: object | None) -> None:
    monkeypatch.setattr(
        lmcache_mp_connector,
        "current_device_spec",
        SimpleNamespace(event_ipc_backend=backend),
    )


@pytest.mark.parametrize("has_backend", [True, False])
def test_wait_for_save_event_follows_device_capability(monkeypatch, has_backend):
    conn, adapter = _connector(monkeypatch, "STORE")
    _set_event_ipc_backend(monkeypatch, object() if has_backend else None)
    monkeypatch.setattr(
        lmcache_mp_connector, "torch_dev", SimpleNamespace(Event=_RecordingEvent)
    )

    conn.wait_for_save()

    event = adapter.batched_submit_store_requests.call_args.args[2]
    if has_backend:
        assert isinstance(event, _RecordingEvent)
        assert event.interprocess and event.recorded
    else:
        assert event is None


@pytest.mark.parametrize("has_backend", [True, False])
def test_start_load_kv_event_follows_device_capability(monkeypatch, has_backend):
    conn, adapter = _connector(monkeypatch, "RETRIEVE")
    _set_event_ipc_backend(monkeypatch, object() if has_backend else None)
    monkeypatch.setattr(
        lmcache_mp_connector, "torch_dev", SimpleNamespace(Event=_RecordingEvent)
    )

    conn.start_load_kv(forward_context=None)

    event = adapter.batched_submit_retrieve_requests.call_args.args[2]
    if has_backend:
        assert isinstance(event, _RecordingEvent)
        assert event.interprocess and event.recorded
    else:
        assert event is None


def test_no_event_class_on_device_without_backend(monkeypatch):
    """The device module need not expose Event at all (e.g. torch_rbln)."""
    conn, adapter = _connector(monkeypatch, "STORE")
    _set_event_ipc_backend(monkeypatch, None)
    monkeypatch.setattr(lmcache_mp_connector, "torch_dev", SimpleNamespace())

    conn.wait_for_save()

    assert adapter.batched_submit_store_requests.call_args.args[2] is None

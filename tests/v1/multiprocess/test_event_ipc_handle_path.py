# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import MagicMock
import inspect

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture


class _FakeEventBackend:
    """Record event backend calls while returning opaque fake events."""

    device_type = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._next_event = 0

    def check_event_support(self, device: object) -> None:
        self.calls.append(("check", device))

    def create_event(self, device: object) -> object:
        event = ("local", self._next_event)
        self._next_event += 1
        self.calls.append(("create", device, event))
        return event

    def export_event(self, event: object, device: object) -> bytes:
        self.calls.append(("export", event, device))
        return b"completion-handle"

    def import_event(self, handle: bytes, device: object) -> object:
        event = ("remote", handle)
        self.calls.append(("import", handle, device, event))
        return event

    def record_event(self, event: object, stream: object) -> None:
        self.calls.append(("record", event, stream))

    def wait_event(self, event: object, stream: object) -> None:
        self.calls.append(("wait", event, stream))

    def query_event(self, event: object) -> bool:
        self.calls.append(("query", event))
        return True

    def synchronize_event(self, event: object, device: object) -> None:
        self.calls.append(("synchronize", event, device))


class _NoopDispatcher:
    """Avoid starting native callback threads in the server unit test."""

    def register(self, kind: str, handler: object, payload_type: object) -> None:
        return None

    def start(self) -> None:
        return None


class _FakeStorageManager:
    """Minimal storage surface used by the server handle-path test."""

    def finish_write(self, keys: list[object]) -> None:
        return None

    def finish_read_prefetched(self, keys: list[object]) -> None:
        return None

    def reserve_write(
        self,
        keys: list[object],
        layout: object,
        mode: str,
    ) -> dict[object, object]:
        return {}

    @contextmanager
    def read_prefetched_results(self, keys: list[object]) -> Iterator[list[object]]:
        yield []


def _resolved_future(result: object) -> MessagingFuture[object]:
    """Return a messaging future already resolved to ``result``."""
    future: MessagingFuture[object] = MessagingFuture()
    future.set_result(result)
    return future


def test_worker_exports_events_through_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store and retrieve send backend-exported handles and keep device context."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(
        worker_transfer,
        "get_event_ipc_backend",
        lambda device: backend,
    )
    monkeypatch.setattr(
        worker_transfer,
        "wrap_kv_caches",
        lambda kv_caches: list(kv_caches.values()),
    )

    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(True)
    client.store.return_value = MessagingFuture()
    client.retrieve.return_value = MessagingFuture()

    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    kv_caches = {"layer_0": torch.empty(1)}
    context.register(
        kv_caches,
        "model",
        1,
        1,
        1.0,
    )
    unregister_future = context.unregister()
    stream = MagicMock(name="current_stream")
    monkeypatch.setattr(worker_transfer.torch_dev, "current_stream", lambda: stream)
    event = context.create_recorded_event()

    store_future = context.submit_store(
        "request",
        "key",
        kv_caches,
        [[0]],
        event,
        1,
    )
    retrieve_future = context.submit_retrieve(
        "request",
        "key",
        kv_caches,
        [[0]],
        event,
        1,
        skip_first_n_tokens=2,
    )

    assert isinstance(store_future, DeviceMessagingFuture)
    assert isinstance(retrieve_future, DeviceMessagingFuture)
    assert unregister_future is client.unregister_kv_cache.return_value
    client.unregister_kv_cache.assert_called_once_with(1)
    client.store.assert_called_once_with("key", 1, [[0]], b"completion-handle")
    client.retrieve.assert_called_once_with("key", 1, [[0]], b"completion-handle", 2)
    assert [call[0] for call in backend.calls] == [
        "check",
        "create",
        "record",
        "export",
        "export",
    ]
    assert backend.calls[2][2] is stream
    device = torch.device("cpu")
    assert backend.calls[0][1] == device
    assert backend.calls[1][1] == device
    assert all(call[-1] == device for call in backend.calls[3:])


def test_server_holds_import_and_defers_reply_until_stream_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server imports and waits through the backend, records no event of its
    own, and releases both the import and the reply from callbacks queued on
    the transfer stream."""
    # First Party
    from lmcache.v1.multiprocess import transfer_completion
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    backend = _FakeEventBackend()
    submitted: list[tuple[Any, str, Any]] = []
    monkeypatch.setattr(
        transfer_completion,
        "submit_callback_to_stream",
        lambda stream, kind, payload: submitted.append((stream, kind, payload)),
    )
    completion = transfer_completion.TransferCompletion()

    def unexpected_backend_lookup(_device: object) -> _FakeEventBackend:
        raise AssertionError("server should reuse the registered event backend")

    monkeypatch.setattr(
        lmcache_driven_transfer,
        "get_event_ipc_backend",
        unexpected_backend_lookup,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "DeviceHostFuncDispatcher",
        _NoopDispatcher,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "downsample_and_stage_block_ids",
        lambda cache_context, block_ids: block_ids,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "get_layout_desc",
        lambda cache_context, num_tokens, object_group_id: object(),
    )
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "transfer_kv_per_object_group",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev,
        "device",
        lambda device: nullcontext(),
    )
    monkeypatch.setattr(
        lmcache_driven_transfer.torch_dev,
        "stream",
        lambda stream: nullcontext(),
    )

    storage_manager = _FakeStorageManager()
    server_context = SimpleNamespace(
        chunk_size=1,
        storage_manager=storage_manager,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [[]],
        transfer_completion=completion,
    )
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
        cast(Any, server_context)
    )
    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream="transfer-stream",
        cupy_stream="cupy-stream",
        max_batch_size=1,
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            num_kernel_groups=1,
            object_groups=[SimpleNamespace(kernel_group_indices=[0])],
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1], group_kinds=()
            ),
        ),
        calculate_num_blocks=lambda chunk_size, group_idx: 1,
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cast(Any, cache_context),
        model_name="model",
        world_size=1,
        event_backend=cast(Any, backend),
    )
    monkeypatch.setattr(
        module,
        "get_and_touch_context_entry",
        lambda instance_id: entry,
    )
    key = SimpleNamespace(request_id="request", cache_salt="", worker_id=0)

    store_reply = module.store(key, 1, [[]], b"store-producer")
    retrieve_reply = module.retrieve(key, 1, [[]], b"retrieve-producer")

    # Both replies wait for the stream; nothing has been sent yet.
    assert isinstance(store_reply, Future) and not store_reply.done()
    assert isinstance(retrieve_reply, Future) and not retrieve_reply.done()

    # The server only imports and waits; it creates, records, exports nothing.
    assert [call[0] for call in backend.calls] == ["import", "wait"] * 2
    imported_handles = [call[1] for call in backend.calls if call[0] == "import"]
    waited_handles = [call[1][1] for call in backend.calls if call[0] == "wait"]
    assert imported_handles == [b"store-producer", b"retrieve-producer"]
    assert waited_handles == [b"store-producer", b"retrieve-producer"]
    assert all(
        call[2] == "transfer-stream" for call in backend.calls if call[0] == "wait"
    )
    assert completion.held_import_count() == 2
    assert completion.pending_reply_count() == 2

    # Per transfer: release the import right behind the wait, then resolve
    # the reply behind the copy, all on the transfer stream.
    assert [(stream, kind) for stream, kind, _payload in submitted] == [
        ("cupy-stream", transfer_completion.RELEASE_IMPORTED_EVENT_KIND),
        ("cupy-stream", transfer_completion.RESOLVE_DEFERRED_REPLY_KIND),
    ] * 2

    # Drain the callbacks in stream order through the registered handlers.
    handlers: dict[str, Any] = {}
    completion.register_host_funcs(
        lambda kind, handler, payload_type: handlers.__setitem__(kind, handler)
    )
    for _stream, kind, payload in submitted:
        handlers[kind](payload)

    assert completion.held_import_count() == 0
    assert completion.pending_reply_count() == 0
    assert store_reply.result(timeout=0) == (b"", True)
    assert retrieve_reply.result(timeout=0) == (b"", False)


def test_handle_path_has_no_musa_specific_imports_or_branches() -> None:
    """The scoped multiprocess modules remain backend-neutral."""
    # First Party
    from lmcache.v1.multiprocess import futures
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    for module in (futures, lmcache_driven_transfer, worker_transfer):
        source = inspect.getsource(module)
        assert "lmcache.v1.platform.musa" not in source
        assert 'device.type == "musa"' not in source

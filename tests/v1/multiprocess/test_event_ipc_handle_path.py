# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import MagicMock
from unittest.mock import call as mock_call
import inspect

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.chunk_event_future import ChunkStoreResponse
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
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


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_chunk_store_discovery_and_unregister_drain(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Discover once, fall back when disabled, and ACK before unregister."""
    # First Party
    from lmcache.v1.multiprocess.chunk_event_future import (
        ChunkEventDeviceMessagingFuture,
    )
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda _: backend)
    monkeypatch.setattr(worker_transfer, "wrap_kv_caches", lambda kv: list(kv.values()))
    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(True)
    client.get_experimental.return_value = _resolved_future(
        ["chunk_store"] if enabled else []
    )
    client.store.return_value = _resolved_future((b"terminal", True))
    raw: MessagingFuture[ChunkStoreResponse] = MessagingFuture()
    client.store_with_chunk_events.return_value = raw
    ack: MessagingFuture[None] = MessagingFuture()
    client.release_chunk_store_events.return_value = ack
    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    caches = {"layer_0": torch.empty(1)}
    context.register(caches, "model", 1, 1, 0.01)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, list(range(16)), 0, 16)
    event = cast(worker_transfer.IPCEvent, object())
    futures = [
        context.submit_store_with_chunk_events("r", key, caches, [[1]], event, 1)
        for _ in range(2)
    ]
    client.get_experimental.assert_called_once()
    if not enabled:
        assert all(isinstance(f, DeviceMessagingFuture) for f in futures)
        assert client.store.call_count == 2
        client.store_with_chunk_events.assert_not_called()
        context.unregister()
        context.close()
        return

    assert all(isinstance(f, ChunkEventDeviceMessagingFuture) for f in futures)
    client.store.assert_not_called()
    raw.set_result((b"terminal", [(b"chunk", 0, 16)], True, "lease"))
    # Unregister must not destroy exporter events while ACKs are outstanding.
    with pytest.raises(TimeoutError):
        context.unregister()
    client.unregister_kv_cache.assert_not_called()
    ack.set_result(None)
    context.unregister()
    context.close()
    assert all(f.result(0) for f in futures)
    client.unregister_kv_cache.assert_called_once_with(1)


def test_chunk_release_failure_does_not_lose_other_futures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue draining other leases, preserve failures, and retry on close."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda _: backend)
    monkeypatch.setattr(worker_transfer, "wrap_kv_caches", lambda kv: list(kv.values()))
    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(True)
    client.get_experimental.return_value = _resolved_future(["chunk_store"])
    raws: list[MessagingFuture[ChunkStoreResponse]] = [
        MessagingFuture(),
        MessagingFuture(),
    ]
    client.store_with_chunk_events.side_effect = raws
    failed_ack: MessagingFuture[None] = MessagingFuture()
    failed_ack.set_exception(RuntimeError("release failed"))
    second_ack: MessagingFuture[None] = MessagingFuture()
    retry_ack = _resolved_future(None)
    client.release_chunk_store_events.side_effect = [failed_ack, second_ack, retry_ack]
    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    caches = {"layer_0": torch.empty(1)}
    context.register(caches, "model", 1, 1, 0.01)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, list(range(16)), 0, 16)
    event = cast(worker_transfer.IPCEvent, object())
    first = context.submit_store_with_chunk_events("r1", key, caches, [[1]], event, 1)
    raws[0].set_result((b"terminal", [], True, "first"))
    # Polling cleanup for the previous failed ACK cannot swallow this store.
    second = context.submit_store_with_chunk_events("r2", key, caches, [[2]], event, 1)
    raws[1].set_result((b"terminal", [], True, "second"))
    assert first.result(0) and second.result(0)
    with pytest.raises(TimeoutError):
        context.close()
    second_ack.set_result(None)
    context.close()
    assert client.release_chunk_store_events.call_args_list == [
        mock_call(1, "first"),
        mock_call(1, "second"),
        mock_call(1, "first"),
    ]


@pytest.mark.parametrize("failure", ["rpc", "import"])
def test_chunk_store_cleanup_distinguishes_rpc_and_import_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Retire raw RPC failures, but retain handles after a failed event import."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda _: backend)
    monkeypatch.setattr(worker_transfer, "wrap_kv_caches", lambda kv: list(kv.values()))
    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(True)
    client.get_experimental.return_value = _resolved_future(["chunk_store"])
    client.release_chunk_store_events.return_value = _resolved_future(None)
    failed: MessagingFuture[ChunkStoreResponse] = MessagingFuture()
    client.store_with_chunk_events.return_value = failed
    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    caches = {"layer_0": torch.empty(1)}
    context.register(caches, "model", 1, 1, 0.01)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, list(range(16)), 0, 16)
    future = context.submit_store_with_chunk_events(
        "r", key, caches, [[1]], cast(worker_transfer.IPCEvent, object()), 1
    )
    if failure == "rpc":
        failed.set_exception(RuntimeError("response lost"))
    else:
        failed.set_result((b"terminal", [(b"chunk", 0, 16)], True, "lease"))
        importer = backend.import_event
        monkeypatch.setattr(
            backend,
            "import_event",
            MagicMock(side_effect=RuntimeError("import failed")),
        )
        with pytest.raises(RuntimeError, match="import failed"):
            context.unregister()
        client.unregister_kv_cache.assert_not_called()
        client.release_chunk_store_events.assert_not_called()
        monkeypatch.setattr(backend, "import_event", importer)
    context.unregister()
    context.close()
    client.unregister_kv_cache.assert_called_once_with(1)
    if failure == "rpc":
        client.release_chunk_store_events.assert_not_called()
        with pytest.raises(RuntimeError, match="response lost"):
            future.result(0)
    else:
        client.release_chunk_store_events.assert_called_once_with(1, "lease")
        assert future.result(0)


def test_server_store_and_retrieve_delegate_event_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server imports, waits, records, and exports through the event backend."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    backend = _FakeEventBackend()

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
        null_block_id=0,
        storage_manager=storage_manager,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [[]],
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

    assert module.store(key, 1, [[]], b"store-producer") == (
        b"completion-handle",
        True,
    )
    assert module.retrieve(key, 1, [[]], b"retrieve-producer") == (
        b"completion-handle",
        False,
    )

    imported_handles = [call[1] for call in backend.calls if call[0] == "import"]
    waited_handles = [call[1][1] for call in backend.calls if call[0] == "wait"]
    assert imported_handles == [b"store-producer", b"retrieve-producer"]
    assert waited_handles == [b"store-producer", b"retrieve-producer"]
    assert sum(call[0] == "record" for call in backend.calls) == 2
    assert sum(call[0] == "export" for call in backend.calls) == 2
    for index, call in enumerate(backend.calls):
        if call[0] == "export":
            assert backend.calls[index - 1][0] == "record"


def test_handle_path_has_no_musa_specific_imports_or_branches() -> None:
    """The scoped multiprocess modules remain backend-neutral."""
    # First Party
    from lmcache.v1.multiprocess import futures
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    for module in (futures, lmcache_driven_transfer, worker_transfer):
        source = inspect.getsource(module)
        assert "lmcache.v1.platform.devices.musa" not in source
        assert 'device.type == "musa"' not in source

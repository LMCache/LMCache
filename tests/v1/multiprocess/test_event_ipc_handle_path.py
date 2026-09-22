# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
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

    def __init__(self) -> None:
        self.uses_shared_l1 = False
        self.reserved: dict[object, object] = {}
        self.aborted: list[list[object]] = []
        self.finished: list[list[object]] = []

    def finish_write(self, keys: list[object]) -> None:
        self.finished.append(keys)

    def abort_write(self, keys: list[object]) -> None:
        self.aborted.append(keys)

    def finish_read_prefetched(self, keys: list[object]) -> None:
        return None

    def reserve_write(
        self,
        keys: list[object],
        layout: object,
    ) -> dict[object, object]:
        return self.reserved

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
    stream_events: list[Any] = []
    server_context = SimpleNamespace(
        chunk_size=1,
        storage_manager=storage_manager,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: stream_events.append(event),
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

    # Shared-L1 store: commit is a synchronous boundary — record, then
    # synchronize the event, then finish_write — so nothing becomes globally
    # readable before every D2H byte is on the device.
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "all_null_chunk_masks",
        lambda *args, **kwargs: [[False]],
    )
    shared_key = object()
    storage_manager.uses_shared_l1 = True
    storage_manager.reserved = {shared_key: SimpleNamespace(get_size=lambda: 64)}
    server_context.resolve_obj_keys = lambda key, group_ids: [[shared_key]]
    assert module.store(key, 1, [[0]], b"shared-store") == (
        b"completion-handle",
        True,
    )
    assert storage_manager.finished == [[shared_key]]
    assert [call[0] for call in backend.calls[-5:]] == [
        "import",
        "wait",
        "record",
        "synchronize",
        "export",
    ]
    assert stream_events[-1].metadata["stored_count"] == 1
    assert stream_events[-1].metadata["total_bytes"] == 64

    def fail_transfer(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected D2H failure")

    monkeypatch.setattr(
        lmcache_driven_transfer,
        "transfer_kv_per_object_group",
        fail_transfer,
    )
    assert module.store(key, 1, [[0]], b"failed-store") == (
        b"completion-handle",
        False,
    )
    assert storage_manager.aborted == [[shared_key]]

    record_failure_key = object()
    storage_manager.reserved = {
        record_failure_key: SimpleNamespace(get_size=lambda: 64)
    }
    server_context.resolve_obj_keys = lambda key, group_ids: [[record_failure_key]]
    record_event = backend.record_event

    def fail_record(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected record failure")

    monkeypatch.setattr(backend, "record_event", fail_record)
    with pytest.raises(RuntimeError, match="record failure"):
        module.store(key, 1, [[0]], b"failed-record")
    assert storage_manager.aborted[-1] == [record_failure_key]
    monkeypatch.setattr(backend, "record_event", record_event)

    synchronize_key = object()
    storage_manager.reserved = {synchronize_key: SimpleNamespace(get_size=lambda: 64)}
    server_context.resolve_obj_keys = lambda key, group_ids: [[synchronize_key]]
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "transfer_kv_per_object_group",
        lambda *args, **kwargs: None,
    )

    def fail_synchronize(event: object, device: object) -> None:
        raise RuntimeError("injected synchronization failure")

    monkeypatch.setattr(backend, "synchronize_event", fail_synchronize)
    assert module.store(key, 1, [[0]], b"failed-sync") == (
        b"completion-handle",
        False,
    )
    assert storage_manager.aborted[-1] == [synchronize_key]


def test_shared_l1_rejects_tp_above_one_at_registration() -> None:
    """Shared L1 M0 is TP=1 only: a TP>1 worker cannot register KV caches."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    storage_manager = _FakeStorageManager()
    storage_manager.uses_shared_l1 = True
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule.__new__(
        lmcache_driven_transfer.LMCacheDrivenTransferModule
    )
    module._ctx = cast(Any, SimpleNamespace(storage_manager=storage_manager))

    with pytest.raises(ValueError, match="TP=1"):
        module.register_kv_cache(
            instance_id=1,
            kv_caches=cast(Any, {}),
            model_name="model",
            world_size=2,
            engine_type=cast(Any, None),
            layout_hints=cast(Any, None),
            engine_group_infos=[],
        )


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

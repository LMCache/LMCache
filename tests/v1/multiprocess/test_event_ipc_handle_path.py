# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import ast
import inspect

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.transport.futures import DeviceMessagingFuture, MessagingFuture


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
    """Record registrations instead of starting native callback threads."""

    payload_types: dict[str, object] = {}

    def register(self, kind: str, handler: object, payload_type: object) -> None:
        _NoopDispatcher.payload_types[kind] = payload_type

    def start(self) -> None:
        return None


class _FakeStorageManager:
    """Minimal storage surface used by the server handle-path test."""

    def finish_write(
        self, keys: list[object], l1_owners: dict[object, int] | None = None
    ) -> None:
        return None

    def abort_write(self, keys: list[object], l1_owners: dict[object, int]) -> None:
        return None

    def finish_read_by_owner(self, batch: list[tuple[int, list[object]]]) -> None:
        return None

    def finish_read_prefetched(self, keys: list[object]) -> None:
        return None

    def reserve_write(
        self,
        keys: list[object],
        layout: object,
    ) -> dict[object, object]:
        return {}

    def prepare_read_completion(self, keys, l1_owners=None):
        return []

    def unsafe_read(self, keys, l1_owners=None):
        return [], []


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
    from lmcache.v1.client import worker_transfer

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
    monkeypatch.setattr(
        worker_transfer,
        "kv_wrapper_fds_delivered",
        lambda wrappers, fd_channel_path, timeout: nullcontext(),
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
        lmcache_chunk_tokens=256,
    )
    assert client.register_kv_cache.call_args.args[-1] == 256
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


def _server_handle_path(
    monkeypatch: pytest.MonkeyPatch, backend: Any
) -> SimpleNamespace:
    """A LMCacheDrivenTransferModule with GPU instance 1 registered on CPU fakes.

    Returns the module, the registration ``entry``, a request ``key``, and the
    stream ``callbacks``, held imports (``held``) and released import tokens
    (``released``) it records.
    """
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

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
    callbacks: list[tuple[str, object]] = []
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "submit_callback_to_stream",
        lambda stream, kind, payload: callbacks.append((kind, payload)),
    )
    held: list[tuple[str, bytes]] = []
    released: list[int] = []

    def hold_imported_event(event: tuple[str, bytes]) -> int:
        held.append(event)
        return len(held) - 1

    server_context = SimpleNamespace(
        chunk_size=1,
        null_block_id=0,
        storage_manager=storage_manager,
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [[]],
        layout_desc_registry=SimpleNamespace(unregister=lambda *args: None),
    )
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
        cast(Any, server_context)
    )
    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream="transfer-stream",
        cupy_stream="cupy-stream",
        hold_imported_event=hold_imported_event,
        release_imported_event=released.append,
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
        transfer_object_group=lambda *args, **kwargs: None,
        close=lambda: None,
    )
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cast(Any, cache_context),
        model_name="model",
        world_size=1,
        chunk_size=256,
        event_backend=cast(Any, backend),
    )
    # Registered directly: register_kv_cache needs real IPC handles.
    module._cache_contexts[1] = entry
    key = SimpleNamespace(
        request_id="request",
        cache_salt="",
        worker_id=0,
        has_unfull_chunk=lambda chunk_size: False,
    )

    return SimpleNamespace(
        module=module,
        entry=entry,
        key=key,
        callbacks=callbacks,
        held=held,
        released=released,
    )


def test_server_store_and_retrieve_delegate_event_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server imports, waits, records, and exports through the event backend."""
    backend = _FakeEventBackend()
    server = _server_handle_path(monkeypatch, backend)
    module, key = server.module, server.key
    callbacks, held, released = server.callbacks, server.held, server.released

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

    # Each import is held and its release is queued on the transfer stream
    # right behind the wait; the callback payload decodes as registered.
    assert held == [("remote", handle) for handle in imported_handles]
    assert callbacks == [
        ("release_imported_event", (1, 0)),
        ("release_imported_event", (1, 1)),
    ]
    for kind, payload in callbacks:
        decoder = msgspec.msgpack.Decoder(type=_NoopDispatcher.payload_types[kind])
        module._release_imported_event(decoder.decode(msgspec.msgpack.encode(payload)))
    assert released == [0, 1]
    assert sum(call[0] == "record" for call in backend.calls) == 2
    assert sum(call[0] == "export" for call in backend.calls) == 2
    for index, call in enumerate(backend.calls):
        if call[0] == "export":
            assert backend.calls[index - 1][0] == "record"


class _AbandonedFenceBackend(_FakeEventBackend):
    """A host-ready backend whose producer fence never completes; the first
    poll runs ``on_first_query`` (e.g. the reaper dropping the engine)."""

    host_ready_before_imported_wait = True

    def __init__(self) -> None:
        super().__init__()
        self.on_first_query = lambda: None

    def query_event(self, event: object) -> bool:
        self.calls.append(("query", event))
        if sum(call[0] == "query" for call in self.calls) == 1:
            self.on_first_query()
        return False


def test_server_fails_transfers_whose_engine_is_reaped_mid_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fence whose engine is reaped while the server polls it fails that
    STORE / RETRIEVE instead of holding the worker: no stream wait, no import
    held, a completion event still recorded, and the retrieve still releases
    the read locks of its range."""
    backend = _AbandonedFenceBackend()
    server = _server_handle_path(monkeypatch, backend)
    module, key = server.module, server.key
    # One chunk, so the retrieve has read locks to release.
    module.context.resolve_obj_keys = lambda key, group_ids: [["chunk-0"]]

    def reap() -> None:
        assert module.reap_stale_instances(-1.0, -1.0) == [1]

    backend.on_first_query = reap
    assert module.store_with_chunk_mask(key, 1, [[5]], b"store-producer") == (
        b"completion-handle",
        False,
        [],
    )
    assert [call[0] for call in backend.calls[-2:]] == ["record", "export"]

    module._cache_contexts[1] = server.entry
    backend.calls.clear()
    backend.on_first_query = reap
    assert module.retrieve(key, 1, [[5]], b"retrieve-producer") == (
        b"completion-handle",
        False,
    )
    assert [call[0] for call in backend.calls[-2:]] == ["record", "export"]

    assert not any(call[0] == "wait" for call in backend.calls)
    assert server.held == []
    assert [kind for kind, _ in server.callbacks] == ["finish_read_by_owner"]


def test_handle_path_has_no_musa_specific_imports_or_branches() -> None:
    """The scoped multiprocess modules remain backend-neutral."""
    # First Party
    from lmcache.v1.client import worker_transfer
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    from lmcache.v1.transport import futures

    for module in (futures, lmcache_driven_transfer, worker_transfer):
        source = inspect.getsource(module)
        assert "lmcache.v1.platform.devices.musa" not in source
        assert 'device.type == "musa"' not in source


def test_server_waits_on_imported_engine_events_poll_on_the_host() -> None:
    """Every server-side stream wait on an event imported from an engine goes
    through ``wait_producer_event``, so a host-ready backend (ROCm) never
    queues a GPU wait on a pending fence. The CacheBlend retrieve cannot run in
    a unit test, so this checks the call sites in the source."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
    from lmcache.v1.multiprocess.modules.blend import retrieve
    from lmcache.v1.multiprocess.plugin_modules import qstore

    routed = {}
    for module in (lmcache_driven_transfer, retrieve, qstore):
        calls = [
            node.func
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call)
        ]
        assert not [
            func
            for func in calls
            if isinstance(func, ast.Attribute) and func.attr == "wait_event"
        ], module.__name__
        routed[module.__name__.rsplit(".", 1)[-1]] = sum(
            isinstance(func, ast.Name) and func.id == "wait_producer_event"
            for func in calls
        )
    # STORE + RETRIEVE, the CacheBlend retrieve, STORE_Q + the SDK ring wait.
    assert routed == {"lmcache_driven_transfer": 2, "retrieve": 1, "qstore": 2}

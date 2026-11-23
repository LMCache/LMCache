# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
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


def _server_module(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a server transfer module whose device and storage work are fakes.

    Returns:
        The module, a cache context to register, the imports held by it, the
        stream callbacks queued and the import tokens released.
    """
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

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
        "get_unfull_chunk_layout_desc",
        lambda cache_context, object_group_id: object(),
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
        hash_algorithm_name="blake3",
        storage_manager=storage_manager,
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [[]],
        layout_desc_registry=SimpleNamespace(register=lambda *args, **kwargs: None),
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
        num_layers=1,
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
    )
    return SimpleNamespace(
        module=module,
        cache_context=cache_context,
        held=held,
        callbacks=callbacks,
        released=released,
    )


def _transfer_key() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request",
        cache_salt="",
        worker_id=0,
        has_unfull_chunk=lambda chunk_size: False,
    )


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
    server = _server_module(monkeypatch)
    module = server.module
    held, callbacks, released = server.held, server.callbacks, server.released
    entry = lmcache_driven_transfer.ContextEntry(
        cache_context=cast(Any, server.cache_context),
        model_name="model",
        world_size=1,
        event_backend=cast(Any, backend),
    )
    monkeypatch.setattr(
        module,
        "get_and_touch_context_entry",
        lambda instance_id: entry,
    )
    key = _transfer_key()

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


class _ReusingEventBackend(_FakeEventBackend):
    """Fake backend whose exported handles track every later record."""

    max_event_records = None

    def export_event(self, event: object, device: object) -> bytes:
        self.calls.append(("export", event, device))
        return b"handle-%d" % cast(tuple[str, int], event)[1]


def _negotiation_reply(server_agrees: bool | None) -> MessagingFuture[object]:
    """Server answer to NEGOTIATE_EVENT_REUSE; ``None`` is a server without it."""
    if server_agrees is not None:
        return _resolved_future(server_agrees)
    future: MessagingFuture[object] = MessagingFuture()
    future.set_exception(RuntimeError("Operation 'negotiate_event_reuse' unknown"))
    return future


@pytest.mark.parametrize(
    ("server_agrees", "producer_reused"),
    [(True, True), (False, False), (None, False)],
    ids=["new-server", "server-declines", "old-server"],
)
def test_worker_reuses_producer_events_only_if_server_agrees(
    monkeypatch: pytest.MonkeyPatch,
    server_agrees: bool | None,
    producer_reused: bool,
) -> None:
    """Producer events are recorded again only after the server agreed.

    Completion imports are kept whenever the worker offered reuse, so a
    server that agreed but whose answer was lost is still safe.
    """
    # First Party
    from lmcache.v1.client import worker_transfer

    backend = _ReusingEventBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda d: backend)
    monkeypatch.setattr(worker_transfer, "wrap_kv_caches", lambda kv: list(kv.values()))
    monkeypatch.setattr(
        worker_transfer,
        "kv_wrapper_fds_delivered",
        lambda wrappers, fd_channel_path, timeout: nullcontext(),
    )
    monkeypatch.setattr(worker_transfer.torch_dev, "current_stream", lambda: "stream")
    monkeypatch.setattr(worker_transfer, "_PRODUCER_EVENT_SLOTS", 1)
    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(None)
    client.negotiate_event_reuse.return_value = _negotiation_reply(server_agrees)
    client.store.side_effect = lambda *args: _resolved_future((b"completion", True))
    client.retrieve.side_effect = lambda *args: _resolved_future((b"completion", True))

    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    kv_caches = {"layer_0": torch.empty(1)}
    context.register(kv_caches, "model", 1, 1, 1.0)
    futures = []
    for _ in range(2):
        event = context.create_recorded_event()
        futures.append(
            context.submit_store("request", "key", kv_caches, [[0]], event, 1)
        )

    client.negotiate_event_reuse.assert_called_once_with(1, True)
    assert all(future.query() for future in futures)
    ops = [call[0] for call in backend.calls]
    assert ops.count("create") == (1 if producer_reused else 2)
    assert ops.count("export") == (1 if producer_reused else 2)
    # The same completion handle is imported once either way.
    assert ops.count("import") == 1
    producer_handles = [c.args[3] for c in client.store.call_args_list]
    assert (producer_handles[0] == producer_handles[1]) is producer_reused


def test_worker_without_reusable_events_does_not_negotiate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device whose handles do not track re-records keeps today's path."""
    # First Party
    from lmcache.v1.client import worker_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(worker_transfer, "get_event_ipc_backend", lambda d: backend)
    monkeypatch.setattr(worker_transfer, "wrap_kv_caches", lambda kv: list(kv.values()))
    monkeypatch.setattr(
        worker_transfer,
        "kv_wrapper_fds_delivered",
        lambda wrappers, fd_channel_path, timeout: nullcontext(),
    )
    client = MagicMock()
    client.register_kv_cache.return_value = _resolved_future(None)

    context = worker_transfer.LMCacheDrivenTransferContext(1, client)
    context.register({"layer_0": torch.empty(1)}, "model", 1, 1, 1.0)

    client.negotiate_event_reuse.assert_not_called()
    assert context._event_backend is backend


@pytest.mark.parametrize(
    "worker_offers",
    [True, False, None],
    ids=["new-worker", "worker-declines", "old-worker"],
)
def test_server_reuses_events_only_for_workers_that_offered(
    monkeypatch: pytest.MonkeyPatch,
    worker_offers: bool | None,
) -> None:
    """Producer imports are kept and completion events recorded again only
    for a worker that offered reuse; any other worker gets one event per
    operation and its handles imported anew. ``None`` is a worker that
    predates the negotiation and never sends it.
    """
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    backend = _ReusingEventBackend()
    monkeypatch.setattr(
        lmcache_driven_transfer, "get_event_ipc_backend", lambda device: backend
    )
    monkeypatch.setattr(lmcache_driven_transfer, "_COMPLETION_EVENT_SLOTS", 2)
    server = _server_module(monkeypatch)
    module = server.module
    monkeypatch.setattr(
        module,
        "_create_cache_context",
        lambda *args, **kwargs: server.cache_context,
    )
    module.register_kv_cache(1, [], "model", 1, None, None, [], None)
    reused = bool(worker_offers)
    if worker_offers is not None:
        assert module.negotiate_event_reuse(1, worker_offers) is reused
        if reused:
            # A recovering worker negotiates again for the same instance.
            entry = module.get_and_touch_context_entry(1)
            backend_before = entry.event_backend
            assert module.negotiate_event_reuse(1, True) is True
            assert entry.event_backend is backend_before
    key = _transfer_key()

    completions = [module.store(key, 1, [[]], b"reused")[0] for _ in range(3)]
    completions += [
        module.retrieve(key, 1, [[]], handle)[0] for handle in (b"new-1", b"new-2")
    ]

    imported = [call[1] for call in backend.calls if call[0] == "import"]
    creates = sum(call[0] == "create" for call in backend.calls)
    exports = sum(call[0] == "export" for call in backend.calls)
    assert sum(call[0] == "record" for call in backend.calls) == 5
    if reused:
        assert imported == [b"reused", b"new-1", b"new-2"]
        assert server.held[:3] == [server.held[0]] * 3
        assert completions == [b"handle-0", b"handle-1"] * 2 + [b"handle-0"]
        assert (creates, exports) == (2, 2)
    else:
        assert imported == [b"reused"] * 3 + [b"new-1", b"new-2"]
        assert completions == [b"handle-%d" % i for i in range(5)]
        assert (creates, exports) == (5, 5)


def test_server_declines_reuse_for_unknown_instance_or_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No registration, or a device whose handles snapshot one record: no reuse."""
    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer

    backend = _FakeEventBackend()
    monkeypatch.setattr(
        lmcache_driven_transfer, "get_event_ipc_backend", lambda device: backend
    )
    server = _server_module(monkeypatch)
    module = server.module
    monkeypatch.setattr(
        module,
        "_create_cache_context",
        lambda *args, **kwargs: server.cache_context,
    )
    assert module.negotiate_event_reuse(1, True) is False
    module.register_kv_cache(1, [], "model", 1, None, None, [], None)
    entry = module.get_and_touch_context_entry(1)
    assert module.negotiate_event_reuse(1, True) is False
    assert module.get_and_touch_context_entry(1).event_backend is entry.event_backend


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

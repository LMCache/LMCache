# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import inspect

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.client import worker_transfer
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey
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


@pytest.fixture
def worker_handle_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[worker_transfer.LMCacheDrivenTransferContext, MagicMock, _FakeEventBackend]:
    """A registered handle context with fake transport and event IPC boundaries."""
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
    return context, client, backend


def test_worker_exports_events_through_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
    worker_handle_context: tuple[
        worker_transfer.LMCacheDrivenTransferContext, MagicMock, _FakeEventBackend
    ],
) -> None:
    """Store and retrieve send backend-exported handles and keep device context."""
    context, client, backend = worker_handle_context
    kv_caches = {"layer_0": torch.empty(1)}
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


@pytest.mark.parametrize("has_rope", [False, True])
def test_blend_registers_tables_before_event_ordered_scatter(
    monkeypatch: pytest.MonkeyPatch,
    worker_handle_context: tuple[
        worker_transfer.LMCacheDrivenTransferContext, MagicMock, _FakeEventBackend
    ],
    has_rope: bool,
) -> None:
    """RoPE handles stay deliverable through ACK; scatter orders the consumer."""
    context, client, backend = worker_handle_context
    tables = [torch.empty(32, 8), torch.empty(32, 4)] if has_rope else []
    group_to_cache = [0, 1] if has_rope else [-1, -1]
    group_rot = [[0, 8], [0, 4]] if has_rope else [[], []]
    delivery: list[str] = []

    @contextmanager
    def deliver(
        wrappers: list[torch.Tensor], path: str, timeout: float
    ) -> Iterator[None]:
        assert len(wrappers) == len(tables)
        assert all(w is t for w, t in zip(wrappers, tables, strict=True))
        assert timeout == 2.0
        delivery.append("open")
        yield
        delivery.append("close")

    def register(*args: object) -> MessagingFuture[object]:
        assert delivery == ["open"]
        delivery.append("register")
        return _resolved_future(None)

    monkeypatch.setattr(worker_transfer, "kv_wrapper_fds_delivered", deliver)
    client.cb_register_rope.side_effect = register
    context.register_blend_rope(tables, 8, True, group_to_cache, group_rot, [8, 8], 2.0)
    assert delivery == ["open", "register", "close"]
    args = client.cb_register_rope.call_args.args
    assert args[0] == 1
    assert args[2:] == (8, True, group_to_cache, group_rot, [8, 8], "")
    key = IPCCacheServerKey.from_token_ids(
        "model", 1, 0, list(range(16)), 0, 16, "blend"
    )
    matches = [CBMatchResult(0, 8, 8, 16, b"hash")]
    raw: MessagingFuture[tuple[bytes, bool]] = MessagingFuture()
    client.cb_retrieve_pre_computed.return_value = raw
    event = cast(worker_transfer.IPCEvent, object())
    future = context.submit_blend_retrieve(key, matches, [[4, 5]], event)
    assert not future.query()
    client.cb_retrieve_pre_computed.assert_called_once_with(
        key, matches, [[4, 5]], 1, b"completion-handle"
    )
    raw.set_result((b"scatter-complete", True))
    consumer = object()
    assert future.wait_on_stream(consumer, timeout=0) is True
    assert (
        "import",
        b"scatter-complete",
        torch.device("cpu"),
        ("remote", b"scatter-complete"),
    ) in backend.calls
    assert ("wait", ("remote", b"scatter-complete"), consumer) in backend.calls
    assert not any(call[0] == "synchronize" for call in backend.calls)
    context.close()
    with pytest.raises(RuntimeError, match="register"):
        context.submit_blend_retrieve(key, matches, [[4, 5]], event)


def test_blend_rejects_retrieve_without_acknowledged_rope(
    worker_handle_context: tuple[
        worker_transfer.LMCacheDrivenTransferContext, MagicMock, _FakeEventBackend
    ],
) -> None:
    context, client, _backend = worker_handle_context
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, [], 0, 0, "blend")
    with pytest.raises(RuntimeError, match="register_blend_rope"):
        context.submit_blend_retrieve(key, [], [[]], None)
    failed: MessagingFuture[None] = MessagingFuture()
    failed.set_exception(ValueError("invalid rope geometry"))
    client.cb_register_rope.return_value = failed
    with pytest.raises(ValueError, match="invalid rope"):
        context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    with pytest.raises(RuntimeError, match="register_blend_rope"):
        context.submit_blend_retrieve(key, [], [[]], None)
    client.cb_register_rope.return_value = _resolved_future(None)
    context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    with pytest.raises(RuntimeError, match="IPC event"):
        context.submit_blend_retrieve(key, [], [[]], None)
    client.cb_register_rope.return_value = failed
    with pytest.raises(ValueError, match="invalid rope"):
        context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    with pytest.raises(RuntimeError, match="register_blend_rope"):
        context.submit_blend_retrieve(key, [], [[]], None)
    client.cb_retrieve_pre_computed.assert_not_called()


@pytest.mark.parametrize(
    "registration_failed,unregister_error",
    [(False, False), (True, False), (False, True)],
)
def test_blend_unregister_waits_for_rope_then_kv_ack(
    worker_handle_context: tuple[
        worker_transfer.LMCacheDrivenTransferContext, MagicMock, _FakeEventBackend
    ],
    registration_failed: bool,
    unregister_error: bool,
) -> None:
    context, client, _backend = worker_handle_context
    registration: MessagingFuture[None] = MessagingFuture()
    client.cb_register_rope.return_value = registration
    if registration_failed:
        registration.set_exception(ConnectionError("registration response lost"))
        with pytest.raises(ConnectionError):
            context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    else:
        registration.set_result(None)
        context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    rope, kv = MessagingFuture[None](), MessagingFuture[None]()
    client.cb_unregister_rope.return_value = rope
    client.unregister_kv_cache.return_value = kv
    completion = context.unregister()
    assert completion is not None and not completion.query()
    client.cb_unregister_rope.assert_called_once_with(1)
    client.unregister_kv_cache.assert_not_called()
    if unregister_error:
        rope.set_exception(ConnectionError("rope cleanup response lost"))
    else:
        rope.set_result(None)
    client.unregister_kv_cache.assert_called_once_with(1)
    assert not completion.query()
    kv.set_result(None)
    if unregister_error:
        with pytest.raises(ConnectionError, match="rope cleanup"):
            completion.result(timeout=0)
    else:
        assert completion.result(timeout=0) is None


def test_engine_driven_context_rejects_blend_before_sending_rpcs() -> None:
    client = MagicMock()
    context = worker_transfer.EngineDrivenTransferContext(1, client)
    key = IPCCacheServerKey.from_token_ids("model", 1, 0, [], 0, 0, "blend")
    with pytest.raises(NotImplementedError, match="CacheBlend registration"):
        context.register_blend_rope([], 8, True, [-1], [[]], [8], 1.0)
    with pytest.raises(NotImplementedError, match="CacheBlend retrieve"):
        context.submit_blend_retrieve(key, [], [[]], None)
    client.cb_register_rope.assert_not_called()
    client.cb_retrieve_pre_computed.assert_not_called()


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
    key = SimpleNamespace(
        request_id="request",
        cache_salt="",
        worker_id=0,
        has_unfull_chunk=lambda chunk_size: False,
    )

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

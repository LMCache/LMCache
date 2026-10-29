# SPDX-License-Identifier: Apache-2.0
"""Tests for platform event IPC use in the LMCache-driven handle path."""

# Standard
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import inspect

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache import device_ops
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ReadReservation
from lmcache.v1.distributed.storage_manager import (
    L1ReadGenerationCompletion,
    L1ReadTransferCompletion,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_hasher import TokenHasher


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
    handlers: dict[str, Callable[[Any], None]] = {}

    def register(
        self, kind: str, handler: Callable[[Any], None], payload_type: object
    ) -> None:
        _NoopDispatcher.payload_types[kind] = payload_type
        _NoopDispatcher.handlers[kind] = handler

    def start(self) -> None:
        return None


class _FakeStorageManager:
    """Minimal storage surface used by the server handle-path test."""

    def __init__(self, objects: dict[ObjectKey, MemoryObj]) -> None:
        self.objects = objects
        self.pin_calls: list[tuple[L1ReadGenerationCompletion, str]] = []
        self.read_completions: list[L1ReadTransferCompletion] = []
        self.unpinned_releases: list[tuple[L1ReadReservation, ...]] = []

    def finish_write(self, keys: list[object]) -> None:
        return None

    def finish_write_by_owner(self, batch: list[tuple[int, list[object]]]) -> None:
        """Accept the owner-tagged callback registered by the transfer module."""
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

    def prepare_read_generation_completion(
        self, reservations: tuple[L1ReadReservation, ...]
    ) -> L1ReadGenerationCompletion:
        """Preserve this fixture's captured owner/key/generation receipt."""
        assert all(item.owner == 0 and item.count == 1 for item in reservations)
        if not reservations:
            return ()
        return ((0, tuple((item.key, item.generation) for item in reservations)),)

    def pin_read_buffers(
        self, completion: L1ReadGenerationCompletion, tag: str
    ) -> dict[ObjectKey, MemoryObj]:
        """Record the pin before any device transfer completion is dispatched."""
        self.pin_calls.append((completion, tag))
        return {
            key: self.objects[key]
            for _owner, reservations in completion
            for key, _generation in reservations
        }

    def finish_read_reservations(
        self, reservations: tuple[L1ReadReservation, ...]
    ) -> None:
        """Record pre-pin failure cleanup so the test can reject unexpected errors."""
        self.unpinned_releases.append(reservations)

    def finish_read_transfer(self, completion: L1ReadTransferCompletion) -> None:
        """Record receipt release only when its stream callback is dispatched."""
        self.read_completions.append(completion)


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


@pytest.mark.parametrize("cached", [False, True])
def test_server_store_and_retrieve_delegate_event_ordering(
    monkeypatch: pytest.MonkeyPatch, cached: bool
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

    obj_key = ObjectKey(b"chunk", "model", 0)
    memory_obj = MagicMock(spec=MemoryObj)
    memory_obj.get_size.return_value = 1
    storage_manager = _FakeStorageManager({obj_key: memory_obj} if cached else {})
    key = IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=(1,),
        start=0,
        end=1,
        request_id="request",
    )
    session = Session(
        key.request_id, TokenHasher(chunk_size=1, hash_algorithm="blake3")
    )
    session.begin_lookup(key.no_worker_id_version(), (-1,))
    session.record_prefetch_result(
        int(cached),
        (0,),
        {obj_key: 0} if cached else {},
        read_generations={obj_key: 7} if cached else {},
    )
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
        session_manager=SimpleNamespace(get=lambda request_id: session),
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None,
            publish_on_stream=lambda stream, event: None,
            has_subscribers=lambda event_type: False,
        ),
        resolve_obj_keys=lambda key, group_ids: [
            [obj_key] if cached and key.end else []
        ],
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
    assert module.store(replace(key, end=0), 1, [[]], b"store-producer") == (
        b"completion-handle",
        True,
    )
    assert module.retrieve(key, 1, [[0]] if cached else [[]], b"retrieve-producer") == (
        b"completion-handle",
        cached,
    )

    imported_handles = [call[1] for call in backend.calls if call[0] == "import"]
    waited_handles = [call[1][1] for call in backend.calls if call[0] == "wait"]
    assert imported_handles == [b"store-producer", b"retrieve-producer"]
    assert waited_handles == [b"store-producer", b"retrieve-producer"]

    # Each import is held and its release is queued on the transfer stream
    # right behind the wait; the callback payload decodes as registered.
    assert held == [("remote", handle) for handle in imported_handles]
    assert callbacks[:2] == [
        ("release_imported_event", (1, 0)),
        ("release_imported_event", (1, 1)),
    ]
    completion: L1ReadGenerationCompletion = ((0, ((obj_key, 7),)),) if cached else ()
    assert len(storage_manager.pin_calls) == 1
    assert storage_manager.pin_calls[0][0] == completion
    transfer_tag = storage_manager.pin_calls[0][1]
    expected_reads = [(transfer_tag, completion)] if cached else []
    assert callbacks[2:] == [("finish_read_transfer", item) for item in expected_reads]
    assert storage_manager.unpinned_releases == []
    assert storage_manager.read_completions == []
    for kind, payload in callbacks:
        decoder = msgspec.msgpack.Decoder(type=_NoopDispatcher.payload_types[kind])
        _NoopDispatcher.handlers[kind](decoder.decode(msgspec.msgpack.encode(payload)))
    assert released == [0, 1]
    assert storage_manager.read_completions == expected_reads
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


@pytest.mark.parametrize("num_contexts", [0, 2])
def test_close_delivers_pending_stream_callbacks(
    monkeypatch: pytest.MonkeyPatch, num_contexts: int
) -> None:
    """Close delivers completions from every stream while their contexts exist."""
    ready: list[tuple[str, bytes]] = []
    pending: dict[int, list[tuple[str, bytes]]] = {}

    def drain() -> list[tuple[str, bytes]]:
        result = ready.copy()
        ready.clear()
        return result

    monkeypatch.setattr(device_ops, "drain_recorded_completions", drain)
    # Hold the background consumer so close must perform the final real drain.
    monkeypatch.setattr(
        native_completion.DeviceHostFuncDispatcher, "start", lambda self: None
    )
    contexts = []
    for instance_id in range(num_contexts):
        pending[instance_id] = [
            ("release_imported_event", msgspec.msgpack.encode((instance_id, 7)))
        ]

        def complete(instance_id: int = instance_id) -> None:
            ready.extend(pending.pop(instance_id, []))

        context = MagicMock()
        context.device = torch.device("cpu")
        context.kv_layer_groups_manager.num_object_groups = 1
        context.stream.synchronize.side_effect = complete
        context.close.side_effect = complete
        contexts.append(context)

    monkeypatch.setattr(
        lmcache_driven_transfer, "create_cache_context", MagicMock(side_effect=contexts)
    )
    monkeypatch.setattr(lmcache_driven_transfer, "get_layout_desc", MagicMock())
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "get_event_ipc_backend",
        lambda device: _FakeEventBackend(),
    )
    monkeypatch.setattr(lmcache_driven_transfer, "torch_dev", MagicMock())
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(MagicMock())
    for instance_id in range(num_contexts):
        module.register_kv_cache(
            instance_id, [], "close-test", 1, EngineType.VLLM, {}, []
        )

    module.close()

    assert module.tracked_instance_count() == 0
    assert ready == []
    for context in contexts:
        context.release_imported_event.assert_called_once_with(7)
        context.close.assert_called_once_with()


def test_close_preserves_contexts_when_stream_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stream wait must not release possibly active device resources."""
    dispatcher = MagicMock()
    monkeypatch.setattr(
        lmcache_driven_transfer, "DeviceHostFuncDispatcher", lambda: dispatcher
    )
    contexts = [MagicMock(), MagicMock()]
    for context in contexts:
        context.device = torch.device("cpu")
        context.kv_layer_groups_manager.num_object_groups = 1
    contexts[0].stream.synchronize.side_effect = RuntimeError("stream sync failed")
    monkeypatch.setattr(
        lmcache_driven_transfer, "create_cache_context", MagicMock(side_effect=contexts)
    )
    monkeypatch.setattr(lmcache_driven_transfer, "get_layout_desc", MagicMock())
    monkeypatch.setattr(
        lmcache_driven_transfer,
        "get_event_ipc_backend",
        lambda device: _FakeEventBackend(),
    )
    monkeypatch.setattr(lmcache_driven_transfer, "torch_dev", MagicMock())
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(MagicMock())
    for instance_id in range(2):
        module.register_kv_cache(
            instance_id, [], "close-sync-error", 1, EngineType.VLLM, {}, []
        )

    with pytest.raises(RuntimeError, match="stream sync failed"):
        module.close()

    assert module.tracked_instance_count() == 2
    dispatcher.stop.assert_not_called()
    for context in contexts:
        context.close.assert_not_called()

    # The injected error is transient; this does not simulate device recovery.
    contexts[0].stream.synchronize.side_effect = None
    module.close()
    assert module.tracked_instance_count() == 0
    dispatcher.stop.assert_called_once_with()
    for context in contexts:
        context.close.assert_called_once_with()

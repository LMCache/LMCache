# SPDX-License-Identifier: Apache-2.0
"""Chunk-store public contracts, including exporter destruction and late import."""

# Standard
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, cast
from unittest.mock import MagicMock
import gc
import itertools
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.chunk_event_future import (
    ChunkEventDeviceMessagingFuture,
    ChunkStoreResponse,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer_mod
from lmcache.v1.multiprocess.modules.experimental import chunk_store as mod
from lmcache.v1.multiprocess.modules.experimental.chunk_store import ChunkStoreModule
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


class _Event:
    """Torch-style events whose imports do not keep their exporters alive."""

    exporters: ClassVar[weakref.WeakValueDictionary[bytes, "_Event"]] = (
        weakref.WeakValueDictionary()
    )
    sequence: ClassVar[Iterator[int]] = itertools.count()
    record_sequence: ClassVar[Iterator[int]] = itertools.count()

    def __init__(self, interprocess: bool = True) -> None:
        self.handle = str(next(self.sequence)).encode()
        self.ready = False
        self.order = -1
        self.exporters[self.handle] = self

    def ipc_handle(self) -> bytes:
        return self.handle

    @classmethod
    def from_ipc_handle(cls, device: object, handle: bytes) -> "_Event":
        if handle not in cls.exporters:
            raise ReferenceError("exporter already destroyed")
        imported = cls.__new__(cls)
        imported.handle = handle
        return imported

    def record(self, stream: object) -> None:
        self.order = next(self.record_sequence)

    def wait(self, stream: object) -> None:
        return None

    def query(self) -> bool:
        try:
            return self.exporters[self.handle].ready
        except KeyError as exc:
            raise ReferenceError("exporter already destroyed") from exc

    def synchronize(self) -> None:
        order = self.exporters[self.handle].order
        for event in self.exporters.values():
            if 0 <= event.order <= order:
                event.ready = True

    @classmethod
    def complete(cls, handle: bytes) -> None:
        cls.exporters[handle].ready = True


@dataclass
class _Harness:
    module: ChunkStoreModule
    backend: DefaultEventIPCBackend
    storage: MagicMock
    transfers: list[tuple[int, list[list[int]], list[list[int]]]]
    callbacks: list[tuple[str, list[str]]]
    producer: _Event

    def store(self) -> ChunkStoreResponse:
        return self.module.store_with_chunk_events(
            cast(
                IPCCacheServerKey,
                SimpleNamespace(request_id="r", worker_id=1, start=8, end=16),
            ),
            1,
            [[1, 2, 3, 4, 5, 6, 7, 8], [-1, -1, -1, -1, 10, 11, 12, 13]],
            self.producer.ipc_handle(),
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    # Only metadata/storage are mocked. GPU event mocks would accidentally keep
    # exporters alive through call_args and hide precisely the lifetime bug.
    backend = DefaultEventIPCBackend(SimpleNamespace(Event=_Event), "fake")
    producer = _Event()
    producer.record(None)
    storage = MagicMock()
    storage.reserve_write.side_effect = lambda keys, layout: {
        k: SimpleNamespace(get_size=lambda: 16) for k in keys
    }
    ctx = SimpleNamespace(
        chunk_size=4,
        null_block_id=-1,
        storage_manager=storage,
        event_bus=MagicMock(),
        resolve_obj_keys=lambda key, groups: [["a0", "a1"], ["b0", "b1"]],
    )
    groups = SimpleNamespace(
        num_kernel_groups=2,
        num_object_groups=2,
        object_groups=[
            SimpleNamespace(kernel_group_indices=[0]),
            SimpleNamespace(kernel_group_indices=[1]),
        ],
        get_subchunk_sw_size_tokens=lambda group: 4 if group == 0 else 2,
    )
    cache = SimpleNamespace(
        device="cpu",
        stream=None,
        cupy_stream=None,
        lmcache_tokens_per_chunk=4,
        kv_layer_groups_manager=groups,
        calculate_num_blocks=lambda tokens, group: tokens,
        stage_block_ids=lambda ids: [torch.tensor(group) for group in ids],
    )
    entry = transfer_mod.ContextEntry(
        cast(BaseCacheContext, cache), "model", 1, event_backend=backend
    )
    monkeypatch.setattr(transfer_mod, "DeviceHostFuncDispatcher", MagicMock())
    transfer = transfer_mod.LMCacheDrivenTransferModule(cast(MPCacheServerContext, ctx))
    monkeypatch.setattr(
        transfer,
        "get_and_touch_context_entry",
        lambda instance: entry if instance == 1 else None,
    )
    monkeypatch.setattr(mod, "get_layout_desc", lambda *args: object())
    monkeypatch.setattr(
        mod,
        "torch_dev",
        SimpleNamespace(
            device=lambda device: nullcontext(), stream=lambda stream: nullcontext()
        ),
    )
    transfers = []

    def copy(
        cache: object, ids: list[torch.Tensor], objects: list[object], **kwargs: object
    ) -> None:
        transfers.append(
            (
                cast(int, kwargs["object_group_id"]),
                [i.tolist() for i in ids],
                cast(list[list[int]], kwargs["block_ids_host"]),
            )
        )

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", copy)
    callbacks = []
    monkeypatch.setattr(
        mod,
        "submit_callback_to_stream",
        lambda stream, kind, keys: callbacks.append((kind, keys)),
    )
    module = ChunkStoreModule(cast(MPCacheServerContext, ctx), transfer)
    yield _Harness(module, backend, storage, transfers, callbacks, producer)
    module.close()


def _future(
    harness: _Harness, response: ChunkStoreResponse
) -> ChunkEventDeviceMessagingFuture:
    raw: MessagingFuture[ChunkStoreResponse] = MessagingFuture()
    raw.set_result(response)

    def release(lease_id: str) -> MessagingFuture[None]:
        harness.module.release_chunk_store_events(1, lease_id)
        result: MessagingFuture[None] = MessagingFuture()
        result.set_result(None)
        return result

    return ChunkEventDeviceMessagingFuture(raw, "cpu", harness.backend, release)


def test_delayed_import_survives_gc_and_export_ring_eviction(harness: _Harness) -> None:
    response = harness.store()
    for _ in range(2100):
        unrelated = harness.backend.create_event("cpu")
        harness.backend.record_event(unrelated, None)
        harness.backend.export_event(unrelated, "cpu")
    del unrelated
    gc.collect()
    assert harness.module.report_status()["chunk_store_event_leases"] == 1
    future = _future(harness, response)
    assert not future.query()
    for handle, start, end in response[1]:
        _Event.complete(handle)
        assert future.take_completed_ranges() == ((start, end),)
        assert future.take_completed_ranges() == ()
    assert not future.query()
    _Event.complete(response[0])
    assert future.result(0)
    assert future.release_complete()
    assert harness.module.report_status()["chunk_store_event_leases"] == 0
    gc.collect()
    for handle, _, _ in response[1]:
        assert handle not in _Event.exporters
    assert future.query()
    assert future.result(0)
    assert future.take_completed_ranges() == ()


def test_terminal_wait_preserves_undrained_ranges(harness: _Harness) -> None:
    future = _future(harness, harness.store())
    assert future.result(0)
    assert future.take_completed_ranges() == ((8, 12), (12, 16))
    assert future.take_completed_ranges() == ()
    future.wait_for_release(0)


def test_chunk_geometry_and_one_commit(harness: _Harness) -> None:
    response = harness.store()
    assert response[2]
    assert [g for g, _, _ in harness.transfers] == [0, 1, 0, 1]
    assert harness.transfers[0][1:] == (
        [[1, 2, 3, 4], [-1, -1]],
        [[1, 2, 3, 4], [-1, -1]],
    )
    assert harness.transfers[2][1:] == (
        [[5, 6, 7, 8], [12, 13]],
        [[5, 6, 7, 8], [12, 13]],
    )
    assert harness.callbacks == [("finish_write", ["a0", "a1", "b1"])]
    assert [call.args[0] for call in harness.storage.reserve_write.call_args_list] == [
        ["a0", "a1"],
        ["b1"],
    ]


def test_partial_failure_aborts_all_reserved_objects(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_later(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("copy failed")

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", fail_later)
    response = harness.store()
    assert not response[2]
    assert [(start, end) for _, start, end in response[1]] == [(8, 12)]
    assert harness.callbacks == [("abort_chunk_store", ["a0", "a1", "b1"])]
    assert not _future(harness, response).result(0)
    assert harness.module.report_status()["chunk_store_event_leases"] == 0


def test_release_is_idempotent_and_scoped_to_owner(harness: _Harness) -> None:
    response = harness.store()
    harness.module.release_chunk_store_events(2, response[3])
    assert harness.module.report_status()["chunk_store_event_leases"] == 1
    _future(harness, response).wait_for_release(0)
    harness.module.release_chunk_store_events(1, response[3])
    assert harness.module.report_status()["chunk_store_event_leases"] == 0


def test_worker_cleanup_and_lease_budget(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "MAX_LEASES_PER_INSTANCE", 1)
    response = harness.store()
    assert harness.store() == (b"", [], False, "")
    assert harness.backend.import_event(response[0], "cpu") is not None
    harness.module.drop_instance_state(1)
    assert harness.module.report_status()["chunk_store_event_leases"] == 0


def test_unregistered_or_malformed_store_submits_no_work(harness: _Harness) -> None:
    key = cast(IPCCacheServerKey, SimpleNamespace(request_id="r"))
    assert harness.module.store_with_chunk_events(key, 99, [], b"") == (
        b"",
        [],
        False,
        "",
    )
    assert harness.module.store_with_chunk_events(key, 1, [[1]], b"") == (
        b"",
        [],
        False,
        "",
    )
    assert harness.transfers == []
    assert harness.callbacks == []


def test_future_before_response_failure_and_concurrent_drain(harness: _Harness) -> None:
    raw: MessagingFuture[ChunkStoreResponse] = MessagingFuture()
    future = ChunkEventDeviceMessagingFuture(
        raw, "cpu", harness.backend, lambda lease: MessagingFuture()
    )
    assert not future.query()
    assert not future.wait(0)
    assert future.take_completed_ranges() == ()
    raw.set_exception(RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        future.result(0)
    assert future.release_complete()
    future.wait_for_release(0)
    with pytest.raises(RuntimeError, match="transport failed"):
        future.result(0)
    response = harness.store()
    done = _future(harness, response)
    done.wait()
    ranges: list[tuple[int, int]] = []
    threads = [
        threading.Thread(target=lambda: ranges.extend(done.take_completed_ranges()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert ranges == [(8, 12), (12, 16)]


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
def test_chunk_store_and_release_over_both_transports(
    harness: _Harness, transport: RequestTransport, unused_tcp_port: int
) -> None:
    url = request_server_url(transport, unused_tcp_port)
    server = create_request_server(
        [harness.module], request_server_config(transport, url)
    )
    server.start()
    client = RequestClientFactory.create(url)
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=1,
        num_kv_readers=1,
        token_ids=tuple(range(16)),
        start=8,
        end=16,
        request_id="r",
    )
    try:
        raw = client.store_with_chunk_events(
            key,
            1,
            [[1, 2, 3, 4, 5, 6, 7, 8], [-1, -1, -1, -1, 10, 11, 12, 13]],
            harness.producer.ipc_handle(),
        )
        future = ChunkEventDeviceMessagingFuture(
            raw,
            "cpu",
            harness.backend,
            lambda lease: client.release_chunk_store_events(1, lease),
        )
        assert future.result(10)
        assert future.take_completed_ranges() == ((8, 12), (12, 16))
        future.wait_for_release(10)
        assert harness.module.report_status()["chunk_store_event_leases"] == 0
    finally:
        client.close()
        server.close()

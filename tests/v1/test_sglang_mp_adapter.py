# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPConnector.store_kv_async``. The MQ and
GPU boundaries are stubbed; no live daemon or CUDA device needed. Covers the
async store contract added for SGLang MP mode: ``store_kv_async`` always
returns a pollable future -- an already-completed one on the no-op paths
(unhealthy connector / no chunk-aligned range), and the daemon's real
completion future on the happy path -- and never blocks on the result."""

# Standard
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture

pytestmark = pytest.mark.sglang


def _import_adapter_symbols() -> tuple[
    Any,
    type,
    Callable[[bool], MessagingFuture],
    type,
    type,
    type,
]:
    # First Party
    from lmcache.integration.sglang import multi_process_adapter as adapter_mod
    from lmcache.integration.sglang.multi_process_adapter import (
        LMCacheMPConnector,
        _completed_future,
    )
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )

    return (
        adapter_mod,
        LMCacheMPConnector,
        _completed_future,
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )


_CHUNK_SIZE = 256


def _make_connector(healthy: bool = True) -> Any:
    """Build a connector without running ``__init__`` (which opens ZMQ).

    Sets only the attributes the store paths touch; anything a given
    test needs beyond this it stubs itself.
    """
    _, LMCacheMPConnector, _, _, _, _ = _import_adapter_symbols()
    conn: Any = object.__new__(LMCacheMPConnector)
    conn._health_event = threading.Event()
    if healthy:
        conn._health_event.set()
    conn._lmcache_chunk_size = _CHUNK_SIZE
    conn._mq_timeout = 5.0
    return conn


def _store_metadata(num_tokens: int) -> Any:
    *_, StoreMetadata = _import_adapter_symbols()
    return StoreMetadata(
        last_node=None,
        token_ids=list(range(num_tokens)),
        kv_indices=torch.empty(0, dtype=torch.int64),
        offset=0,
        request_id="req-test",
    )


class _SpyFuture(MessagingFuture):
    """Future that records whether the caller blocked on ``result``."""

    def __init__(self) -> None:
        super().__init__()
        self.result_called = False
        self.retained_references: list[object] = []

    def result(self, timeout=None):
        self.result_called = True
        return super().result(timeout)

    def retain_reference(self, value: object) -> None:
        self.retained_references.append(value)
        super().retain_reference(value)


class _FakeRaw:
    """Stand-in returning a preset platform-aware completion future."""

    def __init__(self, future: MessagingFuture) -> None:
        self._future = future

    def to_device_future(self, device=None) -> MessagingFuture:
        return self._future


class _FakeEvent:
    def __init__(self, interprocess: bool = False) -> None:
        pass

    def record(self, stream) -> None:
        pass

    def ipc_handle(self) -> bytes:
        return b"fake-ipc-handle"


class _FakeTorchDev:
    Event = _FakeEvent

    @staticmethod
    def current_stream():
        return object()


def test_completed_future_resolves_to_given_result() -> None:
    _, _, _completed_future, _, _, _ = _import_adapter_symbols()
    done_true = _completed_future(True)
    assert done_true.query() is True
    assert done_true.result(timeout=0) is True

    done_false = _completed_future(False)
    assert done_false.query() is True
    assert done_false.result(timeout=0) is False


def test_wrap_sglang_kv_caches_uses_platform_wrapper_in_kv_order(
    monkeypatch,
) -> None:
    """Registration dispatches each tensor through the platform wrapper."""
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    k_pool = [torch.tensor([1]), torch.tensor([2])]
    v_pool = [torch.tensor([3]), torch.tensor([4])]
    wrapped_tensors: list[torch.Tensor] = []

    def wrap_one(tensor: torch.Tensor) -> object:
        wrapped_tensors.append(tensor)
        return SimpleNamespace(tensor=tensor)

    monkeypatch.setattr(
        adapter_mod,
        "wrap_one_kv_cache",
        wrap_one,
        raising=False,
    )

    wrapped = adapter_mod._wrap_sglang_kv_caches(k_pool, v_pool)
    wrapped_namespaces = cast(list[SimpleNamespace], wrapped)

    assert wrapped_tensors == [*k_pool, *v_pool]
    assert [wrapper.tensor for wrapper in wrapped_namespaces] == [*k_pool, *v_pool]


def test_wrap_sglang_kv_caches_requires_full_handle_capability(
    monkeypatch,
) -> None:
    """Registration fails before export when a platform capability is absent."""
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    monkeypatch.setattr(
        adapter_mod,
        "get_device_spec",
        lambda _device_type: SimpleNamespace(
            is_handle_transfer_available=lambda: False
        ),
    )

    with pytest.raises(ValueError, match="required memory IPC, event IPC"):
        adapter_mod._wrap_sglang_kv_caches(
            [torch.tensor([1])],
            [torch.tensor([2])],
        )


def test_wrap_sglang_kv_caches_rejects_mismatched_layers() -> None:
    """Registration rejects a wire payload that cannot split into K/V pairs."""
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    with pytest.raises(ValueError, match="matching K and V layers"):
        adapter_mod._wrap_sglang_kv_caches(
            [torch.tensor([1]), torch.tensor([2])],
            [torch.tensor([3])],
        )


@pytest.mark.parametrize(
    ("k_pool", "v_pool", "message"),
    [
        ([], [torch.tensor([1])], "non-empty K and V pools"),
        ([torch.tensor([1])], [], "non-empty K and V pools"),
        (
            [torch.tensor([1]), torch.tensor([2])],
            [torch.tensor([3])],
            "matching K and V layers",
        ),
    ],
)
def test_mp_connector_validates_pools_before_opening_mq(
    k_pool: list[torch.Tensor],
    v_pool: list[torch.Tensor],
    message: str,
) -> None:
    """Invalid registration fails before reading the first tensor or opening MQ."""
    _, LMCacheMPConnector, _, _, _, _ = _import_adapter_symbols()
    with pytest.raises(ValueError, match=message):
        LMCacheMPConnector(
            sgl_config=SimpleNamespace(model_path="test-model"),
            tp_size=1,
            rank=0,
            page_size=2,
            host="127.0.0.1",
            port=5556,
            k_pool=k_pool,
            v_pool=v_pool,
        )


def test_store_kv_async_unhealthy_returns_failed_future_no_send(monkeypatch) -> None:
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    conn = _make_connector(healthy=False)

    def _fail_send(*args, **kwargs):
        pytest.fail("send_lmcache_request must not be called when unhealthy")

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", _fail_send)

    future = conn.store_kv_async(_store_metadata(num_tokens=4 * _CHUNK_SIZE))

    assert isinstance(future, MessagingFuture)
    assert future.query() is True
    # Unhealthy connector stored nothing -> the future must report failure.
    assert future.result(timeout=0) is False


def test_store_kv_async_no_aligned_range_returns_completed_future_no_send(
    monkeypatch,
) -> None:
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    conn = _make_connector(healthy=True)

    def _fail_send(*args, **kwargs):
        pytest.fail("send_lmcache_request must not be called with no aligned range")

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", _fail_send)

    # Fewer tokens than one chunk -> aligned_end == 0 -> no wire send.
    future = conn.store_kv_async(_store_metadata(num_tokens=_CHUNK_SIZE - 1))

    assert isinstance(future, MessagingFuture)
    assert future.result(timeout=0) is True


def test_store_kv_async_happy_path_returns_daemon_future_without_blocking(
    monkeypatch,
) -> None:
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    conn = _make_connector(healthy=True)
    conn.mq_client = object()  # type: ignore[assignment]
    conn.instance_id = 123
    conn.device = "cpu"
    # Stub the helpers store_kv_async calls so we exercise only its own logic.
    conn._slot_mapping_to_block_ids = lambda kv_indices: [0, 1]  # type: ignore[method-assign,assignment]
    conn._create_key = lambda *args, **kwargs: "fake-key"  # type: ignore[method-assign,assignment]

    sentinel = _SpyFuture()
    monkeypatch.setattr(adapter_mod, "torch_dev", _FakeTorchDev)
    monkeypatch.setattr(
        adapter_mod,
        "send_lmcache_request",
        lambda mq_client, request_type, payload: _FakeRaw(sentinel),
    )

    future = conn.store_kv_async(_store_metadata(num_tokens=4 * _CHUNK_SIZE))

    # It returns the daemon's own future, and must NOT have blocked on it.
    assert future is sentinel
    assert sentinel.result_called is False
    # The exporting device event must be pinned to the future so it isn't
    # garbage-collected before the daemon waits on its IPC handle.
    assert len(sentinel.retained_references) == 1
    assert isinstance(sentinel.retained_references[0], _FakeEvent)


def test_submit_retrieve_retains_exported_device_event(monkeypatch) -> None:
    """The producer event outlives daemon import through the returned future."""
    adapter_mod, _, _, _, _, _ = _import_adapter_symbols()
    connector = _make_connector(healthy=True)
    connector.mq_client = object()  # type: ignore[assignment]
    connector.instance_id = 123
    connector.device = "cpu"
    connector.model_name = "test-model"
    connector.tp_size = 1
    connector.worker_id = 0
    sentinel = _SpyFuture()
    raw = _FakeRaw(sentinel)
    monkeypatch.setattr(adapter_mod, "torch_dev", _FakeTorchDev)
    monkeypatch.setattr(
        adapter_mod,
        "send_lmcache_request",
        lambda mq_client, request_type, payload: raw,
    )

    raw_future, future = connector._submit_retrieve(
        request_id="request-1",
        token_ids=[1, 2],
        offset=0,
        matched_end=2,
        block_ids=[0],
    )

    assert raw_future is raw
    assert future is sentinel
    assert len(sentinel.retained_references) == 1
    assert isinstance(sentinel.retained_references[0], _FakeEvent)


@pytest.mark.parametrize(
    ("event_handle", "expected_free_ranges"),
    [
        (b"", [(0, _CHUNK_SIZE)]),
        (
            b"completion-event",
            [(0, _CHUNK_SIZE), (_CHUNK_SIZE, 2 * _CHUNK_SIZE)],
        ),
    ],
)
def test_retrieve_failure_uses_single_cleanup_owner(
    event_handle: bytes,
    expected_free_ranges: list[tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event-free failure must not repeat server-side lock cleanup."""
    adapter_mod, _, _completed_future, _, LoadMetadata, _ = _import_adapter_symbols()
    connector = _make_connector(healthy=True)
    connector.mq_client = object()  # type: ignore[assignment]
    connector.model_name = "test-model"
    connector.tp_size = 2
    connector.worker_id = 0
    connector.page_size = _CHUNK_SIZE
    connector._pending_lookups = {
        "request-1": SimpleNamespace(
            token_ids=list(range(2 * _CHUNK_SIZE)),
            matched_token_num=2 * _CHUNK_SIZE,
            locks_held=True,
        )
    }
    connector._pending_lookups_lock = threading.Lock()
    connector._slot_mapping_to_block_ids = MagicMock(return_value=[1])

    free_ranges: list[tuple[int, int]] = []

    def record_free_request(
        _mq_client: object,
        request_type: object,
        payload: list[Any],
    ) -> MessagingFuture[bool]:
        assert request_type is adapter_mod.RequestType.FREE_LOOKUP_LOCKS
        free_ranges.append((payload[0].start, payload[0].end))
        return _completed_future(True)

    monkeypatch.setattr(adapter_mod, "send_lmcache_request", record_free_request)

    raw_future: MessagingFuture[tuple[bytes, bool]] = MessagingFuture()
    raw_future.set_result((event_handle, False))
    connector._submit_retrieve = MagicMock(
        return_value=(raw_future, _completed_future(False))
    )
    token_ids = connector._pending_lookups["request-1"].token_ids
    metadata = LoadMetadata(
        token_ids=token_ids,
        slot_mapping=torch.arange(2 * _CHUNK_SIZE),
        offset=_CHUNK_SIZE,
        request_id="request-1",
    )

    with pytest.raises(RuntimeError, match="LMCache MP retrieve failed"):
        connector.retrieve_kv(metadata)

    assert free_ranges == expected_free_ranges
    assert connector._pending_lookups["request-1"].locks_held is False


def test_layerwise_load_forwards_partial_slot_mapping_offset() -> None:
    """Layerwise retrieval tells the GPU connector where its slot map begins."""
    _, _, _, LMCacheLayerwiseConnector, LoadMetadata, _ = _import_adapter_symbols()
    connector: Any = object.__new__(LMCacheLayerwiseConnector)
    connector.lmcache_engine = MagicMock()
    connector.lmcache_engine.lookup.return_value = 6
    connector.lmcache_engine.retrieve_layer.return_value = iter([None, None, None])
    connector.sgl_config = SimpleNamespace(num_hidden_layers=2)
    connector.tp_size = 1
    connector.rank = 0
    connector.tp_group = None
    connector.kvcaches = [[], []]
    connector.layerwise_retrievers = []
    connector.layer_load_layer = []
    connector.lookup_id_list = []
    metadata = LoadMetadata(
        token_ids=list(range(8)),
        slot_mapping=torch.tensor([10, 11, 12, 13, 14]),
        offset=3,
    )

    assert connector.start_load_kv(metadata) == 3

    retrieve_kwargs = connector.lmcache_engine.retrieve_layer.call_args.kwargs
    assert retrieve_kwargs["offset"] == 3
    assert torch.equal(
        retrieve_kwargs["slot_mapping"].cpu(),
        metadata.slot_mapping.cpu(),
    )

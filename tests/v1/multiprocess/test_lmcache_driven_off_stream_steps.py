# SPDX-License-Identifier: Apache-2.0
"""RETRIEVE completes (event, read-lock release, end event) once all its
device work is queued, whether its steps ran on or off the stream."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.object_group_transfer.transfer_step import (
    TransferStep,
    hand_off_step,
)
from tests.v1.multiprocess.test_event_ipc_handle_path import (
    _FakeEventBackend,
    _FakeStorageManager,
    _NoopDispatcher,
)

_CHUNK_SIZE = 256
_COMPLETION = ["record", "finish_read_by_owner", EventType.MP_RETRIEVE_END]


class _Storage(_FakeStorageManager):
    def unsafe_read(self, keys, l1_owners=None):
        return list(keys), [SimpleNamespace(get_size=lambda: 64) for _ in keys]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    """A transfer module whose transfer hands ``handed_off`` to ``engine``;
    ``order`` logs completion calls, ``ended`` the end events' metadata."""
    order: list[Any] = []
    ended: list[dict] = []
    handed_off: list[TransferStep] = []
    state = SimpleNamespace(engine=None)
    monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", _NoopDispatcher)
    monkeypatch.setattr(transfer, "downsample_and_stage_block_ids", lambda c, b: b)
    monkeypatch.setattr(transfer.torch_dev, "device", lambda d: nullcontext())
    monkeypatch.setattr(transfer.torch_dev, "stream", lambda s: nullcontext())
    monkeypatch.setattr(
        transfer, "submit_callback_to_stream", lambda s, kind, p: order.append(kind)
    )

    def publish_on_stream(stream: Any, event: Any) -> None:
        order.append(event.event_type)
        ended.append(event.metadata)

    ctx = SimpleNamespace(
        chunk_size=1,
        default_chunk_size=1,
        storage_manager=_Storage(),
        get_read_owners=lambda request_id: None,
        event_bus=SimpleNamespace(
            publish=lambda event: None, publish_on_stream=publish_on_stream
        ),
        resolve_obj_keys=lambda key, group_ids: [["k0", "k1"]],
    )
    module: Any = transfer.LMCacheDrivenTransferModule(cast(Any, ctx))
    backend = _FakeEventBackend()
    record = backend.record_event

    def record_event(event: object, stream: object) -> None:
        order.append("record")
        record(event, stream)

    def transfer_object_group(*args: Any, **kwargs: Any) -> None:
        for step in handed_off:
            hand_off_step(state.engine, step)

    backend.record_event = record_event  # type: ignore[method-assign]
    cache_context = SimpleNamespace(
        device=torch.device("cpu"),
        stream="stream",
        cupy_stream="cupy",
        hold_imported_event=lambda event: 0,
        max_batch_size=4,
        kv_layer_groups_manager=SimpleNamespace(
            num_object_groups=1,
            num_kernel_groups=1,
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1], group_kinds=()
            ),
        ),
        calculate_num_blocks=lambda chunk_size, group_idx: 1,
        transfer_object_group=transfer_object_group,
    )
    entry = transfer.ContextEntry(
        cast(Any, cache_context),
        "model",
        1,
        _CHUNK_SIZE,
        event_backend=cast(Any, backend),
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda iid: entry)
    key = SimpleNamespace(
        request_id="r", cache_salt="", end=2, has_unfull_chunk=lambda size: False
    )
    yield SimpleNamespace(
        retrieve=lambda: module.retrieve(key, 1, [[0, 1]], b"producer"),
        order=order,
        ended=ended,
        handed_off=handed_off,
        state=state,
    )


class _Engine:
    """An off-stream engine that is done when a test says so."""

    def __init__(self) -> None:
        self.on_done: list = []

    def submit(self, cache_context, steps, on_done) -> None:
        self.on_done.append(on_done)


def _off_stream_step() -> TransferStep:
    return TransferStep([], [], lambda staging: [], lambda staging, ids: None)


@pytest.mark.parametrize("off_stream, succeeded", [(False, True), (True, False)])
def test_a_retrieve_the_stream_orders_is_answered_when_it_returns(
    harness, off_stream, succeeded
) -> None:
    """Also with a step no engine runs: it fails at once, releases its read
    locks and reports no tokens."""
    if off_stream:
        harness.handed_off.append(_off_stream_step())
    response = harness.retrieve()
    assert response.done()
    assert response.result() == (b"completion-handle", succeeded)
    assert harness.order[-3:] == _COMPLETION
    assert harness.order.count("finish_read_by_owner") == 1
    assert harness.ended[-1]["num_tokens"] == (2 * _CHUNK_SIZE if succeeded else 0)


@pytest.mark.parametrize("succeeded", [True, False])
def test_a_retrieve_with_off_stream_steps_is_answered_when_its_engine_is_done(
    harness, succeeded
) -> None:
    """The event, the lock release and the reply wait until the engine queued
    the kernels, or gave up."""
    engine = _Engine()
    harness.state.engine = engine
    harness.handed_off.append(_off_stream_step())
    response = harness.retrieve()
    assert not response.done()
    assert not set(_COMPLETION) & set(harness.order)
    (on_done,) = engine.on_done
    on_done(succeeded)
    assert response.result(timeout=5) == (b"completion-handle", succeeded)
    assert harness.order[-3:] == _COMPLETION
    assert harness.order.count("finish_read_by_owner") == 1
    assert harness.ended[-1]["num_tokens"] == (2 * _CHUNK_SIZE if succeeded else 0)

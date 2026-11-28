# SPDX-License-Identifier: Apache-2.0
"""RETRIEVE with off-stream steps: no engine runs them yet, so the retrieve
fails, and it is still completed (event, read-lock release, end event)."""

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
    """A transfer module with one registered context whose transfer hands
    off the steps in ``handed_off``; ``order`` logs the completion calls and
    ``ended`` the end events' metadata."""
    order: list[Any] = []
    ended: list[dict] = []
    handed_off: list[TransferStep] = []
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
    backend.record_event = lambda e, s: order.append("record") or record(e, s)  # type: ignore[method-assign]
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
        transfer_object_group=lambda *a, **k: [
            hand_off_step(step) for step in handed_off
        ],
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
    )


@pytest.mark.parametrize("off_stream", [False, True])
def test_a_retrieve_with_off_stream_steps_fails_and_is_still_completed(
    harness, off_stream
) -> None:
    """Their bytes would never land, so the engine must recompute the blocks;
    the read locks are released and the end event reports no tokens."""
    if off_stream:
        harness.handed_off.append(
            TransferStep([], [], lambda staging: [], lambda staging, ids: None)
        )
    assert harness.retrieve() == (b"completion-handle", not off_stream)
    assert harness.order[-3:] == _COMPLETION
    assert harness.order.count("finish_read_by_owner") == 1
    assert harness.ended[-1]["num_tokens"] == (0 if off_stream else 2 * _CHUNK_SIZE)

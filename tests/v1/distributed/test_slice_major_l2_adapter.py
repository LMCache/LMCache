# SPDX-License-Identifier: Apache-2.0
"""Tests for the slice-major native connector L2 adapter."""

# Standard
from typing import Any, cast
import ctypes
import select
import sys
import threading
import time
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed import pending_fill
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.l2_adapters import mooncake_store_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter import (
    MooncakeStoreL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _object_key_to_string,
)
from lmcache.v1.distributed.l2_adapters.slice_major import (
    SliceMajorL2Adapter,
    SlicedClient,
)
from lmcache.v1.distributed.pending_fill import FillFailedError, FillTracker
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd, create_event_notifier

NUM_SLICES = 4
NUM_FLOATS = 64  # 256 bytes: 64-byte slices
SLICE_BYTES = NUM_FLOATS * 4 // NUM_SLICES
TIMEOUT_S = 5.0
_MEMORY_TAGS = MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)

_Completion = tuple[int, bool, str, "list[bool] | None"]


class FakeConnector:
    """In-memory native connector with per-key get results.

    Gets complete on submit unless ``hold_gets`` is set; held gets complete
    on :meth:`finish_get`.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.hold_gets = False
        self.get_calls: list[list[str]] = []
        self.held: dict[int, tuple[list[str], list[memoryview]]] = {}
        self._efd = create_event_notifier()
        self._lock = threading.Lock()
        self._next_id = 1
        self._completions: list[_Completion] = []

    def event_fd(self) -> int:
        return self._efd.fileno()

    def _new_id(self) -> int:
        with self._lock:
            fid = self._next_id
            self._next_id += 1
        return fid

    def _push(self, completion: _Completion) -> None:
        with self._lock:
            self._completions.append(completion)
        self._efd.notify()

    def submit_batch_set(self, keys: list[str], bufs: list[memoryview]) -> int:
        fid = self._new_id()
        for key, buf in zip(keys, bufs, strict=True):
            self.store[key] = bytes(buf)
        self._push((fid, True, "", None))
        return fid

    def submit_batch_get(self, keys: list[str], bufs: list[memoryview]) -> int:
        fid = self._new_id()
        self.get_calls.append(list(keys))
        if self.hold_gets:
            self.held[fid] = (keys, bufs)
        else:
            self._push(self._get(fid, keys, bufs))
        return fid

    def finish_get(self, slice_index: int) -> None:
        """Complete the held get of ``slice_index``."""
        suffix = f"/s{slice_index}of{NUM_SLICES}"
        (fid,) = [f for f, (keys, _) in self.held.items() if keys[0].endswith(suffix)]
        keys, bufs = self.held.pop(fid)
        self._push(self._get(fid, keys, bufs))

    def _get(self, fid: int, keys: list[str], bufs: list[memoryview]) -> _Completion:
        results = []
        for key, buf in zip(keys, bufs, strict=True):
            data = self.store.get(key)
            if data is None or len(data) != buf.nbytes:
                results.append(False)
                continue
            dst = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            ctypes.memmove(dst, data, len(data))
            results.append(True)
        return (fid, True, "", results)

    def submit_batch_exists(self, keys: list[str]) -> int:
        fid = self._new_id()
        self._push((fid, True, "", [key in self.store for key in keys]))
        return fid

    def submit_batch_delete(self, keys: list[str]) -> int:
        fid = self._new_id()
        results = [self.store.pop(key, None) is not None for key in keys]
        self._push((fid, True, "", results))
        return fid

    def drain_completions(self) -> list[_Completion]:
        try:
            consume_fd(self._efd.fileno())
        except BlockingIOError:
            pass
        with self._lock:
            completions, self._completions = self._completions, []
        return completions

    def close(self) -> None:
        self._efd.close()


def make_key(index: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(index), model_name="m", kv_rank=0
    )


def make_obj(fill: float | None = None) -> MemoryObj:
    data = torch.zeros(NUM_FLOATS, dtype=torch.float32)
    if fill is None:
        data.copy_(torch.arange(NUM_FLOATS, dtype=torch.float32))
    else:
        data.fill_(fill)
    metadata = MemoryObjMetadata(
        shape=torch.Size([NUM_FLOATS]),
        dtype=torch.float32,
        address=0,
        phy_size=NUM_FLOATS * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(data, metadata, parent_allocator=None)


def wait_fd(fd: int) -> None:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(TIMEOUT_S * 1000), "event fd never fired"
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass


def wait_until(predicate: Any) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not predicate():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.01)


def store(adapter: NativeConnectorL2Adapter, keys: list[ObjectKey]) -> None:
    objs = [make_obj(float(i + 1)) for i in range(len(keys))]
    objs[0] = make_obj()
    task_id = adapter.submit_store_task(keys, objs)
    wait_fd(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[task_id].is_successful()


def load(adapter: NativeConnectorL2Adapter, task_id: L2TaskId) -> Bitmap:
    deadline = time.monotonic() + TIMEOUT_S
    while (bitmap := adapter.query_load_result(task_id)) is None:
        assert time.monotonic() < deadline, "load never completed"
        time.sleep(0.01)
    return bitmap


@pytest.fixture
def connector() -> FakeConnector:
    return FakeConnector()


def make_adapter(
    connector: FakeConnector, pipelined: bool, max_inflight: int = 2
) -> SliceMajorL2Adapter:
    return SliceMajorL2Adapter(
        SlicedClient(connector, NUM_SLICES, max_inflight),
        _MEMORY_TAGS,
        pipelined_load=pipelined,
    )


def test_objects_are_stored_as_slices(connector: FakeConnector) -> None:
    adapter = make_adapter(connector, pipelined=False)
    key = make_key(0)
    store(adapter, [key])

    name = _object_key_to_string(key)
    assert sorted(connector.store) == [
        f"{name}/s{g}of{NUM_SLICES}" for g in range(NUM_SLICES)
    ]
    expected = make_obj()
    whole = bytes(expected.byte_array)
    for g in range(NUM_SLICES):
        part = connector.store[f"{name}/s{g}of{NUM_SLICES}"]
        assert part == whole[g * SLICE_BYTES : (g + 1) * SLICE_BYTES]
    assert adapter.get_usage().total_bytes_used == NUM_FLOATS * 4
    adapter.close()


def test_lookup_needs_every_slice(connector: FakeConnector) -> None:
    adapter = make_adapter(connector, pipelined=False)
    keys = [make_key(0), make_key(1)]
    store(adapter, keys)
    del connector.store[f"{_object_key_to_string(keys[1])}/s2of{NUM_SLICES}"]

    task_id = adapter.submit_lookup_and_lock_task(keys, {})
    wait_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None and bitmap.get_indices_list() == [0]
    adapter.close()


def test_delete_removes_every_slice(connector: FakeConnector) -> None:
    adapter = make_adapter(connector, pipelined=False)
    keys = [make_key(0), make_key(1)]
    store(adapter, keys)

    adapter.delete(keys)

    assert connector.store == {}
    assert adapter.get_usage().total_bytes_used == 0
    adapter.close()


def test_non_pipelined_load_reports_once_every_slice_landed(
    connector: FakeConnector,
) -> None:
    adapter = make_adapter(connector, pipelined=False)
    keys = [make_key(0), make_key(1)]
    store(adapter, keys)
    del connector.store[f"{_object_key_to_string(keys[1])}/s3of{NUM_SLICES}"]
    objs = [make_obj(0.0), make_obj(0.0)]

    task_id = adapter.submit_load_task(keys, objs)

    assert load(adapter, task_id).get_indices_list() == [0]
    assert adapter.take_load_fill(task_id) is None
    expected = make_obj()
    assert bytes(objs[0].byte_array) == bytes(expected.byte_array)
    assert len(connector.get_calls) == NUM_SLICES
    adapter.close()


def test_pipelined_load_fills_slices_in_order(connector: FakeConnector) -> None:
    adapter = make_adapter(connector, pipelined=True)
    keys = [make_key(0), make_key(1)]
    store(adapter, keys)
    connector.hold_gets = True
    objs = [make_obj(0.0), make_obj(0.0)]

    task_id = adapter.submit_load_task(keys, objs)
    # Two slices in flight, slice 0 first.
    assert [calls[0][-5:] for calls in connector.get_calls] == ["s0of4", "s1of4"]
    assert adapter.query_load_result(task_id) is None

    connector.finish_get(0)
    assert load(adapter, task_id).get_indices_list() == [0, 1]
    fill = adapter.take_load_fill(task_id)
    assert fill is not None and pending_fill.find(objs[0]) is fill
    fill.wait(SLICE_BYTES)
    with pytest.raises(FillFailedError):
        fill.wait(2 * SLICE_BYTES, timeout=0.05)

    wait_until(lambda: len(connector.get_calls) == 3)
    connector.finish_get(2)  # out of order: the watermark holds
    with pytest.raises(FillFailedError):
        fill.wait(2 * SLICE_BYTES, timeout=0.05)
    connector.finish_get(1)
    fill.wait(3 * SLICE_BYTES)

    wait_until(lambda: len(connector.get_calls) == NUM_SLICES)
    connector.finish_get(3)
    fill.wait()
    assert pending_fill.find(objs[0]) is None
    expected = make_obj()
    assert bytes(objs[0].byte_array) == bytes(expected.byte_array)
    adapter.close()


def test_pipelined_load_fails_its_fill_when_a_later_slice_misses(
    connector: FakeConnector,
) -> None:
    adapter = make_adapter(connector, pipelined=True)
    keys = [make_key(0), make_key(1)]
    store(adapter, keys)
    del connector.store[f"{_object_key_to_string(keys[1])}/s1of{NUM_SLICES}"]
    connector.hold_gets = True
    objs = [make_obj(0.0), make_obj(0.0)]

    task_id = adapter.submit_load_task(keys, objs)
    connector.finish_get(0)
    assert load(adapter, task_id).get_indices_list() == [0, 1]
    fill = cast(FillTracker, adapter.take_load_fill(task_id))
    connector.finish_get(1)
    # Slice 2 went out when slice 0 landed; the fill waits for it.
    time.sleep(0.05)
    assert not fill.settled
    connector.finish_get(2)

    wait_until(lambda: fill.settled)
    assert fill.failed
    # No slice is fetched past the failure.
    assert len(connector.get_calls) == 3
    fill.release()
    adapter.close()


def test_pipelined_load_settles_only_after_slices_in_flight_land(
    connector: FakeConnector,
) -> None:
    adapter = make_adapter(connector, pipelined=True)
    keys = [make_key(0)]
    store(adapter, keys)
    del connector.store[f"{_object_key_to_string(keys[0])}/s0of{NUM_SLICES}"]
    connector.hold_gets = True
    objs = [make_obj(0.0)]

    task_id = adapter.submit_load_task(keys, objs)
    connector.finish_get(0)
    assert load(adapter, task_id).popcount() == 0
    fill = cast(FillTracker, adapter.take_load_fill(task_id))
    # Slice 1 still writes into the buffer.
    time.sleep(0.05)
    assert not fill.settled

    connector.finish_get(1)
    wait_until(lambda: fill.settled)
    assert not fill.failed
    assert len(connector.get_calls) == 2
    adapter.close()


def test_config_consumes_slice_keys() -> None:
    config = MooncakeStoreL2AdapterConfig.from_dict(
        {
            "type": "mooncake_store",
            "protocol": "tcp",
            "num_slices": 4,
            "max_inflight_slices": 3,
            "pipelined_load": "false",
        }
    )
    assert config.num_slices == 4
    assert config.max_inflight_slices == 3
    assert config.pipelined_load is False
    assert config.setup_config == {"protocol": "tcp"}

    with pytest.raises(ValueError):
        MooncakeStoreL2AdapterConfig.from_dict({"num_slices": 0})


@pytest.mark.parametrize(("num_slices", "sliced"), [(1, False), (4, True)])
def test_factory_slices_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, num_slices: int, sliced: bool
) -> None:
    class FakeRegistration:
        enabled = False
        base = 0
        size = 0

    class FakeClient(FakeConnector):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

    fake_module = cast(Any, types.ModuleType("lmcache.lmcache_mooncake"))
    fake_module.L1RegistrationConfig = FakeRegistration
    fake_module.LMCacheMooncakeClient = FakeClient
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_mooncake", fake_module)

    config = MooncakeStoreL2AdapterConfig.from_dict(
        {"protocol": "tcp", "num_slices": num_slices}
    )
    adapter = mooncake_store_l2_adapter._create_mooncake_store_l2_adapter(config)

    assert isinstance(adapter, SliceMajorL2Adapter) is sliced
    assert adapter.report_status()["type"] == "FakeClient"
    adapter.close()

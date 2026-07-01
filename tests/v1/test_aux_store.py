# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``AuxBlobStore`` (the opaque per-chunk aux blob store).

CPU-only: the context and storage manager are faked, and the reserved-write /
prefetched memory objects are lightweight CPU-tensor doubles. Covers the layout
math, key resolution, the store write path (incl. the already-resident no-op and
the size/count validation), and the GPU-IPC fetch path (copy correctness, the
miss path, and read-lock release).
"""

# Standard
from contextlib import contextmanager

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.aux_store import AuxBlobStore

GROUP = 7000


def _key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        token_ids=[1, 2, 3, 4],
        start=0,
        end=4,
        request_id="req-0",
    )


def _obj_keys(n: int, group: int = GROUP) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=i.to_bytes(4, "big"),
            model_name="m",
            kv_rank=0,
            object_group_id=group,
        )
        for i in range(n)
    ]


class _Reserved:
    """Duck-types the bit of ``MemoryObj`` that ``store`` touches: ``.tensor``."""

    def __init__(self, n_bf16: int, tensor_is_none: bool = False) -> None:
        self._t = None if tensor_is_none else torch.zeros(n_bf16, dtype=torch.bfloat16)

    @property
    def tensor(self):
        return self._t


class _MemObj:
    """Duck-types ``MemoryObj`` for ``lmcache_memcpy_async_h2d`` (non-GDS/-Lazy)."""

    def __init__(self, nbytes: int, fill: int) -> None:
        self._raw = torch.full((nbytes,), fill, dtype=torch.uint8)

    def get_size(self) -> int:  # asserted == dst slice nbytes
        return self._raw.numel()

    @property
    def raw_tensor(self):
        return self._raw

    def parent(self):  # must NOT be a LazyMemoryAllocator / GDS obj
        return None


class FakeStorageManager:
    """Minimal CPU StorageManager double covering only what AuxBlobStore calls."""

    def __init__(self) -> None:
        self.written: list = []
        self.read_released: list = []
        self.reserve_result: dict | None = None  # None => derive from keys
        self.prefetched: list | None = None  # None => derive from keys; [] sentinel
        self.reserve_all_none = False

    # --- store path ---
    def reserve_write(self, keys, layout: MemoryLayoutDesc, mode="new"):
        if self.reserve_result is not None:
            return self.reserve_result
        nel = layout.shapes[0].numel()  # size // 2
        return {k: _Reserved(nel, tensor_is_none=self.reserve_all_none) for k in keys}

    def finish_write(self, keys) -> None:
        self.written.extend(keys)

    # --- fetch path ---
    def submit_prefetch_task(self, keys, layout, **kw):
        return object()  # opaque handle, only passed to wait_prefetch_status

    def wait_prefetch_status(self, handle, timeout) -> bool:
        return True

    @contextmanager
    def read_prefetched_results(self, keys):
        if self.prefetched == "miss":
            yield None
        elif self.prefetched is not None:
            yield self.prefetched
        else:
            yield [_MemObj(8, fill=i + 1) for i, _ in enumerate(keys)]

    def finish_read_prefetched(self, keys, extra_count=0) -> None:
        self.read_released.extend(keys)


@pytest.fixture
def sm() -> FakeStorageManager:
    return FakeStorageManager()


@pytest.fixture
def store(sm: FakeStorageManager) -> AuxBlobStore:
    class _Ctx:
        storage_manager = sm

        def resolve_obj_keys(self, key, groups):
            return [_obj_keys(2, groups[0])]

    # _Ctx duck-types only the two attributes AuxBlobStore touches.
    return AuxBlobStore(_Ctx())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# _layout
# --------------------------------------------------------------------------- #
def test_layout_is_single_bf16_group_half_the_bytes():
    layout = AuxBlobStore._layout(16)
    assert isinstance(layout, MemoryLayoutDesc)
    assert layout.shapes == [torch.Size([8])]  # bytes // 2
    assert layout.dtypes == [torch.bfloat16]


# --------------------------------------------------------------------------- #
# _obj_keys
# --------------------------------------------------------------------------- #
def test_obj_keys_delegates_and_unwraps_group(store: AuxBlobStore):
    keys = store._obj_keys(_key(), GROUP)
    assert len(keys) == 2
    assert all(isinstance(k, ObjectKey) for k in keys)
    assert {k.object_group_id for k in keys} == {GROUP}


# --------------------------------------------------------------------------- #
# store: validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n_keys, sizes",
    [
        (0, []),  # no chunks
        (2, [8]),  # count mismatch
        (2, [8, 4]),  # unequal sizes (single-group layout needs equal chunks)
    ],
)
def test_store_rejects_bad_shapes(sm, n_keys, sizes):
    class _Ctx:
        storage_manager = sm

        def resolve_obj_keys(self, key, groups):
            return [_obj_keys(n_keys, groups[0])]

    store = AuxBlobStore(_Ctx())
    blob = torch.zeros(sum(sizes) or 1, dtype=torch.uint8)
    assert store.store(_key(), GROUP, sizes, blob) is False
    assert sm.written == []


def test_store_all_resident_is_noop_true(store, sm):
    sm.reserve_result = {}  # empty dict => everything already resident
    blob = torch.zeros(16, dtype=torch.uint8)
    assert store.store(_key(), GROUP, [8, 8], blob) is True
    assert sm.written == []


def test_store_no_writable_tensor_returns_true(store, sm):
    # Defensive branch: reserved entries whose .tensor is None -> nothing to write.
    sm.reserve_all_none = True
    blob = torch.zeros(16, dtype=torch.uint8)
    assert store.store(_key(), GROUP, [8, 8], blob) is True
    assert sm.written == []


def test_store_exception_returns_false(sm):
    class _Ctx:
        storage_manager = sm

        def resolve_obj_keys(self, key, groups):
            raise RuntimeError("boom")

    store = AuxBlobStore(_Ctx())
    assert (
        store.store(_key(), GROUP, [8, 8], torch.zeros(16, dtype=torch.uint8)) is False
    )


# --------------------------------------------------------------------------- #
# store: happy path (bytes actually land in the reserved tensors)
# --------------------------------------------------------------------------- #
def test_store_writes_each_chunk_and_finishes(store, sm):
    keys = _obj_keys(2)
    reserved = {k: _Reserved(4) for k in keys}  # 4 bf16 elems == 8 bytes each
    sm.reserve_result = reserved

    blob = torch.arange(16, dtype=torch.uint8)  # 16 bytes = 2 chunks * 8
    assert store.store(_key(), GROUP, [8, 8], blob) is True

    # finish_write got exactly the two written keys.
    assert sm.written == keys
    # Each reserved tensor holds its chunk's bytes, bit-for-bit.
    assert torch.equal(reserved[keys[0]].tensor.view(torch.uint8), blob[0:8])
    assert torch.equal(reserved[keys[1]].tensor.view(torch.uint8), blob[8:16])


# --------------------------------------------------------------------------- #
# fetch_into_ipc
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sizes", [[], [8], [8, 4]])
def test_fetch_rejects_bad_shapes(store, sm, sizes):
    keys = _obj_keys(2)  # 2 keys vs sizes of len 0/1/2-unequal
    dst = torch.zeros(sum(sizes) or 1, dtype=torch.uint8)
    assert store.fetch_into_ipc(keys, sizes, dst) is False
    assert sm.read_released == []


def test_fetch_miss_returns_false_and_no_release(store, sm):
    sm.prefetched = "miss"  # read_prefetched_results yields None
    keys = _obj_keys(2)
    dst = torch.zeros(16, dtype=torch.uint8)
    assert store.fetch_into_ipc(keys, [8, 8], dst) is False
    assert sm.read_released == []  # locks released only on the success path


def test_fetch_copies_chunks_and_releases_locks(store, sm, monkeypatch):
    # Keep it hermetic: current_stream().synchronize() must be a no-op on CPU.
    # First Party
    import lmcache

    class _Stream:
        def synchronize(self):
            pass

    monkeypatch.setattr(lmcache.torch_dev, "current_stream", lambda: _Stream())

    keys = _obj_keys(2)
    dst = torch.zeros(16, dtype=torch.uint8)
    assert store.fetch_into_ipc(keys, [8, 8], dst) is True

    # Each chunk's bytes landed at its offset (fill = i + 1 from the fake objs).
    assert torch.equal(dst[0:8], torch.full((8,), 1, dtype=torch.uint8))
    assert torch.equal(dst[8:16], torch.full((8,), 2, dtype=torch.uint8))
    # Read locks were released for exactly the fetched keys.
    assert sm.read_released == keys

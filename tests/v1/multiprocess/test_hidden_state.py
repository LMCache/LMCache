# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP hidden-state module."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import MagicMock, patch
import pickle
import sys
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

CHUNK_SIZE = 4
NUM_LAYERS = 2
HIDDEN_SIZE = 3
MODEL = "test-model"


@pytest.fixture
def stub_lmcache_native() -> Any:
    """Stub the native module so the MP imports work in a source-only run."""
    module = types.ModuleType("lmcache.lmcache_native")
    module_any: Any = module
    module_any.PageBufferShapeDesc = type("PageBufferShapeDesc", (), {})
    module_any.KernelGroupSpec = type(
        "KernelGroupSpec", (), {"__init__": lambda self, *a, **k: None}
    )
    module_any.TTLLock = type("TTLLock", (), {})
    module_any.Bitmap = type("Bitmap", (), {})
    module_any.PeriodicEventNotifier = type("PeriodicEventNotifier", (), {})
    with patch.dict(
        sys.modules, {"lmcache.lmcache_native": module, "cupy": MagicMock()}
    ):
        yield


@dataclass
class _MemoryObj:
    tensor: torch.Tensor


class _Bitmap:
    """Just enough of the native bitmap for the prefetch status path."""

    def __init__(self, hits: list[bool]) -> None:
        self._hits = hits

    def count_leading_ones(self) -> int:
        count = 0
        for hit in self._hits:
            if not hit:
                break
            count += 1
        return count


class _FakeStorageManager:
    """In-memory stand-in: a dict of committed objects, no locking, no L2."""

    def __init__(self) -> None:
        self.committed: dict[ObjectKey, torch.Tensor] = {}
        self._reserved: dict[ObjectKey, _MemoryObj] = {}
        self._handles: dict[int, list[ObjectKey]] = {}
        self._next_handle = 0

    def reserve_write(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> dict[ObjectKey, _MemoryObj]:
        reserved = {}
        for key in keys:
            if key in self.committed:
                continue
            obj = _MemoryObj(torch.zeros(layout_desc.shapes[0], dtype=torch.float32))
            self._reserved[key] = obj
            reserved[key] = obj
        return reserved

    def finish_write(self, keys: list[ObjectKey]) -> None:
        for key in keys:
            self.committed[key] = self._reserved.pop(key).tensor

    def submit_prefetch_task(self, spec: Any, external_request_id: str = "") -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = list(spec.key_groups[0].keys)
        return handle

    def wait_prefetch_status(self, handle: int, timeout: float) -> bool:
        return True

    def query_prefetch_status(self, handle: int) -> list[_Bitmap]:
        return [_Bitmap([key in self.committed for key in self._handles[handle]])]

    @contextmanager
    def read_prefetched_results(
        self, keys: list[ObjectKey]
    ) -> Iterator[list[_MemoryObj] | None]:
        if any(key not in self.committed for key in keys):
            yield None
            return
        yield [_MemoryObj(self.committed[key]) for key in keys]

    def finish_read_prefetched(self, keys: list[ObjectKey]) -> None:
        return


class _FakeSession:
    """Hashes a token range the way the real session does: one per chunk."""

    def __init__(self) -> None:
        self._tokens: list[int] = []

    def set_tokens(self, token_ids: list[int]) -> None:
        self._tokens = token_ids

    def get_hashes(self, start: int, end: int) -> list[int]:
        return [
            hash(tuple(self._tokens[:offset]))
            for offset in range(start + CHUNK_SIZE, end + 1, CHUNK_SIZE)
        ]


class _FakeContext:
    """The slice of MPCacheServerContext the hidden-state module touches."""

    def __init__(self) -> None:
        # First Party
        from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

        self.chunk_size = CHUNK_SIZE
        self.storage_manager = _FakeStorageManager()
        self.layout_desc_registry = LayoutDescRegistry()
        self.token_hasher = types.SimpleNamespace(
            hash_to_bytes=lambda value: str(value).encode()
        )
        self._sessions: dict[str, _FakeSession] = {}
        self.session_manager = types.SimpleNamespace(
            get_or_create=self._get_or_create_session
        )

    def _get_or_create_session(self, request_id: str) -> _FakeSession:
        return self._sessions.setdefault(request_id, _FakeSession())


def _make_module(ctx: _FakeContext) -> Any:
    # First Party
    from lmcache.v1.multiprocess.modules.hidden_state import HiddenStateModule

    module = HiddenStateModule(ctx)  # type: ignore[arg-type]
    assert module.register_hidden_state(
        instance_id=0,
        model_name=MODEL,
        world_size=1,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        dtype_name="float32",
    )
    return module


def _make_key(num_chunks: int, request_id: str = "req") -> IPCCacheServerKey:
    num_tokens = num_chunks * CHUNK_SIZE
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=1,
        worker_id=0,
        token_ids=tuple(range(num_tokens)),
        start=0,
        end=num_tokens,
        request_id=request_id,
    )


def _chunks(num_chunks: int) -> list[torch.Tensor]:
    return [
        torch.full((CHUNK_SIZE, NUM_LAYERS, HIDDEN_SIZE), float(i))
        for i in range(num_chunks)
    ]


def test_store_then_retrieve_round_trips(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)
    key = _make_key(3)

    assert module.store_hidden_state(key, 0, pickle.dumps(_chunks(3)))

    data, num_tokens = module.retrieve_hidden_state(key, 0)
    assert num_tokens == 3 * CHUNK_SIZE
    for restored, original in zip(pickle.loads(data), _chunks(3)):
        assert torch.equal(restored, original)


def test_retrieve_returns_the_cached_prefix_only(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)

    assert module.store_hidden_state(_make_key(2), 0, pickle.dumps(_chunks(2)))

    data, num_tokens = module.retrieve_hidden_state(_make_key(4, "other"), 0)
    assert num_tokens == 2 * CHUNK_SIZE
    assert len(pickle.loads(data)) == 2


def test_lookup_reports_covered_tokens(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)

    assert module.lookup_hidden_state(_make_key(3), 0) == 0

    assert module.store_hidden_state(_make_key(2), 0, pickle.dumps(_chunks(2)))
    assert module.lookup_hidden_state(_make_key(3, "other"), 0) == 2 * CHUNK_SIZE


def test_hidden_objects_never_collide_with_kv_objects(
    stub_lmcache_native: Any,
) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)
    key = _make_key(2)

    assert module.store_hidden_state(key, 0, pickle.dumps(_chunks(2)))

    for obj_key in ctx.storage_manager.committed:
        assert obj_key.model_name == f"{MODEL}##hidden"
        # Replicated across ranks, so a single rank-0 copy is stored.
        assert obj_key.kv_rank == 0


def test_a_different_prefix_is_a_different_object(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)

    assert module.store_hidden_state(_make_key(1), 0, pickle.dumps(_chunks(1)))
    diverged = replace(
        _make_key(1, "other"), token_ids=tuple(range(100, 100 + CHUNK_SIZE))
    )

    assert module.lookup_hidden_state(diverged, 0) == 0


def test_store_rejects_a_chunk_of_the_wrong_shape(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)
    bad = [torch.zeros(CHUNK_SIZE, NUM_LAYERS, HIDDEN_SIZE + 1)]

    assert not module.store_hidden_state(_make_key(1), 0, pickle.dumps(bad))
    assert not ctx.storage_manager.committed


def test_requests_from_an_unregistered_instance_are_refused(
    stub_lmcache_native: Any,
) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)
    key = _make_key(1)

    assert not module.store_hidden_state(key, 99, pickle.dumps(_chunks(1)))
    assert module.retrieve_hidden_state(key, 99) == (b"", 0)
    assert module.lookup_hidden_state(key, 99) == 0


def test_unregister_drops_the_layout(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    module = _make_module(ctx)
    name = f"{MODEL}##hidden"
    assert ctx.layout_desc_registry.find(name, 1) is not None

    module.unregister_hidden_state(0)

    assert ctx.layout_desc_registry.find(name, 1) is None
    assert module.report_status() == {"hidden_state_registrations": 0}


def test_register_rejects_an_unknown_dtype(stub_lmcache_native: Any) -> None:
    # First Party
    from lmcache.v1.multiprocess.modules.hidden_state import HiddenStateModule

    ctx = _FakeContext()
    module = HiddenStateModule(ctx)  # type: ignore[arg-type]

    assert not module.register_hidden_state(0, MODEL, 1, NUM_LAYERS, HIDDEN_SIZE, "f32")


def test_registered_layout_is_one_object_per_chunk(stub_lmcache_native: Any) -> None:
    ctx = _FakeContext()
    _make_module(ctx)

    layout = ctx.layout_desc_registry.find(f"{MODEL}##hidden", 1)
    assert layout is not None
    assert layout.shapes == [torch.Size([CHUNK_SIZE, NUM_LAYERS, HIDDEN_SIZE])]
    assert layout.dtypes == [torch.float32]

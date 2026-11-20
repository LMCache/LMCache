# SPDX-License-Identifier: Apache-2.0

"""Tests for NPU cache-context helpers.

``_TempNpuBuffer`` is built from a real :class:`KVLayerGroupsManager` on CPU,
the same way ``test_gpu_cache_context.py`` builds ``_TempGPUBuffer``.
``NpuCacheContext`` itself requires ``device.type == "npu"``, so close / spec /
pointer tests use test doubles (same pattern as ``test_musa_cache_context.py``).
One ``@requires_npu`` case checks staging actually lands on device.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.utils import get_group_data_ptrs
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.devices.npu import NpuDeviceSpec
from lmcache.v1.platform.devices.npu.cache_context import (
    NpuCacheContext,
    _NpuHostCallbackStream,
    _TempNpuBuffer,
)
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.no_shared_allocator

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)

NL, NB, BS, W, NH, HS = 4, 8, 16, 576, 2, 8
CHUNK = 256
DT = torch.float32
F = lmcache_native.EngineKVFormat

_RANK3 = (F.NL_X_NB_BS_HS, F.NL_X_NP_X_NB_BS_ONE_HS)


def _packed_planes(widths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    # Independently-allocated unequal-width planes disagree on dim-0 byte
    # stride; vLLM-Ascend packs them in one buffer so KVLayerGroupsManager
    # can store a single block_stride_elems.
    packed = torch.zeros(NB, BS, 1, sum(widths), dtype=DT)
    planes: list[torch.Tensor] = []
    offset = 0
    for width in widths:
        planes.append(packed[..., offset : offset + width])
        offset += width
    return tuple(planes)


def _caches(kind: str) -> tuple[list[Any], Any]:
    if kind == "mla":
        return [torch.zeros(NB, BS, W, dtype=DT) for _ in range(NL)], F.NL_X_NB_BS_HS
    if kind == "mla_tuple":
        return [_packed_planes((8, 2)) for _ in range(NL)], F.NL_X_NP_X_NB_BS_ONE_HS
    if kind == "dsa_tuple":
        return [_packed_planes((8, 2, 4)) for _ in range(NL)], F.NL_X_NP_X_NB_BS_ONE_HS
    if kind == "fused":
        return [
            torch.zeros(NB, BS, NH, 2 * HS, dtype=DT) for _ in range(NL)
        ], F.NL_X_NB_BS_NH_CS
    raise ValueError(kind)


def _manager(kind: str, compress_ratio: int = 1) -> KVLayerGroupsManager:
    caches, fmt = _caches(kind)
    infos: tuple[EngineGroupInfo, ...] = ()
    if compress_ratio > 1:
        infos = (
            EngineGroupInfo(0, tuple(range(NL)), tokens_per_block=BS * compress_ratio),
        )
    return KVLayerGroupsManager(
        caches,
        engine_kv_formats=[fmt] * NL,
        engine_group_infos=infos,
        lmcache_tokens_per_chunk=CHUNK,
    )


def _expected_kernel_group_shape(
    manager: KVLayerGroupsManager, num_tokens: int, kernel_group_idx: int
) -> torch.Size:
    """MLA/DSA tuples are rank-3 ``[L, slots, W]``; fused KV is rank-4."""
    group = manager.kernel_groups[kernel_group_idx]
    num_slots = num_tokens * group.slots_per_block // group.tokens_per_block
    if group.engine_kv_format in _RANK3:
        return torch.Size((group.num_layers, num_slots, group.hidden_dim_size))
    sd = group.shape_desc
    return torch.Size((sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size))


def _make_temp_buffer(
    manager: KVLayerGroupsManager,
    max_batch_size: int = 4,
) -> _TempNpuBuffer:
    return _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=CHUNK,
        device=torch.device("cpu"),
        max_batch_size=max_batch_size,
    )


class _FakeStream:
    def __init__(self) -> None:
        self.npu_stream = 0x1234
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1


def test_host_callback_stream_ptr_reads_npu_stream() -> None:
    adapter = _NpuHostCallbackStream(_FakeStream())
    assert adapter.ptr == 0x1234


def test_host_callback_stream_launch_runs_callback_inline_after_sync() -> None:
    stream = _FakeStream()
    adapter = _NpuHostCallbackStream(stream)
    seen: list[Any] = []
    adapter.launch_host_func(seen.append, 42)
    assert seen == [42]
    assert stream.synchronized == 1


class TestTempNpuBuffer:
    @pytest.mark.parametrize("kind", ["mla", "mla_tuple", "fused"])
    @pytest.mark.parametrize("compress_ratio", [1, 2], ids=["1x", "2x"])
    def test_temp_buffer_shape_and_dtype(self, kind: str, compress_ratio: int) -> None:
        manager = _manager(kind, compress_ratio=compress_ratio)
        buf = _make_temp_buffer(manager)
        kg = manager.kernel_groups[0]
        assert kg.tokens_per_block // kg.slots_per_block == compress_ratio
        for num_tokens in (CHUNK, 2 * CHUNK):
            shape, dtype = buf.get_kernel_group_shape_dtype(num_tokens, 0)
            assert shape == _expected_kernel_group_shape(manager, num_tokens, 0)
            assert dtype == kg.dtype
        view = buf.get_temp_kernel_group_buffer(0, 0)
        chunk_shape, _ = buf.get_kernel_group_shape_dtype(CHUNK, 0)
        assert view.shape == chunk_shape
        flat = buf.get_temp_object_group_buffer(0, 0)
        assert flat.dtype == torch.uint8
        assert flat.nbytes == chunk_shape.numel() * kg.dtype.itemsize

    @pytest.mark.parametrize("kind", ["mla", "mla_tuple", "fused"])
    def test_temp_buffer_object_group_view_is_contiguous_union(self, kind: str) -> None:
        buf = _make_temp_buffer(_manager(kind), max_batch_size=2)
        kg0 = buf.get_temp_kernel_group_buffer(1, 0)
        flat = buf.get_temp_object_group_buffer(1, 0)
        assert kg0.data_ptr() == flat.data_ptr()
        assert buf.max_batch_size == 2


@pytest.mark.parametrize("kind", ["mla", "mla_tuple", "dsa_tuple", "fused"])
def test_kernel_group_kv_pointers_returns_int64_pointer_table(kind: str) -> None:
    """CUDA-shaped int64 table: one ptr/layer, or interleaved planes for tuples."""
    caches, fmt = _caches(kind)
    ptrs = get_group_data_ptrs(caches, fmt, list(range(NL)))
    table = torch.tensor(ptrs, dtype=torch.int64)

    class _TestContext(NpuCacheContext):
        def __init__(self) -> None:
            self.group_kv_pointers_ = [table]  # type: ignore[assignment]

    entries = _TestContext().get_kernel_group_kv_pointers(0)
    assert isinstance(entries, torch.Tensor)
    assert entries.dtype == torch.int64
    assert entries.dim() == 1
    assert entries.tolist() == ptrs


@pytest.mark.parametrize("with_stream", [True, False], ids=["stream", "no_stream"])
@pytest.mark.parametrize("n_wrappers", [0, 1, 2], ids=["empty", "one", "two"])
def test_close_synchronizes_before_releasing_ipc_owners(
    with_stream: bool, n_wrappers: int
) -> None:
    """Context close waits for transfers, releases owners, and is idempotent."""
    calls: list[str] = []

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    names = ["first", "second"][:n_wrappers]
    expected = (["synchronize"] if with_stream and n_wrappers else []) + names

    class _TestContext(NpuCacheContext):
        def __init__(self) -> None:
            if with_stream:
                self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = tuple(_Wrapper(n) for n in names)  # type: ignore[assignment]
            self.kv_caches_ = ["keep"]  # type: ignore[assignment]

    context = _TestContext()
    context.close()
    context.close()
    assert calls == expected
    assert context.kv_caches_ == ([] if n_wrappers else ["keep"])


def test_device_spec_creates_npu_cache_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spec hook constructs NpuCacheContext with forwarded arguments."""

    class _Sentinel:
        pass

    created: dict[str, object] = {}

    def _fake_init(
        self: object,
        kv_caches: object,
        lmcache_tokens_per_chunk: int = 256,
        **kwargs: object,
    ) -> None:
        created["kv_caches"] = kv_caches
        created["chunk"] = lmcache_tokens_per_chunk

    monkeypatch.setattr(NpuCacheContext, "__init__", _fake_init)
    context = NpuDeviceSpec().create_cache_context([_Sentinel()], 128)
    assert isinstance(context, NpuCacheContext)
    assert isinstance(created["kv_caches"], list)
    assert created["chunk"] == 128


@requires_npu
@pytest.mark.npu
def test_temp_buffer_allocates_on_device() -> None:
    device = torch.device("npu:0")
    tensors = [torch.zeros(NB, BS, W, dtype=DT, device=device) for _ in range(NL)]
    manager = KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[F.NL_X_NB_BS_HS] * NL,
        engine_group_infos=(),
        lmcache_tokens_per_chunk=CHUNK,
    )
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=CHUNK,
        device=device,
        max_batch_size=4,
    )
    view = buffer.get_temp_kernel_group_buffer(0, 0)
    assert view.device.type == "npu"
    view.fill_(1.0)
    flat = buffer.get_temp_object_group_buffer(0, 0)
    assert flat.view(torch.float32).sum().item() == float(view.numel())

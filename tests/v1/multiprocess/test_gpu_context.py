# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``_TempGPUBuffer`` -- the temporary transfer buffer used by
``GPUCacheContext`` -- and the object-group / kernel-group layout it is built
on.

These tests exercise only the public method surface of ``_TempGPUBuffer``:

  - ``get_temp_kernel_group_buffer``  (shaped/typed per kernel group)
  - ``get_temp_object_group_buffer``  (flat uint8 per object group)
  - ``get_kernel_group_shape_dtype``
  - ``get_cache_size_per_token``
  - ``max_batch_size``

They never touch the buffer's private members. The buffer is allocated on CPU
so layout/offset/view behavior can be checked without a live GPU; constructing
``KVLayerGroupsManager`` (``PageBufferShapeDesc``) still needs the CUDA build,
so the module is skipped when CUDA is unavailable.
"""

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager  # noqa: E402
from lmcache.v1.multiprocess.gpu_context import _TempGPUBuffer  # noqa: E402
import lmcache.c_ops as lmc_ops  # noqa: E402

# Buffer layout/offset/view logic is device-independent; CPU keeps the test
# cheap and lets us read values back without GPU synchronization.
_CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_manager(
    groups: list[dict],
    *,
    num_blocks: int = 1,
    chunk_size: int = 256,
    layout_hints: LayoutHints | None = None,
) -> KVLayerGroupsManager:
    """Build a non-MLA ``KVLayerGroupsManager`` from synthetic tensors.

    Each entry in ``groups`` is a dict with keys ``num_layers``, ``num_heads``,
    ``head_size`` and optionally ``block_size`` (tensor BS dim, default 1) and
    ``dtype`` (default bfloat16). Tensors use the per-layer NHD layout
    ``[2, NB, BS, NH, HS]`` matched by ``NL_X_TWO_NB_BS_NH_HS``.
    """
    kv_caches: list[torch.Tensor] = []
    for g in groups:
        nl = g["num_layers"]
        nh = g["num_heads"]
        hs = g["head_size"]
        bs = g.get("block_size", 1)
        dt = g.get("dtype", torch.bfloat16)
        kv_caches.extend(
            torch.empty(2, num_blocks, bs, nh, hs, dtype=dt) for _ in range(nl)
        )
    return KVLayerGroupsManager(
        kv_caches,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        num_blocks=num_blocks,
        layout_hints=layout_hints,
        lmcache_logical_chunk_size=chunk_size,
    )


def _make_buffer(
    groups: list[dict],
    *,
    chunk_size: int = 256,
    max_batch_size: int = 4,
    layout_hints: LayoutHints | None = None,
    num_blocks: int = 1,
) -> tuple[_TempGPUBuffer, KVLayerGroupsManager]:
    manager = _build_manager(
        groups,
        num_blocks=num_blocks,
        chunk_size=chunk_size,
        layout_hints=layout_hints,
    )
    buffer = _TempGPUBuffer(
        manager,
        lmcache_logical_chunk_size=chunk_size,
        device=_CPU,
        max_batch_size=max_batch_size,
    )
    return buffer, manager


def _byte_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


_SINGLE = [{"num_layers": 4, "num_heads": 8, "head_size": 128}]
_TWO_GROUPS = [
    {"num_layers": 4, "num_heads": 8, "head_size": 128, "dtype": torch.bfloat16},
    {"num_layers": 2, "num_heads": 16, "head_size": 64, "dtype": torch.float16},
]


# ---------------------------------------------------------------------------
# get_kernel_group_shape_dtype
# ---------------------------------------------------------------------------


class TestKernelGroupShapeDtype:
    def test_shape_uncompressed(self) -> None:
        buf, _ = _make_buffer(_SINGLE, chunk_size=256)
        shape, dtype = buf.get_kernel_group_shape_dtype(256, 0)
        # (kv_size, num_layers, num_slots, hidden_dim) for non-MLA.
        assert tuple(shape) == (2, 4, 256, 8 * 128)
        assert dtype == torch.bfloat16

    def test_dtype_per_group(self) -> None:
        buf, _ = _make_buffer(_TWO_GROUPS, chunk_size=256)
        assert buf.get_kernel_group_shape_dtype(256, 0)[1] == torch.bfloat16
        assert buf.get_kernel_group_shape_dtype(256, 1)[1] == torch.float16

    def test_shape_compressed(self) -> None:
        # ie_logical_block_size=16, tensor block_size=8 => compress_ratio=2,
        # so a 256-logical-token chunk packs into 128 physical slots.
        groups = [{"num_layers": 2, "num_heads": 8, "head_size": 64, "block_size": 8}]
        buf, _ = _make_buffer(
            groups,
            chunk_size=256,
            layout_hints={"inference_engine_logical_block_size": 16},
        )
        shape, _ = buf.get_kernel_group_shape_dtype(256, 0)
        assert tuple(shape) == (2, 2, 128, 8 * 64)

    def test_token_count_not_divisible_by_compress_ratio_raises(self) -> None:
        groups = [{"num_layers": 1, "num_heads": 8, "head_size": 64, "block_size": 8}]
        buf, _ = _make_buffer(
            groups,
            chunk_size=256,
            layout_hints={"inference_engine_logical_block_size": 16},
        )
        # compress_ratio=2; 5 is not a multiple of it.
        with pytest.raises(ValueError, match="compress_ratio"):
            buf.get_kernel_group_shape_dtype(5, 0)


# ---------------------------------------------------------------------------
# get_temp_kernel_group_buffer
# ---------------------------------------------------------------------------


class TestKernelGroupBuffer:
    def test_shape_and_dtype_match_metadata(self) -> None:
        buf, _ = _make_buffer(_SINGLE, chunk_size=256)
        shape, dtype = buf.get_kernel_group_shape_dtype(256, 0)
        tensor = buf.get_temp_kernel_group_buffer(0, 0)
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_contiguous()

    def test_repeated_calls_same_ptr(self) -> None:
        buf, _ = _make_buffer(_SINGLE)
        a = buf.get_temp_kernel_group_buffer(0, 0)
        b = buf.get_temp_kernel_group_buffer(0, 0)
        assert a.data_ptr() == b.data_ptr()

    def test_distinct_batch_slots_have_distinct_ptrs(self) -> None:
        buf, _ = _make_buffer(_SINGLE, max_batch_size=4)
        ptrs = {buf.get_temp_kernel_group_buffer(i, 0).data_ptr() for i in range(4)}
        assert len(ptrs) == 4

    def test_all_kernel_buffers_non_overlapping(self) -> None:
        """Every (batch, kernel_group) region must be disjoint in memory."""
        buf, manager = _make_buffer(_TWO_GROUPS, max_batch_size=3)
        regions: list[tuple[int, int, str]] = []
        for batch_idx in range(buf.max_batch_size):
            for g in range(manager.num_kernel_groups):
                t = buf.get_temp_kernel_group_buffer(batch_idx, g)
                start = t.data_ptr()
                regions.append((start, start + _byte_size(t), f"b{batch_idx}g{g}"))
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                s_i, e_i, l_i = regions[i]
                s_j, e_j, l_j = regions[j]
                assert e_i <= s_j or e_j <= s_i, f"{l_i} overlaps {l_j}"

    def test_write_isolation_across_groups(self) -> None:
        buf, manager = _make_buffer(_TWO_GROUPS, chunk_size=64)
        tensors = [
            buf.get_temp_kernel_group_buffer(0, g)
            for g in range(manager.num_kernel_groups)
        ]
        for i, t in enumerate(tensors):
            t.view(torch.uint8).fill_(i + 1)
        for i, t in enumerate(tensors):
            raw = t.view(torch.uint8)
            assert raw.min().item() == i + 1
            assert raw.max().item() == i + 1

    def test_invalid_batch_idx_raises(self) -> None:
        buf, _ = _make_buffer(_SINGLE, max_batch_size=2)
        with pytest.raises(ValueError, match="batch_idx"):
            buf.get_temp_kernel_group_buffer(2, 0)

    def test_invalid_kernel_group_idx_raises(self) -> None:
        buf, _ = _make_buffer(_SINGLE)
        with pytest.raises(ValueError, match="kernel_group_idx"):
            buf.get_temp_kernel_group_buffer(0, 99)


# ---------------------------------------------------------------------------
# get_temp_object_group_buffer
# ---------------------------------------------------------------------------


class TestObjectGroupBuffer:
    def test_flat_uint8(self) -> None:
        buf, _ = _make_buffer(_SINGLE)
        obj = buf.get_temp_object_group_buffer(0, 0)
        assert obj.dtype == torch.uint8
        assert obj.dim() == 1

    def test_size_equals_sum_of_kernel_group_sizes(self) -> None:
        buf, manager = _make_buffer(_TWO_GROUPS, chunk_size=256)
        expected = sum(
            _byte_size(buf.get_temp_kernel_group_buffer(0, g))
            for g in range(manager.num_kernel_groups)
        )
        assert buf.get_temp_object_group_buffer(0, 0).numel() == expected

    def test_spans_same_range_as_its_kernel_groups(self) -> None:
        """The single object group covers exactly its kernel groups' bytes."""
        buf, manager = _make_buffer(_TWO_GROUPS)
        obj = buf.get_temp_object_group_buffer(0, 0)
        first = buf.get_temp_kernel_group_buffer(0, 0)
        last = buf.get_temp_kernel_group_buffer(0, manager.num_kernel_groups - 1)
        assert obj.data_ptr() == first.data_ptr()
        assert obj.data_ptr() + obj.numel() == last.data_ptr() + _byte_size(last)

    def test_reflects_kernel_group_writes(self) -> None:
        """Bytes written through kernel-group views appear in the object view,
        in kernel-group layout order."""
        buf, manager = _make_buffer(_TWO_GROUPS, chunk_size=64)
        sizes = []
        for g in range(manager.num_kernel_groups):
            t = buf.get_temp_kernel_group_buffer(0, g)
            t.view(torch.uint8).fill_(g + 1)
            sizes.append(_byte_size(t))

        obj = buf.get_temp_object_group_buffer(0, 0)
        offset = 0
        for g, size in enumerate(sizes):
            region = obj[offset : offset + size]
            assert region.min().item() == g + 1
            assert region.max().item() == g + 1
            offset += size

    def test_distinct_batch_slots_non_overlapping(self) -> None:
        buf, _ = _make_buffer(_TWO_GROUPS, max_batch_size=4)
        regions = []
        for batch_idx in range(buf.max_batch_size):
            o = buf.get_temp_object_group_buffer(batch_idx, 0)
            regions.append((o.data_ptr(), o.data_ptr() + o.numel()))
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                s_i, e_i = regions[i]
                s_j, e_j = regions[j]
                assert e_i <= s_j or e_j <= s_i

    def test_invalid_batch_idx_raises(self) -> None:
        buf, _ = _make_buffer(_SINGLE, max_batch_size=2)
        with pytest.raises(ValueError, match="batch_idx"):
            buf.get_temp_object_group_buffer(2, 0)

    def test_invalid_object_group_idx_raises(self) -> None:
        buf, _ = _make_buffer(_SINGLE)
        with pytest.raises(ValueError, match="object_group_idx"):
            buf.get_temp_object_group_buffer(0, 99)


# ---------------------------------------------------------------------------
# max_batch_size / get_cache_size_per_token
# ---------------------------------------------------------------------------


class TestMiscPublicApi:
    @pytest.mark.parametrize("max_batch_size", [1, 2, 4])
    def test_max_batch_size(self, max_batch_size: int) -> None:
        buf, _ = _make_buffer(_SINGLE, max_batch_size=max_batch_size)
        assert buf.max_batch_size == max_batch_size

    def test_cache_size_per_token_uncompressed(self) -> None:
        chunk = 256
        buf, manager = _make_buffer(_TWO_GROUPS, chunk_size=chunk)
        total = sum(
            _byte_size(buf.get_temp_kernel_group_buffer(0, g))
            for g in range(manager.num_kernel_groups)
        )
        assert buf.get_cache_size_per_token() == total // chunk

    def test_cache_size_per_token_compressed(self) -> None:
        chunk = 256
        groups = [{"num_layers": 2, "num_heads": 8, "head_size": 64, "block_size": 8}]
        buf, manager = _make_buffer(
            groups,
            chunk_size=chunk,
            layout_hints={"inference_engine_logical_block_size": 16},
        )
        total = sum(
            _byte_size(buf.get_temp_kernel_group_buffer(0, g))
            for g in range(manager.num_kernel_groups)
        )
        assert buf.get_cache_size_per_token() == total // chunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

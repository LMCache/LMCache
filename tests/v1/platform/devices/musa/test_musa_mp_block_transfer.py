# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat
from lmcache.utils import EngineType
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.musa.cache_context import MUSACacheContext
from lmcache.v1.platform.devices.musa.device_ops import MusaDeviceOps
from lmcache.v1.platform.devices.musa.ipc_wrapper import (
    is_musa_block_transfer_available,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

device_ops = cast(MusaDeviceOps, resolve_device_ops("musa"))


def _using_python_fallback() -> bool:
    """Return whether the test should exercise the torch-based fallback."""
    return not _has_musa_runtime()


def _has_musa_runtime() -> bool:
    """Return whether TorchMUSA is visible in this test process."""
    if not hasattr(torch, "musa"):
        return False
    try:
        return bool(torch.musa.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


def _shape_desc(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    kv_size: int,
    dtype: torch.dtype,
) -> PageBufferShapeDesc:
    """Build a block-transfer shape descriptor for compact test tensors."""
    desc = device_ops.PageBufferShapeDesc()
    desc.nl = num_layers
    desc.nb = num_blocks
    desc.bs = block_size
    desc.nh = num_heads
    desc.hs = head_size
    desc.kv_size = kv_size
    desc.element_size = torch.empty((), dtype=dtype).element_size()
    return desc


def _round_trip(
    source: list[torch.Tensor],
    chunk: torch.Tensor,
    block_ids: torch.Tensor,
    device: torch.device,
    shape_desc: PageBufferShapeDesc,
    chunk_tokens: int,
    engine_kv_format: EngineKVFormat,
) -> list[torch.Tensor]:
    """Gather selected blocks into ``chunk`` and scatter them to new tensors."""
    device_ops.multi_layer_block_kv_transfer(
        source,
        [chunk],
        block_ids,
        device,
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format,
        0,
    )
    target = [torch.zeros_like(layer) for layer in source]
    device_ops.multi_layer_block_kv_transfer(
        target,
        [chunk],
        block_ids,
        device,
        lmcache_native.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format,
        0,
    )
    return target


def test_musa_block_transfer_is_available() -> None:
    """The MUSA platform reports its built-in block-transfer path."""
    assert is_musa_block_transfer_available()


@pytest.mark.skipif(
    not _using_python_fallback(),
    reason="CPU fallback correctness check needs the Python fallback backend",
)
def test_musa_block_transfer_fallback_non_mla_d2h_and_h2d() -> None:
    """Fallback MUSA wrapper copies NL x [2, NB, BS, NH, HS] both ways."""
    num_layers = 2
    num_blocks = 4
    block_size = 2
    num_heads = 2
    head_size = 4
    chunk_tokens = 4
    hidden_dim = num_heads * head_size
    dtype = torch.float32
    source = [
        torch.arange(
            2 * num_blocks * block_size * num_heads * head_size,
            dtype=dtype,
        )
        .reshape(2, num_blocks, block_size, num_heads, head_size)
        .add(layer_idx * 10_000)
        for layer_idx in range(num_layers)
    ]
    chunk = torch.zeros(2, num_layers, chunk_tokens, hidden_dim, dtype=dtype)
    block_ids = torch.tensor([1, 3], dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        kv_size=2,
        dtype=dtype,
    )

    target = _round_trip(
        source,
        chunk,
        block_ids,
        torch.device("cpu"),
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    )

    for layer_idx in range(num_layers):
        assert torch.equal(target[layer_idx][:, 1], source[layer_idx][:, 1])
        assert torch.equal(target[layer_idx][:, 3], source[layer_idx][:, 3])
        assert torch.count_nonzero(target[layer_idx][:, 0]) == 0
        assert torch.count_nonzero(target[layer_idx][:, 2]) == 0


@pytest.mark.skipif(
    not _using_python_fallback(),
    reason="CPU fallback correctness check needs the Python fallback backend",
)
def test_musa_block_transfer_fallback_mla_d2h_and_h2d() -> None:
    """Fallback MUSA wrapper copies MLA NL x [NB, BS, HS] both ways."""
    num_layers = 2
    num_blocks = 4
    block_size = 2
    head_size = 8
    chunk_tokens = 4
    dtype = torch.float32
    source = [
        torch.arange(num_blocks * block_size * head_size, dtype=dtype)
        .reshape(num_blocks, block_size, head_size)
        .add(layer_idx * 10_000)
        for layer_idx in range(num_layers)
    ]
    chunk = torch.zeros(num_layers, chunk_tokens, head_size, dtype=dtype)
    block_ids = torch.tensor([0, 2], dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=1,
        head_size=head_size,
        kv_size=1,
        dtype=dtype,
    )

    target = _round_trip(
        source,
        chunk,
        block_ids,
        torch.device("cpu"),
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
    )

    for layer_idx in range(num_layers):
        assert torch.equal(target[layer_idx][0], source[layer_idx][0])
        assert torch.equal(target[layer_idx][2], source[layer_idx][2])
        assert torch.count_nonzero(target[layer_idx][1]) == 0
        assert torch.count_nonzero(target[layer_idx][3]) == 0


@pytest.mark.skipif(
    not _using_python_fallback(),
    reason="CPU fallback stress check needs the Python fallback backend",
)
def test_musa_block_transfer_fallback_repeated_round_trip() -> None:
    """Repeated store/retrieve operations preserve selected MLA blocks."""
    num_layers = 1
    num_blocks = 4
    block_size = 2
    head_size = 4
    chunk_tokens = 4
    source = [
        torch.arange(
            num_blocks * block_size * head_size,
            dtype=torch.float32,
        ).reshape(num_blocks, block_size, head_size)
    ]
    block_ids = torch.tensor([1, 3], dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=1,
        head_size=head_size,
        kv_size=1,
        dtype=torch.float32,
    )

    for _ in range(64):
        chunk = torch.zeros(
            num_layers,
            chunk_tokens,
            head_size,
            dtype=torch.float32,
        )
        target = _round_trip(
            source,
            chunk,
            block_ids,
            torch.device("cpu"),
            shape_desc,
            chunk_tokens,
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
        )
        assert torch.equal(target[0][1], source[0][1])
        assert torch.equal(target[0][3], source[0][3])


@pytest.mark.skipif(not _has_musa_runtime(), reason="MUSA hardware is required")
def test_musa_cache_context_mla_operand_round_trip() -> None:
    """The MUSA context exposes the MLA staging shape consumed by DeviceOps."""
    device = torch.device("musa:0")
    num_layers, num_blocks, block_size, hidden_size = 2, 3, 2, 4

    class _Wrapper(DeviceIPCWrapper):
        device_type = "musa"

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def to_tensor(self) -> torch.Tensor:
            return self.tensor

        def close(self) -> None:
            return None

    source = [
        torch.arange(
            num_blocks * block_size * hidden_size,
            dtype=torch.float32,
            device=device,
        ).reshape(num_blocks, block_size, hidden_size)
        + layer_idx * 1000
        for layer_idx in range(num_layers)
    ]
    context = MUSACacheContext(
        [_Wrapper(tensor) for tensor in source],
        lmcache_tokens_per_chunk=block_size,
        engine_type=EngineType.VLLM,
        layout_hints={},
        engine_group_infos=[],
    )
    try:
        staging = context.get_temp_kernel_group_buffer(0, 0)
        assert staging.shape == (num_layers, block_size, hidden_size)

        expected = [layer.clone() for layer in source]
        block_ids = torch.tensor([2], dtype=torch.int64, device=device)
        device_ops.multi_layer_block_kv_transfer(
            context.get_kernel_group_kv_pointers(0),
            [staging.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.D2H,
            context.get_shape_desc(0),
            block_size,
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )
        for layer in source:
            layer.zero_()
        device_ops.multi_layer_block_kv_transfer(
            context.get_kernel_group_kv_pointers(0),
            [staging.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.H2D,
            context.get_shape_desc(0),
            block_size,
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            0,
        )
        for expected_layer, restored_layer in zip(expected, source, strict=True):
            torch.testing.assert_close(restored_layer[2], expected_layer[2])
    finally:
        context.close()


@pytest.mark.skipif(
    not _using_python_fallback(),
    reason="CPU fallback correctness check needs the Python fallback backend",
)
def test_musa_block_transfer_fallback_sglang_mha_d2h_and_h2d() -> None:
    """Fallback MUSA wrapper copies SGLang's split K/V MHA layout both ways."""
    num_layers = 2
    num_blocks = 4
    block_size = 2
    num_heads = 2
    head_size = 4
    chunk_tokens = 4
    hidden_dim = num_heads * head_size
    dtype = torch.float32
    source = [
        [
            torch.arange(
                num_blocks * block_size * num_heads * head_size,
                dtype=dtype,
            )
            .reshape(num_blocks, block_size, num_heads, head_size)
            .add(kv_idx * 100_000 + layer_idx * 10_000)
            for layer_idx in range(num_layers)
        ]
        for kv_idx in range(2)
    ]
    chunk = torch.zeros(2, num_layers, chunk_tokens, hidden_dim, dtype=dtype)
    block_ids = torch.tensor([1, 3], dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        kv_size=2,
        dtype=dtype,
    )

    device_ops.multi_layer_block_kv_transfer(
        source,
        [chunk],
        block_ids,
        torch.device("cpu"),
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
        0,
    )
    target = [
        [torch.zeros_like(layer) for layer in layer_group] for layer_group in source
    ]
    device_ops.multi_layer_block_kv_transfer(
        target,
        [chunk],
        block_ids,
        torch.device("cpu"),
        lmcache_native.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
        0,
    )

    for kv_idx in range(2):
        for layer_idx in range(num_layers):
            assert torch.equal(
                target[kv_idx][layer_idx][1], source[kv_idx][layer_idx][1]
            )
            assert torch.equal(
                target[kv_idx][layer_idx][3], source[kv_idx][layer_idx][3]
            )
            assert torch.count_nonzero(target[kv_idx][layer_idx][0]) == 0
            assert torch.count_nonzero(target[kv_idx][layer_idx][2]) == 0


@pytest.mark.skipif(not _has_musa_runtime(), reason="MUSA hardware is required")
def test_musa_cache_context_sglang_mha_operand_round_trip() -> None:
    """The MUSA context detects SGLang MHA and preserves K/V IPC order."""
    device = torch.device("musa:0")
    num_layers, num_blocks, block_size = 2, 3, 2
    num_heads, head_size = 2, 4

    class _Wrapper(DeviceIPCWrapper):
        device_type = "musa"

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def to_tensor(self) -> torch.Tensor:
            return self.tensor

        def close(self) -> None:
            return None

    source = [
        [
            torch.arange(
                num_blocks * block_size * num_heads * head_size,
                dtype=torch.float32,
                device=device,
            )
            .reshape(num_blocks * block_size, num_heads, head_size)
            .add(kv_idx * 100_000 + layer_idx * 1_000)
            for layer_idx in range(num_layers)
        ]
        for kv_idx in range(2)
    ]
    wrappers: list[DeviceIPCWrapper] = [
        _Wrapper(tensor) for layer_group in source for tensor in layer_group
    ]
    context = MUSACacheContext(
        wrappers,
        lmcache_tokens_per_chunk=block_size,
        engine_type=EngineType.SGLANG,
        layout_hints={"tokens_per_block": block_size, "kv_list_layout": "k_v"},
        engine_group_infos=[],
    )
    try:
        assert context.get_engine_kv_format(0) == EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
        assert context.get_kernel_group_kv_pointers(0).numel() == 2 * num_layers
        staging = context.get_temp_kernel_group_buffer(0, 0)
        assert staging.shape == (
            2,
            num_layers,
            block_size,
            num_heads * head_size,
        )

        expected = [
            [tensor.clone() for tensor in layer_group] for layer_group in source
        ]
        block_ids = torch.tensor([2], dtype=torch.int64, device=device)
        device_ops.multi_layer_block_kv_transfer(
            context.get_kernel_group_kv_pointers(0),
            [staging.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.D2H,
            context.get_shape_desc(0),
            block_size,
            EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
            0,
        )
        for layer_group in source:
            for tensor in layer_group:
                tensor.zero_()
        device_ops.multi_layer_block_kv_transfer(
            context.get_kernel_group_kv_pointers(0),
            [staging.data_ptr()],
            block_ids,
            device,
            lmcache_native.TransferDirection.H2D,
            context.get_shape_desc(0),
            block_size,
            EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
            0,
        )
        for kv_idx in range(2):
            for layer_idx in range(num_layers):
                torch.testing.assert_close(
                    source[kv_idx][layer_idx].view(
                        num_blocks, block_size, num_heads, head_size
                    )[2],
                    expected[kv_idx][layer_idx].view(
                        num_blocks, block_size, num_heads, head_size
                    )[2],
                )
    finally:
        context.close()


@pytest.mark.skipif(
    not _using_python_fallback(),
    reason="CPU fallback fail-closed check needs the Python fallback backend",
)
def test_musa_block_transfer_rejects_unvalidated_layout() -> None:
    """MUSA block transfer fails closed outside the validated layouts."""
    shape_desc = _shape_desc(
        num_layers=1,
        num_blocks=1,
        block_size=1,
        num_heads=1,
        head_size=1,
        kv_size=2,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="MUSA MP block transfer supports only"):
        device_ops.multi_layer_block_kv_transfer(
            [torch.zeros(1, 2, 1, 1, 1)],
            [torch.zeros(2, 1, 1, 1)],
            torch.tensor([0], dtype=torch.int64),
            torch.device("cpu"),
            lmcache_native.TransferDirection.D2H,
            shape_desc,
            1,
            lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
            0,
        )


@pytest.mark.skipif(
    not _has_musa_runtime(),
    reason="MUSA hardware is required",
)
def test_musa_block_transfer_device_non_mla_d2h_and_h2d() -> None:
    """MUSA copies NL x [2, NB, BS, NH, HS] in both directions."""
    num_layers = 2
    num_blocks = 4
    block_size = 2
    num_heads = 2
    head_size = 4
    chunk_tokens = 4
    hidden_dim = num_heads * head_size
    device = torch.device("musa:0")
    dtype = torch.float16
    source = [
        torch.arange(
            2 * num_blocks * block_size * num_heads * head_size,
            device=device,
            dtype=dtype,
        )
        .reshape(2, num_blocks, block_size, num_heads, head_size)
        .add(layer_idx * 1000)
        for layer_idx in range(num_layers)
    ]
    chunk = torch.zeros(
        2,
        num_layers,
        chunk_tokens,
        hidden_dim,
        device=device,
        dtype=dtype,
    )
    block_ids = torch.tensor([1, 3], device=device, dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        kv_size=2,
        dtype=dtype,
    )

    target = _round_trip(
        source,
        chunk,
        block_ids,
        device,
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
    )

    for layer_idx in range(num_layers):
        assert torch.equal(target[layer_idx][:, 1], source[layer_idx][:, 1])
        assert torch.equal(target[layer_idx][:, 3], source[layer_idx][:, 3])


@pytest.mark.skipif(
    not _has_musa_runtime(),
    reason="MUSA hardware is required",
)
def test_musa_block_transfer_device_mla_d2h_and_h2d() -> None:
    """MUSA copies MLA NL x [NB, BS, HS] in both directions."""
    num_layers = 2
    num_blocks = 4
    block_size = 2
    head_size = 8
    chunk_tokens = 4
    device = torch.device("musa:0")
    dtype = torch.float16
    source = [
        torch.arange(
            num_blocks * block_size * head_size,
            device=device,
            dtype=dtype,
        )
        .reshape(num_blocks, block_size, head_size)
        .add(layer_idx * 1000)
        for layer_idx in range(num_layers)
    ]
    chunk = torch.zeros(
        num_layers,
        chunk_tokens,
        head_size,
        device=device,
        dtype=dtype,
    )
    block_ids = torch.tensor([0, 2], device=device, dtype=torch.int64)
    shape_desc = _shape_desc(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=1,
        head_size=head_size,
        kv_size=1,
        dtype=dtype,
    )

    target = _round_trip(
        source,
        chunk,
        block_ids,
        device,
        shape_desc,
        chunk_tokens,
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
    )

    for layer_idx in range(num_layers):
        assert torch.equal(target[layer_idx][0], source[layer_idx][0])
        assert torch.equal(target[layer_idx][2], source[layer_idx][2])


_DSV4_NL = 2
_DSV4_NB = 4
_DSV4_BS = 4
_DSV4_VALUE_WIDTH = 128
_DSV4_SCALE_WIDTH = 4
_DSV4_ROW_WIDTH = _DSV4_VALUE_WIDTH + _DSV4_SCALE_WIDTH
_DSV4_BLOCK_STRIDE = _DSV4_BS * _DSV4_ROW_WIDTH + 8


def _dsv4_cache() -> torch.Tensor:
    """Build one DSV4 cache view with padding between physical blocks."""
    storage = torch.zeros(
        _DSV4_NB,
        _DSV4_BLOCK_STRIDE,
        dtype=torch.uint8,
    )
    return storage.as_strided(
        (_DSV4_NB, _DSV4_BS, _DSV4_ROW_WIDTH),
        (_DSV4_BLOCK_STRIDE, _DSV4_ROW_WIDTH, 1),
    )


def _dsv4_raw_page(cache: torch.Tensor, block_id: int) -> torch.Tensor:
    """View one padded DSV4 physical block as bytes."""
    raw_pages = cache.as_strided(
        (_DSV4_NB, _DSV4_BLOCK_STRIDE),
        (cache.stride(0), 1),
    )
    return raw_pages[block_id]


def _dsv4_shape_desc() -> PageBufferShapeDesc:
    """Build the DSV4 indexer descriptor used by the MUSA fallback test."""
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = _DSV4_NL
    shape_desc.nb = _DSV4_NB
    shape_desc.bs = _DSV4_BS
    shape_desc.nh = 1
    shape_desc.hs = _DSV4_ROW_WIDTH
    shape_desc.kv_size = 1
    shape_desc.element_size = 1
    shape_desc.dtype = torch.uint8
    shape_desc.block_stride_elems = _DSV4_BLOCK_STRIDE
    return shape_desc


def _write_dsv4_blocked_rows(
    caches: list[torch.Tensor],
    block_id: int,
    rows: torch.Tensor,
) -> None:
    """Write token-major rows into a physical DSV4 blocked page."""
    for layer_idx, cache in enumerate(caches):
        page = _dsv4_raw_page(cache, block_id)
        layer_rows = rows[layer_idx].view(_DSV4_BS, _DSV4_ROW_WIDTH)
        page[: _DSV4_BS * _DSV4_VALUE_WIDTH].view(_DSV4_BS, _DSV4_VALUE_WIDTH).copy_(
            layer_rows[:, :_DSV4_VALUE_WIDTH]
        )
        scale_start = _DSV4_BS * _DSV4_VALUE_WIDTH
        scale_end = scale_start + _DSV4_BS * _DSV4_SCALE_WIDTH
        page[scale_start:scale_end].view(_DSV4_BS, _DSV4_SCALE_WIDTH).copy_(
            layer_rows[:, _DSV4_VALUE_WIDTH:]
        )


def _read_dsv4_blocked_rows(
    caches: list[torch.Tensor],
    block_id: int,
) -> torch.Tensor:
    """Read a physical DSV4 blocked page as token-major rows."""
    rows = []
    for layer_idx, cache in enumerate(caches):
        page = _dsv4_raw_page(cache, block_id)
        values = page[: _DSV4_BS * _DSV4_VALUE_WIDTH].view(_DSV4_BS, _DSV4_VALUE_WIDTH)
        scale_start = _DSV4_BS * _DSV4_VALUE_WIDTH
        scale_end = scale_start + _DSV4_BS * _DSV4_SCALE_WIDTH
        scales = page[scale_start:scale_end].view(_DSV4_BS, _DSV4_SCALE_WIDTH)
        rows.append(torch.cat((values, scales), dim=1))
    return torch.stack(rows)


def test_musa_block_transfer_fallback_dsv4_blocked_scale_round_trip() -> None:
    """MUSA fallback gathers and scatters DSV4 indexer values and scales."""
    shape_desc = _dsv4_shape_desc()
    source = [_dsv4_cache() for _ in range(_DSV4_NL)]
    target = [_dsv4_cache() for _ in range(_DSV4_NL)]
    rows = (
        torch.arange(
            _DSV4_NL * _DSV4_BS * _DSV4_ROW_WIDTH,
            dtype=torch.uint8,
        )
        .reshape(_DSV4_NL, _DSV4_BS, _DSV4_ROW_WIDTH)
        .add(7)
    )
    _write_dsv4_blocked_rows(source, 2, rows)
    staging = [
        torch.zeros(
            _DSV4_NL,
            _DSV4_BS,
            _DSV4_ROW_WIDTH,
            dtype=torch.uint8,
        )
    ]

    device_ops.multi_layer_block_kv_transfer(
        source,
        staging,
        [2],
        torch.device("cpu"),
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        _DSV4_BS,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        0,
    )
    assert torch.equal(staging[0], rows)

    device_ops.multi_layer_block_kv_transfer(
        target,
        staging,
        [1],
        torch.device("cpu"),
        lmcache_native.TransferDirection.H2D,
        shape_desc,
        _DSV4_BS,
        lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
        0,
    )
    assert torch.equal(_read_dsv4_blocked_rows(target, 1), rows)

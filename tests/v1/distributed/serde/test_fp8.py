# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Fp8QuantizationSerializer / Fp8QuantizationDeserializer.

These tests use BytesBufferMemoryObj and TensorMemoryObj directly so they
do not need an L1Manager or GPU; they verify the pure transform logic.
"""

# Standard
from dataclasses import dataclass
from typing import Callable, Optional
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.fp8 import (
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing the ``.tensor`` attribute used by fp8 serde."""

    tensor: Optional[torch.Tensor]


# =============================================================================
# estimate_serialized_size
# =============================================================================


def test_estimate_serialized_size_single_group() -> None:
    """Estimate is exactly num_elements bytes (1 byte/elem, no margin)."""
    serializer = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 256, 128])],
        dtypes=[torch.bfloat16],
    )
    numel = 2 * 4 * 256 * 128
    assert serializer.estimate_serialized_size(layout) == numel


def test_estimate_serialized_size_multi_group() -> None:
    """Multi-group layouts sum element counts across groups."""
    serializer = Fp8QuantizationSerializer()
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([4, 8]), torch.Size([16])],
        dtypes=[torch.bfloat16, torch.float16],
    )
    numel = 32 + 16
    assert serializer.estimate_serialized_size(layout) == numel


# =============================================================================
# serialize / deserialize round-trip
# =============================================================================


def test_roundtrip_bfloat16_preserves_structure() -> None:
    """Values survive fp8 round-trip with high correlation."""
    shape = torch.Size([2, 4, 64, 128])
    original = torch.randn(
        shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
    )
    src = _FakeMemoryObj(tensor=original.clone())

    # fp8 = 1 byte/elem; temp buffer is plain uint8.
    temp = _FakeMemoryObj(tensor=torch.zeros(original.numel(), dtype=torch.uint8))

    serializer = Fp8QuantizationSerializer()
    n = serializer.serialize(src, temp, _TEST_KEY)  # type: ignore[arg-type]
    assert n == original.numel()

    # Round-trip: deserialize into a fresh buffer with the original shape.
    recovered = _FakeMemoryObj(tensor=torch.zeros(shape, dtype=torch.bfloat16))
    Fp8QuantizationDeserializer().deserialize(
        temp,  # type: ignore[arg-type]
        recovered,  # type: ignore[arg-type]
        _TEST_KEY,
    )

    assert recovered.tensor is not None
    corr = torch.corrcoef(
        torch.stack([recovered.tensor.float().flatten(), original.float().flatten()])
    )[0, 1].item()
    assert corr > 0.99, f"fp8 round-trip correlation too low: {corr:.4f}"


def test_serialize_raises_on_missing_tensor() -> None:
    """A MemoryObj without ``.tensor`` is rejected rather than silently no-op'd."""
    serializer = Fp8QuantizationSerializer()
    src = _FakeMemoryObj(tensor=None)
    dst = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        serializer.serialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]


def test_deserialize_raises_on_missing_tensor() -> None:
    deserializer = Fp8QuantizationDeserializer()
    src = _FakeMemoryObj(tensor=torch.zeros(4, dtype=torch.uint8))
    dst = _FakeMemoryObj(tensor=None)
    with pytest.raises(ValueError):
        deserializer.deserialize(src, dst, _TEST_KEY)  # type: ignore[arg-type]


# =============================================================================
# Real TensorMemoryObj layouts, including multi-group buffers
# =============================================================================

# Exact in fp8 e4m3, fp8 e5m2, bfloat16, float16, and float32.
_EXACT_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _pattern(
    shape: torch.Size,
    dtype: torch.dtype,
    rotation: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    base = torch.tensor(_EXACT_VALUES, dtype=torch.float32)
    rotated = torch.roll(base, shifts=rotation)
    count = shape.numel()
    flat = rotated.repeat(count // rotated.numel() + 1)[:count]
    return flat.to(device=device, dtype=dtype).reshape(shape)


def _memory_obj(
    groups: list[torch.Tensor],
    *,
    padding: int = 0,
) -> TensorMemoryObj:
    """Allocator-style object: ``shape``/``dtype`` are group 0, plus the lists."""
    pieces = [
        group.detach().contiguous().view(torch.uint8).reshape(-1) for group in groups
    ]
    logical = torch.cat(pieces)
    raw = torch.empty(
        logical.numel() + padding, dtype=torch.uint8, device=logical.device
    )
    if logical.numel():
        raw[: logical.numel()].copy_(logical)
    if padding:
        raw[logical.numel() :].fill_(0xAB)
    shapes = [group.shape for group in groups]
    dtypes = [group.dtype for group in groups]
    meta = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=0,
        phy_size=raw.numel(),
        ref_count=1,
        shapes=list(shapes),
        dtypes=list(dtypes),
    )
    return TensorMemoryObj(raw, meta, None)


def _byte_obj(num_bytes: int, *, device: torch.device | str = "cpu") -> TensorMemoryObj:
    raw = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    meta = MemoryObjMetadata(
        shape=torch.Size([num_bytes]),
        dtype=torch.uint8,
        address=0,
        phy_size=num_bytes,
        ref_count=1,
        shapes=[torch.Size([num_bytes])],
        dtypes=[torch.uint8],
    )
    return TensorMemoryObj(raw, meta, None)


def _issue_groups(device: torch.device | str = "cpu") -> list[torch.Tensor]:
    """Three groups whose first shape is not the whole buffer (#4972).

    The last group is deliberately misaligned: fp16 ``[3, 5]`` is 30 bytes,
    so the following fp32 group does not start on a 4-byte boundary.
    ``get_tensor`` cannot view that group in place.
    """
    specs = (
        (torch.Size([1, 2, 4, 8]), torch.bfloat16),
        (torch.Size([3, 5]), torch.float16),
        (torch.Size([7]), torch.float32),
    )
    return [
        _pattern(shape, dtype, rotation=index, device=device)
        for index, (shape, dtype) in enumerate(specs)
    ]


def _aligned_groups(device: torch.device | str = "cpu") -> list[torch.Tensor]:
    """Heterogeneous groups whose byte offsets satisfy each dtype's alignment."""
    specs = (
        (torch.Size([1, 2, 4, 8]), torch.bfloat16),
        (torch.Size([4, 4]), torch.float16),
        (torch.Size([8]), torch.float32),
    )
    return [
        _pattern(shape, dtype, rotation=index, device=device)
        for index, (shape, dtype) in enumerate(specs)
    ]


def _same_dtype_groups(device: torch.device | str = "cpu") -> list[torch.Tensor]:
    """Three bf16 groups, matching the dtype layout reported in #4972."""
    shape = torch.Size([1, 2, 4, 8])
    return [
        _pattern(shape, torch.bfloat16, rotation=index, device=device)
        for index in range(3)
    ]


def _materialize(obj: TensorMemoryObj, index: int) -> torch.Tensor:
    """Read a group even when its byte offset is not aligned for ``.view``."""
    try:
        group = obj.get_tensor(index)
    except RuntimeError:
        group = None
    if group is not None:
        return group
    shapes = obj.get_shapes()
    dtypes = obj.get_dtypes()
    begin = 0
    for shape, dtype in zip(shapes[:index], dtypes[:index], strict=True):
        begin += shape.numel() * dtype.itemsize
    nbytes = shapes[index].numel() * dtypes[index].itemsize
    return (
        obj.raw_data[begin : begin + nbytes]
        .clone()
        .view(dtypes[index])
        .view(shapes[index])
    )


def _assert_roundtrip(
    groups: list[torch.Tensor],
    fp8_dtype: torch.dtype,
) -> None:
    src = _memory_obj(groups)
    if len(groups) > 1:
        with pytest.raises(RuntimeError, match="invalid"):
            _ = src.tensor
    n_elements = sum(group.numel() for group in groups)
    dst = _byte_obj(n_elements, device=groups[0].device)
    serializer = Fp8QuantizationSerializer(fp8_dtype)
    n = serializer.serialize(src, dst, _TEST_KEY)
    assert n == n_elements

    restored_groups = [
        torch.empty(group.shape, dtype=group.dtype, device=group.device)
        for group in groups
    ]
    restored = _memory_obj(restored_groups)
    Fp8QuantizationDeserializer(fp8_dtype).deserialize(dst, restored, _TEST_KEY)
    for index, group in enumerate(groups):
        got = _materialize(restored, index)
        assert got.shape == group.shape
        assert got.dtype == group.dtype
        assert torch.equal(got, group)


@pytest.mark.parametrize(
    "groups_fn",
    [_issue_groups, _aligned_groups, _same_dtype_groups],
)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_multi_group_roundtrip_preserves_each_group(
    groups_fn: Callable[[], list[torch.Tensor]],
    fp8_dtype: torch.dtype,
) -> None:
    """Heterogeneous groups stay in metadata order and keep their dtypes."""
    _assert_roundtrip(groups_fn(), fp8_dtype)


def test_single_group_tensor_matches_fake_byte_layout() -> None:
    """A real one-group TensorMemoryObj packs the same bytes as ``.tensor``."""
    shape = torch.Size([2, 4, 8])
    original = torch.randn(
        shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0)
    )
    fake_src = _FakeMemoryObj(tensor=original.clone())
    fake_dst = _FakeMemoryObj(tensor=torch.zeros(original.numel(), dtype=torch.uint8))
    real_src = _memory_obj([original.clone()])
    real_dst = _byte_obj(original.numel())

    serializer = Fp8QuantizationSerializer()
    n_fake = serializer.serialize(fake_src, fake_dst, _TEST_KEY)  # type: ignore[arg-type]
    n_real = serializer.serialize(real_src, real_dst, _TEST_KEY)
    assert n_fake == n_real == original.numel()
    real_bytes = real_dst.tensor
    assert fake_dst.tensor is not None and real_bytes is not None
    assert torch.equal(fake_dst.tensor, real_bytes)


def test_multi_group_randn_correlation() -> None:
    """Lossy values still correlate once each group is quantized on its own."""
    generator = torch.Generator().manual_seed(1)
    specs = (
        (torch.Size([64, 32]), torch.bfloat16),
        (torch.Size([32, 16]), torch.float16),
        (torch.Size([128]), torch.float32),
    )
    groups = []
    for shape, dtype in specs:
        values = torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
        groups.append(values)

    src = _memory_obj([group.clone() for group in groups])
    dst = _byte_obj(sum(group.numel() for group in groups))
    serializer = Fp8QuantizationSerializer()
    n = serializer.serialize(src, dst, _TEST_KEY)
    assert n == sum(group.numel() for group in groups)

    restored = _memory_obj(
        [torch.empty(group.shape, dtype=group.dtype) for group in groups]
    )
    Fp8QuantizationDeserializer().deserialize(dst, restored, _TEST_KEY)
    for index, original in enumerate(groups):
        got = restored.get_tensor(index)
        assert got is not None
        corr = torch.corrcoef(
            torch.stack([got.float().flatten(), original.float().flatten()])
        )[0, 1].item()
        assert corr > 0.99, f"group {index} correlation too low: {corr:.4f}"


def test_empty_middle_group_roundtrip() -> None:
    groups = [
        _pattern(torch.Size([4]), torch.bfloat16, rotation=0),
        torch.empty((0, 3), dtype=torch.float16),
        _pattern(torch.Size([2, 2]), torch.float32, rotation=2),
    ]
    _assert_roundtrip(groups, torch.float8_e4m3fn)


def test_tail_padding_is_not_serialized() -> None:
    """Alignment bytes past the logical groups stay out of the payload."""
    groups = _issue_groups()
    src = _memory_obj(groups, padding=32)
    logical_bytes = sum(group.numel() * group.dtype.itemsize for group in groups)
    payload_bytes = sum(group.numel() for group in groups)
    assert src.raw_data[logical_bytes:].eq(0xAB).all()

    dst = _byte_obj(payload_bytes)
    n = Fp8QuantizationSerializer().serialize(src, dst, _TEST_KEY)
    assert n == payload_bytes
    assert src.raw_data[logical_bytes:].eq(0xAB).all()

    plain = _byte_obj(payload_bytes)
    Fp8QuantizationSerializer().serialize(_memory_obj(groups), plain, _TEST_KEY)
    dst_bytes = dst.tensor
    plain_bytes = plain.tensor
    assert dst_bytes is not None and plain_bytes is not None
    assert torch.equal(dst_bytes, plain_bytes)


def test_short_destination_raises() -> None:
    src = _memory_obj(_issue_groups())
    with pytest.raises(ValueError, match="destination buffer too small"):
        Fp8QuantizationSerializer().serialize(src, _byte_obj(3), _TEST_KEY)


def test_short_source_raises() -> None:
    groups = _issue_groups()
    packed = _byte_obj(sum(group.numel() for group in groups))
    Fp8QuantizationSerializer().serialize(_memory_obj(groups), packed, _TEST_KEY)

    short = _byte_obj(4)
    packed_bytes = packed.tensor
    short_bytes = short.tensor
    assert packed_bytes is not None and short_bytes is not None
    short_bytes.copy_(packed_bytes[:4])
    restored = _memory_obj(
        [torch.empty(group.shape, dtype=group.dtype) for group in groups]
    )
    with pytest.raises(ValueError, match="source buffer too small"):
        Fp8QuantizationDeserializer().deserialize(short, restored, _TEST_KEY)


def test_deserialize_honors_used_size() -> None:
    """``set_used_size`` must hide bytes past the serialized payload."""
    groups = [_pattern(torch.Size([32]), torch.bfloat16, rotation=0)]
    packed = _byte_obj(64)
    n = Fp8QuantizationSerializer().serialize(_memory_obj(groups), packed, _TEST_KEY)
    assert n == 32
    packed.raw_data[n:].fill_(0xCD)
    packed.set_used_size(8)

    restored = _memory_obj(
        [torch.empty(group.shape, dtype=group.dtype) for group in groups]
    )
    with pytest.raises(ValueError, match="source buffer too small"):
        Fp8QuantizationDeserializer().deserialize(packed, restored, _TEST_KEY)


def test_async_processor_roundtrip_sets_used_size() -> None:
    """The thread-pool path narrows the byte buffer to the packed length."""
    groups = _issue_groups()
    n_elements = sum(group.numel() for group in groups)
    src = _memory_obj(groups)
    dst = _byte_obj(n_elements + 16)
    dst.raw_data[n_elements:].fill_(0xCD)

    processor = AsyncSerdeProcessor(
        Fp8QuantizationSerializer(),
        Fp8QuantizationDeserializer(),
        max_workers=1,
    )
    try:
        task_id = processor.submit_serialize([src], [dst], [_TEST_KEY])
        deadline = time.monotonic() + 10
        result: bool | None = None
        while time.monotonic() < deadline:
            result = processor.query_serialize_result(task_id)
            if result is not None:
                break
            time.sleep(0.01)
        assert result is True
        assert dst.get_size() == n_elements
        assert dst.raw_data[n_elements:].eq(0xCD).all()

        restored = _memory_obj(
            [torch.empty(group.shape, dtype=group.dtype) for group in groups]
        )
        load_id = processor.submit_deserialize([dst], [restored], [_TEST_KEY])
        deadline = time.monotonic() + 10
        loaded: bool | None = None
        while time.monotonic() < deadline:
            loaded = processor.query_deserialize_result(load_id)
            if loaded is not None:
                break
            time.sleep(0.01)
        assert loaded is True
        for index, group in enumerate(groups):
            got = _materialize(restored, index)
            assert torch.equal(got, group)
    finally:
        processor.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA runtime")
def test_multi_group_roundtrip_on_cuda() -> None:
    _assert_roundtrip(_issue_groups(device="cuda"), torch.float8_e4m3fn)
    _assert_roundtrip(_aligned_groups(device="cuda"), torch.float8_e4m3fn)
    _assert_roundtrip(_same_dtype_groups(device="cuda"), torch.float8_e5m2)

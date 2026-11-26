# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``gha`` L2 serde.

The serde is driven only through its public surface: the factory-built
``SerdeProcessor`` (submit / query / size estimate) and, end to end, the
``StorageManager`` L1 -> L2 -> L1 path. Per-group facts come from the serde's
static ``groups`` kwarg.

The native gha extension is required; CPU is always exercised and XPU is
added when an XPU device is present.
"""

# Standard
from dataclasses import dataclass
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)

pytest.importorskip("lmcache.gha_quant")

_DEVICES = ["cpu"] + (
    ["xpu"] if torch_device_type == "xpu" and torch_dev.is_available() else []
)

_MODEL = "gha-test-model"
_HEADS, _HEAD_DIM = 4, 64
_HIDDEN = _HEADS * _HEAD_DIM


def _key(group: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=b"\x01" * 32, model_name=_MODEL, kv_rank=0, object_group_id=group
    )


@dataclass
class _Obj:
    """MemoryObj stand-in exposing the accessors the serde uses."""

    parts: list[torch.Tensor]
    used: int | None = None

    @property
    def tensor(self) -> torch.Tensor:
        return self.parts[0]

    def get_tensor(self, index: int) -> torch.Tensor:
        return self.parts[index]

    def get_shapes(self) -> list[torch.Size]:
        return [p.shape for p in self.parts]

    def get_dtypes(self) -> list[torch.dtype]:
        return [p.dtype for p in self.parts]

    def set_used_size(self, n: int) -> None:
        self.used = n


def _wait(processor, kind: str, task_id: int) -> bool:
    query = getattr(processor, f"query_{kind}_result")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = query(task_id)
        if result is not None:
            return result
        time.sleep(0.005)
    raise TimeoutError(kind)


def _round_trip(processor, src: _Obj, group: int) -> tuple[_Obj, _Obj]:
    """Serialize ``src`` then deserialize into a fresh object; return both."""
    key = _key(group)
    layout = MemoryLayoutDesc(shapes=src.get_shapes(), dtypes=src.get_dtypes())
    size = processor.estimate_serialized_size_for_key(layout, key)
    stored = _Obj([torch.zeros(size, dtype=torch.uint8)])
    assert _wait(
        processor, "serialize", processor.submit_serialize([src], [stored], [key])
    )
    # What the L2 adapter persists is the narrowed buffer.
    assert stored.used is not None
    loaded = _Obj([torch.zeros(size, dtype=torch.uint8)])
    loaded.parts[0].copy_(stored.parts[0])
    out = _Obj([torch.zeros_like(p) for p in src.parts])
    assert _wait(
        processor, "deserialize", processor.submit_deserialize([loaded], [out], [key])
    )
    return stored, out


_SPLIT = {"cache_category": "attention", "attention_plane_layout": "split_kv"}
_FUSED = {**_SPLIT, "attention_plane_layout": "fused_kv"}
_MLA = {"cache_category": "attention", "attention_plane_layout": "mla"}
_UNKNOWN = {"cache_category": "unknown"}


def _processor(device: str, **extra):
    kwargs = dict(qbit=4, device=device, num_threads=2)
    kwargs.update(extra)
    return create_serde_processor(SerdeConfig(type="gha", kwargs=kwargs))


def test_gha_is_registered() -> None:
    assert "gha" in get_registered_serde_types()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("per_tensor", [False, True])
@pytest.mark.parametrize("qbit,max_ratio", [(4, 0.30), (8, 0.55)])
def test_split_attention_round_trip(device, per_tensor, qbit, max_ratio) -> None:
    """[2, L, T, H] chunks shrink, stay close, and the estimate is exact."""
    processor = _processor(
        device, qbit=qbit, per_tensor=per_tensor, groups={0: [_SPLIT]}
    )
    try:
        torch.manual_seed(0)
        x = torch.randn(2, 4, 128, _HIDDEN).half()
        stored, out = _round_trip(processor, _Obj([x]), 0)
        assert stored.used == stored.parts[0].numel()  # estimate == actual size
        assert stored.used / (x.numel() * 2) < max_ratio
        tolerance = 0.5 if per_tensor and qbit == 4 else 0.15
        assert (out.tensor.float() - x.float()).abs().mean() < tolerance
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tokens", [16, 48])
def test_split_chunk_not_a_multiple_of_64_tokens(device, tokens) -> None:
    """Regression: per_token chunks shorter than / not aligned to 64 tokens
    used to make the native kernels read and write out of bounds."""
    processor = _processor(device, groups={0: [_SPLIT]})
    try:
        torch.manual_seed(4)
        x = torch.randn(2, 3, tokens, _HIDDEN).half()
        stored, out = _round_trip(processor, _Obj([x]), 0)
        assert stored.used == stored.parts[0].numel()
        assert (out.tensor.float() - x.float()).abs().mean() < 0.15
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("per_tensor", [False, True])
def test_fused_attention_round_trip(device, per_tensor) -> None:
    """[L, T, H] chunks of a fused-packed format use the fused codec."""
    processor = _processor(device, per_tensor=per_tensor, groups={1: [_FUSED]})
    try:
        torch.manual_seed(1)
        x = torch.randn(4, 128, 2 * _HIDDEN).half()
        stored, out = _round_trip(processor, _Obj([x]), 1)
        assert stored.used == stored.parts[0].numel()
        assert stored.used < x.numel() * 2 * 0.30
        tolerance = 0.5 if per_tensor else 0.15
        assert (out.tensor.float() - x.float()).abs().mean() < tolerance
    finally:
        processor.close()


@pytest.mark.parametrize("spec", [_MLA, _UNKNOWN])
def test_unquantizable_groups_pass_through_bit_exact(spec) -> None:
    """Groups the codec must not touch are stored and restored byte for byte."""
    processor = _processor("cpu", groups={2: [spec]})
    try:
        x = torch.randn(4, 128, 128).half()
        stored, out = _round_trip(processor, _Obj([x]), 2)
        assert stored.used == x.numel() * 2
        assert torch.equal(out.tensor.view(torch.uint8), x.view(torch.uint8))
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "shape", [(2, 4, 128, _HIDDEN), (4, 128, 2 * _HIDDEN)], ids=["split", "fused"]
)
def test_unconfigured_group_layout_is_read_from_the_shape(device, shape) -> None:
    """With no ``groups``, [2,L,T,H] is quantized as split K/V and [L,T,H] as
    fused K/V; the size estimate stays exact."""
    processor = create_serde_processor(
        SerdeConfig(type="gha", kwargs={"device": device, "num_threads": 2})
    )
    try:
        torch.manual_seed(5)
        x = torch.randn(*shape).half()
        stored, out = _round_trip(processor, _Obj([x]), 7)
        assert stored.used == stored.parts[0].numel()
        assert stored.used < x.numel() * 2 * 0.30
        assert (out.tensor.float() - x.float()).abs().mean() < 0.15
    finally:
        processor.close()


def test_unconfigured_group_with_other_shapes_passes_through() -> None:
    """A chunk that is neither [2,L,T,H] nor [L,T,H] is never mis-quantized."""
    processor = create_serde_processor(
        SerdeConfig(type="gha", kwargs={"device": "cpu"})
    )
    try:
        x = torch.randn(3, 4, 128, 64).half()  # rank 4 but not a K/V pair
        stored, out = _round_trip(processor, _Obj([x]), 7)
        assert stored.used == x.numel() * 2
        assert torch.equal(out.tensor.view(torch.uint8), x.view(torch.uint8))
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
def test_multi_part_object_round_trips_every_part(device) -> None:
    """An object holding one tensor per kernel group is framed and restored per part."""
    processor = _processor(device, groups={4: [_SPLIT, _MLA]})
    try:
        torch.manual_seed(3)
        parts = [
            torch.randn(2, 2, 128, _HIDDEN).half(),
            torch.randn(2, 128, 128).half(),
        ]
        stored, out = _round_trip(processor, _Obj(parts), 4)
        assert stored.used == stored.parts[0].numel()
        assert (out.parts[0].float() - parts[0].float()).abs().mean() < 0.15
        assert torch.equal(out.parts[1].view(torch.uint8), parts[1].view(torch.uint8))
    finally:
        processor.close()


def test_estimate_depends_on_the_objects_group() -> None:
    """Same shape, different group: the buffer is sized for that key's group."""
    processor = _processor("cpu", groups={0: [_SPLIT], 1: [_UNKNOWN]})
    try:
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 128, _HIDDEN])], dtypes=[torch.float16]
        )
        quantized = processor.estimate_serialized_size_for_key(layout, _key(0))
        raw = processor.estimate_serialized_size_for_key(layout, _key(1))
        assert raw == 2 * 4 * 128 * _HIDDEN * 2
        assert quantized < raw
    finally:
        processor.close()


def test_corrupt_payload_fails_instead_of_returning_garbage() -> None:
    """A buffer that is not a gha payload makes the deserialize task fail."""
    processor = _processor("cpu", groups={0: [_SPLIT]})
    try:
        x = torch.randn(2, 4, 128, _HIDDEN).half()
        layout = MemoryLayoutDesc(shapes=[x.shape], dtypes=[x.dtype])
        size = processor.estimate_serialized_size_for_key(layout, _key(0))
        junk = _Obj([torch.full((size,), 0x5A, dtype=torch.uint8)])
        out = _Obj([torch.zeros_like(x)])
        task = processor.submit_deserialize([junk], [out], [_key(0)])
        assert _wait(processor, "deserialize", task) is False
    finally:
        processor.close()


def test_per_tensor_uses_one_scale_per_plane() -> None:
    """per_tensor drops the per-token / per-channel scales from the payload."""
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 4, 128, _HIDDEN])], dtypes=[torch.float16]
    )
    sizes = {}
    for per_tensor in (False, True):
        processor = _processor("cpu", per_tensor=per_tensor, groups={0: [_SPLIT]})
        try:
            sizes[per_tensor] = processor.estimate_serialized_size_for_key(
                layout, _key(0)
            )
        finally:
            processor.close()
    assert sizes[True] < sizes[False]


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="qbit"):
        create_serde_processor(SerdeConfig(type="gha", kwargs={"qbit": 3}))
    with pytest.raises(ValueError, match="unknown config keys"):
        create_serde_processor(SerdeConfig(type="gha", kwargs={"head_dim": 64}))
    with pytest.raises(ValueError, match="unsupported group keys"):
        create_serde_processor(
            SerdeConfig(
                type="gha",
                kwargs={"groups": {0: [{**_SPLIT, "head_dim": 64}]}},
            )
        )

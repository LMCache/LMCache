# SPDX-License-Identifier: Apache-2.0
"""RBLN block transfer on a real NPU.

``test_rbln_kv_ops.py`` pins the chunk layout with CPU tensors, which never
exercises the device dispatch: the HND path's host staging buffer and the MLA
path's device staging buffer only diverge from the CPU behaviour once the
paged KV lives on ``rbln``. These tests store from device-resident paged KV
into host chunks and retrieve them into a zeroed device cache, then compare
both the chunks and the restored cache against the CPU result.

They skip unless ``torch.rbln`` reports a usable NPU, so they run only on the
``rbln-mp-test`` Buildkite lane.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
MLA_HEAD_SIZE = 16
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    not RblnDeviceSpec().is_available(), reason="RBLN NPU is required"
)


def _shape_desc(kv_size: int, num_heads: int, head_size: int) -> PageBufferShapeDesc:
    """Descriptor for ``NUM_LAYERS`` x ``NUM_BLOCKS`` paged layers."""
    desc = PageBufferShapeDesc()
    desc.kv_size = kv_size
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = num_heads
    desc.hs = head_size
    desc.element_size = DTYPE.itemsize
    return desc


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> None:
    """Move every block between ``layers`` and ``chunks``."""
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        layers[0].device,
        direction,
        shape_desc,
        CHUNK_TOKENS,
        engine_kv_format,
        0,
    )


def _assert_device_round_trip(
    layer_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    engine_kv_format: EngineKVFormat,
    shape_desc: PageBufferShapeDesc,
) -> None:
    """Store and retrieve on ``rbln:0`` and compare against the CPU path."""
    torch.manual_seed(7)
    cpu_layers = [torch.randn(layer_shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]
    num_chunks = NUM_BLOCKS // BLOCKS_PER_CHUNK

    expected_chunks = [torch.zeros(chunk_shape, dtype=DTYPE) for _ in range(num_chunks)]
    _transfer(
        cpu_layers,
        expected_chunks,
        TransferDirection.D2H,
        engine_kv_format,
        shape_desc,
    )

    device = torch.device("rbln", 0)
    device_layers = [layer.to(device) for layer in cpu_layers]
    chunks = [torch.zeros(chunk_shape, dtype=DTYPE) for _ in range(num_chunks)]
    _transfer(
        device_layers, chunks, TransferDirection.D2H, engine_kv_format, shape_desc
    )
    torch.rbln.synchronize()
    for chunk, expected in zip(chunks, expected_chunks, strict=True):
        assert torch.equal(chunk, expected)

    restored = [torch.zeros_like(layer) for layer in device_layers]
    _transfer(restored, chunks, TransferDirection.H2D, engine_kv_format, shape_desc)
    torch.rbln.synchronize()
    for layer, expected in zip(restored, cpu_layers, strict=True):
        assert torch.equal(layer.cpu(), expected)


def test_hnd_round_trip_on_device() -> None:
    """The 6-D HND cache survives store and retrieve on a real NPU."""
    _assert_device_round_trip(
        (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE),
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        _shape_desc(kv_size=2, num_heads=NUM_HEADS, head_size=HEAD_SIZE),
    )


def test_mla_round_trip_on_device() -> None:
    """The MLA cache survives store and retrieve on a real NPU."""
    _assert_device_round_trip(
        (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE),
        (NUM_LAYERS, CHUNK_TOKENS, MLA_HEAD_SIZE),
        EngineKVFormat.NL_X_NB_BS_HS,
        _shape_desc(kv_size=1, num_heads=1, head_size=MLA_HEAD_SIZE),
    )

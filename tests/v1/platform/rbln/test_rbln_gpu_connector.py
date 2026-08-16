# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN vLLM paged-memory connector.

The connector's job is to make vLLM-RBLN's native 6-D KV cache
``[2, NB, NH, 1, BS, HS]`` usable by upstream LMCache. It hands the caches to
discovery as registered -- they are their own ``NL_X_TWO_NB_NH_ONE_BS_HS`` --
and squeezes the always-1 axis 3 only where it indexes the bytes, which is
where it needs the 5-D views for slot indexing anyway.

The load-bearing test is the round trip. HND puts the head axis *between*
blocks and block tokens, so the flat ``view(num_blocks * block_size, ...)``
reshape the NHD connectors use would silently address the wrong slots; only a
gather/scatter comparison against an independently computed expectation catches
that.

Runs on CPU -- no RBLN hardware needed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.rbln_connector import VLLMPagedMemRBLNConnectorV2
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat

NUM_LAYERS = 3
NUM_BLOCKS = 6
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE
DTYPE = torch.float32


def _native_kv(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer KV in the native RBLN 6-D layout."""
    torch.manual_seed(7)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _allocator() -> TensorMemoryAllocator:
    """Host-backed allocator; no pinned or device memory required."""
    return TensorMemoryAllocator(torch.empty(16 * 1024 * 1024, dtype=torch.uint8))


def _slot_mapping(num_tokens: int) -> torch.Tensor:
    """Slots spread across blocks so block *and* offset both vary."""
    return torch.arange(num_tokens, dtype=torch.long)


def _connector_with_attributes() -> VLLMPagedMemRBLNConnectorV2:
    """A connector that has already discovered its geometry."""
    connector = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    connector.register_kv_caches(_native_kv())
    return connector


# ---------------------------------------------------------------------------
# Squeeze
# ---------------------------------------------------------------------------


def test_squeeze_produces_shared_storage_hnd_views() -> None:
    """The squeeze is a free view, not a copy."""
    native = _native_kv()
    views = squeeze_singleton_axis(native)
    assert [tuple(v.shape) for v in views] == [
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
    ] * NUM_LAYERS
    for view, tensor in zip(views, native, strict=True):
        assert view.data_ptr() == tensor.data_ptr()


@pytest.mark.parametrize(
    "shape",
    [
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_BLOCKS, NUM_HEADS, 2, BLOCK_SIZE, HEAD_SIZE),
    ],
    ids=["5d", "non-singleton-axis"],
)
def test_squeeze_rejects_unexpected_layouts(shape: tuple[int, ...]) -> None:
    """A layout that is not 6-D with a singleton fails loudly."""
    with pytest.raises(ValueError, match=r"\[2, NB, NH, 1, BS, HS\]"):
        squeeze_singleton_axis([torch.zeros(shape)])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovers_the_native_format() -> None:
    """The registered 6-D caches resolve to the native RBLN format.

    No layout hint is passed: the format is HND by definition, so both RBLN
    paths report the same format for the same cache.
    """
    connector = _connector_with_attributes()
    assert int(connector.engine_kv_format) == int(
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
    )


def test_geometry_matches_the_native_tensors() -> None:
    """Discovery reads the real dims, not the ones shifted by the singleton."""
    connector = _connector_with_attributes()
    assert connector.num_layers == NUM_LAYERS
    assert connector.num_blocks == NUM_BLOCKS
    assert connector.block_size == BLOCK_SIZE
    assert connector.num_heads == NUM_HEADS
    assert connector.head_size == HEAD_SIZE
    assert connector.hidden_dim_size == HIDDEN_DIM
    assert connector.dtype == DTYPE


def test_get_shape_is_kv_2ltd() -> None:
    """Memory objects follow the standard vLLM connector contract."""
    connector = _connector_with_attributes()
    assert connector.get_shape(12) == torch.Size([2, NUM_LAYERS, 12, HIDDEN_DIM])


def test_get_shape_needs_no_registration() -> None:
    """The metadata fixes the shape, so a memory object can be sized first."""
    connector = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    assert connector.get_shape(4) == torch.Size([2, NUM_LAYERS, 4, HIDDEN_DIM])


def test_caches_disagreeing_with_the_metadata_are_refused() -> None:
    """A cache built for another model would break get_shape()'s sizing."""
    connector = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS + 1
    )
    with pytest.raises(ValueError, match="num_layers"):
        connector.register_kv_caches(_native_kv())


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------


def test_gather_matches_an_independent_hnd_expectation() -> None:
    """`from_gpu` reads the slots HND addressing implies, not a flat reshape."""
    native = _native_kv()
    connector = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    num_tokens = NUM_BLOCKS * BLOCK_SIZE
    slot_mapping = _slot_mapping(num_tokens)

    connector.register_kv_caches(native)
    memory_obj = _allocator().allocate(connector.get_shape(num_tokens), DTYPE)
    assert memory_obj is not None
    connector.from_gpu(
        memory_obj, 0, num_tokens, kvcaches=native, slot_mapping=slot_mapping
    )

    # Independent expectation: token t lives at block t // BS, offset t % BS,
    # and its heads are laid out before the block-token axis.
    for layer_idx, layer in enumerate(native):
        squeezed = layer.squeeze(3)
        for token in range(num_tokens):
            block, offset = divmod(token, BLOCK_SIZE)
            for kv in (0, 1):
                expected = squeezed[kv, block, :, offset, :].reshape(HIDDEN_DIM)
                assert torch.equal(memory_obj.tensor[kv, layer_idx, token], expected)


def test_round_trip_restores_the_native_cache() -> None:
    """Gather then scatter reproduces the source cache exactly."""
    src = _native_kv()
    dst = _native_kv(fill_random=False)
    num_tokens = NUM_BLOCKS * BLOCK_SIZE
    slot_mapping = _slot_mapping(num_tokens)

    reader = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    writer = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    reader.register_kv_caches(src)
    memory_obj = _allocator().allocate(reader.get_shape(num_tokens), DTYPE)
    assert memory_obj is not None

    reader.from_gpu(memory_obj, 0, num_tokens, kvcaches=src, slot_mapping=slot_mapping)
    writer.to_gpu(memory_obj, 0, num_tokens, kvcaches=dst, slot_mapping=slot_mapping)

    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_partial_slice_touches_only_its_tokens() -> None:
    """A [start, end) slice leaves every other slot untouched."""
    src = _native_kv()
    dst = _native_kv(fill_random=False)
    num_tokens = NUM_BLOCKS * BLOCK_SIZE
    slot_mapping = _slot_mapping(num_tokens)
    start, end = BLOCK_SIZE, 3 * BLOCK_SIZE

    reader = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    writer = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    reader.register_kv_caches(src)
    memory_obj = _allocator().allocate(reader.get_shape(end - start), DTYPE)
    assert memory_obj is not None

    reader.from_gpu(memory_obj, start, end, kvcaches=src, slot_mapping=slot_mapping)
    writer.to_gpu(memory_obj, start, end, kvcaches=dst, slot_mapping=slot_mapping)

    for got, expected in zip(dst, src, strict=True):
        written = slice(start // BLOCK_SIZE, end // BLOCK_SIZE)
        assert torch.equal(got[:, written], expected[:, written])
        # Blocks outside the slice stay zero.
        assert torch.count_nonzero(got[:, : start // BLOCK_SIZE]) == 0
        assert torch.count_nonzero(got[:, end // BLOCK_SIZE :]) == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_missing_slot_mapping_is_refused() -> None:
    """The slot mapping is required, not optional."""
    native = _native_kv()
    connector = _connector_with_attributes()
    memory_obj = _allocator().allocate(connector.get_shape(4), DTYPE)
    assert memory_obj is not None
    with pytest.raises(ValueError, match="slot_mapping"):
        connector.from_gpu(memory_obj, 0, 4, kvcaches=native)


def test_missing_kvcaches_is_refused() -> None:
    """Nothing can be transferred before the caches are known.

    The shape comes from a separately registered connector so the connector
    under test has never seen a KV cache at all.
    """
    memory_obj = _allocator().allocate(_connector_with_attributes().get_shape(4), DTYPE)
    assert memory_obj is not None
    unregistered = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    with pytest.raises(ValueError, match="kvcaches"):
        unregistered.from_gpu(memory_obj, 0, 4, slot_mapping=_slot_mapping(4))


def test_non_kv_2ltd_memory_object_is_refused() -> None:
    """The connector only speaks KV_2LTD."""
    native = _native_kv()
    connector = _connector_with_attributes()
    memory_obj = _allocator().allocate(
        connector.get_shape(4), DTYPE, fmt=MemoryFormat.KV_MLA_FMT
    )
    assert memory_obj is not None
    with pytest.raises(ValueError, match="KV_2LTD"):
        connector.from_gpu(
            memory_obj, 0, 4, kvcaches=native, slot_mapping=_slot_mapping(4)
        )


def test_explicit_registration_enables_get_shape() -> None:
    """`register_kv_caches` discovers geometry without a transfer first."""
    connector = VLLMPagedMemRBLNConnectorV2(
        hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
    )
    connector.register_kv_caches(_native_kv())
    assert connector.get_shape(8) == torch.Size([2, NUM_LAYERS, 8, HIDDEN_DIM])


def test_registering_nothing_is_refused() -> None:
    """An empty layer list is not a valid registration."""
    with pytest.raises(ValueError, match="non-empty"):
        VLLMPagedMemRBLNConnectorV2(
            hidden_dim_size=HIDDEN_DIM, num_layers=NUM_LAYERS
        ).register_kv_caches([])

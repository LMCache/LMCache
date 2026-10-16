# SPDX-License-Identifier: Apache-2.0
"""VLLMPagedMemGPUConnectorV3 with kernel groups of different KV layouts.

Models like GLM-5.x / DeepSeek-V3.2 (DSA) register an indexer k-cache
(``NL_X_NB_BSV_BSS``, 132 B per token) next to the MLA latent cache
(``NL_X_NB_BS_HS``, 576 B per token). Each kernel group must be transferred
with its own layout.

These tests run on CPU: CUDA streams are stubbed, the connector's device is
pointed at the CPU and ``multi_layer_kv_transfer`` is replaced by a recorder,
so they check the layout arguments each kernel group's transfer receives.
"""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import gpu_connectors
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.lmcache_native as lmcache_native

NUM_BLOCKS = 16
BLOCK_SIZE = 64
INDEXER_HS = 132
MLA_HS = 576
CHUNK = 256
BSV_BSS = lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS
BS_HS = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS


def _layer(head_size: int) -> torch.Tensor:
    return torch.zeros(NUM_BLOCKS, BLOCK_SIZE, head_size, dtype=torch.uint8)


def _mixed_kv_caches(num_layer_pairs: int = 2) -> list[torch.Tensor]:
    """Alternating indexer / MLA layers, as a DSA model registers them."""
    caches = []
    for _ in range(num_layer_pairs):
        caches.append(_layer(INDEXER_HS))
        caches.append(_layer(MLA_HS))
    return caches


def _mla_kv_caches(num_layers: int = 4) -> list[torch.Tensor]:
    return [_layer(MLA_HS) for _ in range(num_layers)]


def _make_connector(kv_caches: list[torch.Tensor]) -> VLLMPagedMemGPUConnectorV3:
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(len(kv_caches), 1, CHUNK, 1, MLA_HS),
        use_mla=True,
        chunk_size=CHUNK,
    )
    with patch.object(torch.cuda, "Stream", MagicMock):
        connector = VLLMPagedMemGPUConnectorV3(metadata, device=torch.device("cuda"))
    connector.device = torch.device("cpu")
    connector.build_kv_layer_groups(kv_caches)
    return connector


class _MemoryObj:
    """Minimal memory object: one uint8 tensor per group."""

    def __init__(self, shapes: list[torch.Size]):
        self.metadata = SimpleNamespace(fmt=MemoryFormat.KV_MLA_FMT, shapes=shapes)
        self._tensors = [torch.zeros(s, dtype=torch.uint8) for s in shapes]
        self.raw_tensor = self._tensors[0]

    def get_tensor(self, index: int) -> torch.Tensor:
        return self._tensors[index]


def _transfer(connector, memory_obj, kv_caches, direction: str) -> list[dict]:
    """Run to_gpu/from_gpu for one chunk; return the recorded transfer calls."""
    calls: list[dict] = []

    def record(lmc_tensor, _ptrs, _slots, _device, page_buffer_size, _dir, fmt, **kw):
        calls.append(
            {
                "hidden_dim": lmc_tensor.shape[-1],
                "page_buffer_size": page_buffer_size,
                "engine_kv_format": fmt,
                **kw,
            }
        )

    kwargs = {
        "kvcaches": kv_caches,
        "slot_mapping": torch.arange(CHUNK, dtype=torch.int64),
    }
    with (
        patch.object(
            gpu_connectors.device_ops, "multi_layer_kv_transfer", side_effect=record
        ),
        patch.object(torch.cuda, "stream", lambda _stream: nullcontext()),
    ):
        getattr(connector, direction)(memory_obj, 0, CHUNK, **kwargs)
    return calls


@pytest.mark.parametrize("direction", ["to_gpu", "from_gpu"])
def test_each_kernel_group_uses_its_own_layout(direction):
    kv_caches = _mixed_kv_caches()
    connector = _make_connector(kv_caches)
    memory_obj = _MemoryObj(connector.metadata.get_shapes(CHUNK))

    calls = _transfer(connector, memory_obj, kv_caches, direction)

    by_width = {c["hidden_dim"]: c for c in calls}
    assert sorted(by_width) == [INDEXER_HS, MLA_HS]
    indexer, mla = by_width[INDEXER_HS], by_width[MLA_HS]
    assert indexer["engine_kv_format"] == BSV_BSS
    assert indexer["head_size"] == INDEXER_HS
    assert indexer["block_stride_elems"] == BLOCK_SIZE * INDEXER_HS
    assert mla["engine_kv_format"] == BS_HS
    assert mla["head_size"] == MLA_HS
    assert mla["block_stride_elems"] == BLOCK_SIZE * MLA_HS
    for call in calls:
        assert call["block_size"] == BLOCK_SIZE
        assert call["page_buffer_size"] == NUM_BLOCKS * BLOCK_SIZE


def test_shapes_are_per_group_once_kv_caches_are_registered():
    """The first chunk's memory objects must already get per-group shapes."""
    connector = _make_connector(_mixed_kv_caches())

    shapes = [tuple(s) for s in connector.metadata.get_shapes(CHUNK)]

    assert sorted(shapes, key=lambda s: s[-1]) == [
        (1, 2, CHUNK, INDEXER_HS),
        (1, 2, CHUNK, MLA_HS),
    ]


@pytest.mark.parametrize("direction", ["to_gpu", "from_gpu"])
def test_memory_obj_with_wrong_layout_is_rejected(direction):
    """A single-group object allocated for a mixed-layout model must not be
    silently copied into (it would corrupt the KV cache)."""
    kv_caches = _mixed_kv_caches()
    connector = _make_connector(kv_caches)
    legacy = _MemoryObj([torch.Size([1, len(kv_caches), CHUNK, MLA_HS])])

    with pytest.raises(RuntimeError, match="kernel group"):
        _transfer(connector, legacy, kv_caches, direction)


@pytest.mark.parametrize("direction", ["to_gpu", "from_gpu"])
def test_single_group_layout_is_unchanged(direction):
    kv_caches = _mla_kv_caches()
    connector = _make_connector(kv_caches)
    memory_obj = _MemoryObj(connector.metadata.get_shapes(CHUNK))

    calls = _transfer(connector, memory_obj, kv_caches, direction)

    assert len(calls) == 1
    (call,) = calls
    assert call["engine_kv_format"] == BS_HS == connector.engine_kv_format
    assert call["head_size"] == MLA_HS == connector.head_size
    assert call["block_size"] == BLOCK_SIZE == connector.block_size
    assert call["page_buffer_size"] == connector.page_buffer_size


def test_register_kv_caches_builds_layer_groups_before_post_init():
    """The vLLM adapter builds the connector's layer groups before the
    storage backends are created, so no memory object gets a stale shape."""
    pytest.importorskip("vllm")
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

    order = MagicMock()
    fake_impl = SimpleNamespace(
        kv_caches={},
        lmcache_engine=SimpleNamespace(gpu_connector=order.connector),
        _manager=order.manager,
    )
    kv_caches = {f"layer.{i}": t for i, t in enumerate(_mixed_kv_caches())}

    LMCacheConnectorV1Impl.register_kv_caches(fake_impl, kv_caches)

    assert [name for name, *_ in order.mock_calls] == [
        "connector.build_kv_layer_groups",
        "manager.post_init",
    ]
    assert order.connector.build_kv_layer_groups.call_args.args[0] == list(
        kv_caches.values()
    )

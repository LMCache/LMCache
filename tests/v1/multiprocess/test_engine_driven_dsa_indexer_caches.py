# SPDX-License-Identifier: Apache-2.0
"""Engine-driven transfer of vLLM-RBLN's fp8 DeepSeek-V3.2 (DSA) KV caches.

With ``--kv-cache-dtype fp8`` vLLM-RBLN binds three caches per decoder layer,
all in one vLLM KV cache group:

- the packed MLA latent, ``[NB, BS, HS]`` (int8),
- the lightning-indexer key, ``[NB, BS, 128]`` (float8_e4m3fn),
- the indexer key's scale, ``[NB, BS]`` (float16, one value per token).

The scale cache is the only rank-2 KV cache vLLM hands LMCache. These tests
pin that registration gives each cache its own group (the scale cache as a
one-wide plane) and that the scale cache round-trips through the MP
gather/scatter helpers.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

NUM_LAYERS = 2
NUM_BLOCKS = 8
BLOCK_SIZE = 4
LATENT_SIZE = 24
INDEX_HEAD_SIZE = 16
BLOCKS_IN_CHUNK = 2
CHUNK_TOKENS = BLOCKS_IN_CHUNK * BLOCK_SIZE


def _dsa_caches() -> dict[str, torch.Tensor]:
    """Per-layer latent, indexer key and indexer scale caches, layer-major."""
    kv: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        kv[f"layers.{i}.attn"] = torch.zeros(
            NUM_BLOCKS, BLOCK_SIZE, LATENT_SIZE, dtype=torch.int8
        )
        kv[f"layers.{i}.indexer.k_cache"] = torch.zeros(
            NUM_BLOCKS, BLOCK_SIZE, INDEX_HEAD_SIZE, dtype=torch.float8_e4m3fn
        )
        kv[f"layers.{i}.indexer.k_cache.scale_cache"] = torch.zeros(
            NUM_BLOCKS, BLOCK_SIZE, dtype=torch.float16
        )
    return kv


def _register(kv: dict[str, torch.Tensor]) -> RegisterEngineDrivenContextPayload:
    """Register ``kv`` as three groups over one block-id space; return the payload."""
    sent: list[Any] = []

    def _register_rpc(payload: Any) -> Any:
        sent.append(payload)
        future = MagicMock()
        # No SHM pool: the worker goes straight to the inline transport.
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="", pool_size=0
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register_rpc
    EngineDrivenTransferContext(1, req_client).register(
        kv_caches=kv,
        model_name="dsa",
        world_size=1,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        mq_timeout=1.0,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(plane, plane + 3),
                tokens_per_block=BLOCK_SIZE,
            )
            for plane in range(3)
        ],
    )
    return sent[0]


def test_register_gives_each_dsa_cache_its_own_layout() -> None:
    """The latent, key and scale caches register as three single-plane groups,
    each with its own width and dtype; the scale cache is one value wide."""
    payload = _register(_dsa_caches())

    assert [
        (layout.hidden_dim_size, layout.dtype_str, layout.kv_size)
        for layout in payload.group_layouts
    ] == [
        (LATENT_SIZE, "int8", 1),
        (INDEX_HEAD_SIZE, "float8_e4m3fn", 1),
        (1, "float16", 1),
    ]
    assert [layout.window_tokens for layout in payload.group_layouts] == [
        CHUNK_TOKENS
    ] * 3


def test_scale_cache_round_trips_through_gather_and_scatter() -> None:
    """Gather packs each block's scales token-major into ``[L, T, 1]`` chunks;
    scatter writes them back to the same blocks and nowhere else."""
    torch.manual_seed(0)
    src = {
        f"layers.{i}.indexer.k_cache.scale_cache": torch.randn(
            NUM_BLOCKS, BLOCK_SIZE, dtype=torch.float16
        )
        for i in range(NUM_LAYERS)
    }
    block_ids = [2, 5, 3, 7]

    chunks = gather_paged_kv_to_cpu(src, block_ids, BLOCKS_IN_CHUNK)

    assert [tuple(chunk.shape) for chunk in chunks] == [
        (NUM_LAYERS, CHUNK_TOKENS, 1)
    ] * 2
    for layer, scales in enumerate(src.values()):
        assert torch.equal(chunks[0][layer, :, 0], torch.cat([scales[2], scales[5]]))
        assert torch.equal(chunks[1][layer, :, 0], torch.cat([scales[3], scales[7]]))

    dst = {name: torch.zeros_like(scales) for name, scales in src.items()}
    scatter_cpu_to_paged_kv(dst, block_ids, chunks, BLOCKS_IN_CHUNK)

    untouched = [b for b in range(NUM_BLOCKS) if b not in block_ids]
    for name, scales in src.items():
        assert torch.equal(dst[name][block_ids], scales[block_ids])
        assert not dst[name][untouched].any()

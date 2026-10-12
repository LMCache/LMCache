# SPDX-License-Identifier: Apache-2.0
"""Engine-driven multi-group transfer over kernel-paged KV groups.

vLLM may page a group's tensor at a kernel block size smaller than its block
size (``kernel_block_size``): engine block ``b`` then covers tensor pages
``b*k .. b*k+k-1``. vLLM-RBLN does this for a sliding-window group, paging
4096-token blocks as 128-token windows. These tests pin that registration
reads such a group as kernel-paged rather than compressed, that store and
retrieve move tensor pages, that chunk counting (null chunks, retrieve
windows) still runs on engine block IDs, and that a sliding window of one
page keeps only each chunk's last page.

Tensors are ``[2, NB, X, X, HS]`` with the two middle axes equal, so the block
size is the same whichever of NHD / HND the detector resolves.
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
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

NUM_BLOCKS = 5  # engine block IDs per group; block 0 is the null block
BLOCK_TOKENS = 8  # tokens one engine block ID covers
PAGES_PER_BLOCK = 2  # kernel pages per engine block in the paged group
PAGE_TOKENS = BLOCK_TOKENS // PAGES_PER_BLOCK
HEAD_SIZE = 4
BLOCKS_IN_CHUNK = 2
CHUNK_TOKENS = BLOCKS_IN_CHUNK * BLOCK_TOKENS


def _kv(seed: int = 0) -> dict[str, torch.Tensor]:
    """A full-attention layer and a kernel-paged sliding-window layer."""
    torch.manual_seed(seed)
    return {
        "full": torch.randn(2, NUM_BLOCKS, BLOCK_TOKENS, BLOCK_TOKENS, HEAD_SIZE),
        "swa": torch.randn(
            2, NUM_BLOCKS * PAGES_PER_BLOCK, PAGE_TOKENS, PAGE_TOKENS, HEAD_SIZE
        ),
    }


def _group_infos() -> list[EngineGroupInfo]:
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=BLOCK_TOKENS,
            num_blocks=NUM_BLOCKS,
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=BLOCK_TOKENS,
            sw_size_tokens=PAGE_TOKENS,
            num_blocks=NUM_BLOCKS,
        ),
    ]


def _register(
    kv: dict[str, torch.Tensor],
) -> tuple[EngineDrivenTransferContext, RegisterEngineDrivenContextPayload]:
    """Register ``kv`` and return the context and the payload it sent."""
    sent: list[Any] = []

    def _register_rpc(payload):
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_x", pool_size=4096
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register_rpc
    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        mq_timeout=1.0,
        engine_group_infos=_group_infos(),
    )
    return ctx, sent[0]


def _inline_ctx(
    monkeypatch,
) -> tuple[EngineDrivenTransferContext, EngineDrivenContextInline]:
    """A registered context over a real inline transfer context."""
    ctx, _ = _register(_kv())
    ictx = EngineDrivenContextInline(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = ictx
    monkeypatch.setattr(ictx, "prepare_store", lambda _k, _i: None)
    return ctx, ictx


def _capture_gather(monkeypatch) -> list[dict]:
    """Replace gather with a fake recording its block IDs and chunk indices."""
    gathered: list[dict] = []

    def _fake_gather(_kv, block_ids, _blocks_in_chunk, **kwargs: Any):
        gathered.append(
            {"block_ids": block_ids, "chunk_indices": kwargs["chunk_indices"]}
        )
        return [torch.zeros(1) for _ in kwargs["chunk_indices"]]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)
    return gathered


def test_kernel_paged_window_group_registers_in_pages() -> None:
    """The paged group is not refused as compressed; its counts are in pages."""
    ctx, payload = _register(_kv())

    full, swa = ctx._group_states
    assert full.kernel_pages_per_block == 1
    assert full.blocks_in_chunk == full.blocks_per_window == BLOCKS_IN_CHUNK
    assert swa.kernel_pages_per_block == PAGES_PER_BLOCK
    assert swa.blocks_in_chunk == CHUNK_TOKENS // PAGE_TOKENS
    assert swa.blocks_per_window == 1
    assert payload.group_layouts[0].window_tokens == CHUNK_TOKENS
    assert payload.group_layouts[1].window_tokens == PAGE_TOKENS


def test_store_expands_engine_block_ids_to_pages(monkeypatch) -> None:
    """Gather receives page IDs ``b*k .. b*k+k-1`` for the paged group only."""
    ctx, ictx = _inline_ctx(monkeypatch)
    monkeypatch.setattr(ictx, "commit_store", lambda _k, _i, _c: True)
    gathered = _capture_gather(monkeypatch)

    future = ctx.submit_store("req", MagicMock(), _kv(), [[3, 1], [3, 1]], None, 2)

    assert future.result(timeout=1) is True
    assert [g["block_ids"] for g in gathered] == [[3, 1], [6, 7, 2, 3]]


def test_store_skips_all_null_chunks_by_engine_block(monkeypatch) -> None:
    """A chunk whose engine blocks are all null is skipped for the paged group
    too: null chunks are counted in engine block IDs, not in pages."""
    ctx, ictx = _inline_ctx(monkeypatch)
    committed: list[Any] = []
    monkeypatch.setattr(
        ictx, "commit_store", lambda _k, _i, chunks: committed.append(chunks) or True
    )
    gathered = _capture_gather(monkeypatch)

    # Chunk 0 is [0, 0] (all null) in both groups; chunk 1 is [3, 1].
    block_ids = [[0, 0, 3, 1], [0, 0, 3, 1]]
    future = ctx.submit_store("req", MagicMock(), _kv(), block_ids, None, 2)

    assert future.result(timeout=1) is True
    assert [g["chunk_indices"] for g in gathered] == [[1], [1]]
    assert [indices for indices, _ in committed[0]] == [[1], [1]]


def test_window_group_round_trip_restores_window_chunks_last_page(
    monkeypatch,
) -> None:
    """Store then retrieve restores every full-attention block, and for the
    one-page window group only the last page of its one-chunk window."""
    ctx, ictx = _inline_ctx(monkeypatch)
    stored: dict[str, Any] = {}
    monkeypatch.setattr(
        ictx,
        "commit_store",
        lambda _k, _i, chunks: stored.update(chunks=chunks) or True,
    )
    requested: list[list[int]] = []

    def _prepare_retrieve(_k, _i, expected):
        requested.append([count for _, _, count in expected])
        # The server serves the window group only its last chunk.
        (full_idx, full_chunks), (swa_idx, swa_chunks) = stored["chunks"]
        return [(full_idx, full_chunks), (swa_idx[-1:], swa_chunks[-1:])]

    monkeypatch.setattr(ictx, "prepare_retrieve_multigroup", _prepare_retrieve)
    monkeypatch.setattr(ictx, "commit_retrieve", lambda _k, _i: True)

    src = _kv(seed=1)
    dst = {name: torch.zeros_like(t) for name, t in src.items()}
    # Two chunks of two engine blocks each: [2, 4] and [3, 1].
    block_ids = [[2, 4, 3, 1], [2, 4, 3, 1]]
    assert ctx.submit_store("req", MagicMock(), src, block_ids, None, 2).result(1)
    assert ctx.submit_retrieve("req", MagicMock(), dst, block_ids, None, 2).result(1)

    # Both groups span two chunks: the paged group's count is in engine
    # blocks per chunk, not pages.
    assert requested == [[2, 2]]
    # Full attention: every engine block comes back.
    assert torch.equal(dst["full"][:, 1:], src["full"][:, 1:])
    # The window group gets chunk 1 only, and keeps its last page: engine
    # block 1 -> page 3. Every other page stays untouched.
    restored = [
        p for p in range(NUM_BLOCKS * PAGES_PER_BLOCK) if dst["swa"][:, p].any()
    ]
    assert restored == [3]
    assert torch.equal(dst["swa"][:, 3], src["swa"][:, 3])

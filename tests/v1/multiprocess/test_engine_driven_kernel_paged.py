# SPDX-License-Identifier: Apache-2.0
"""Engine-driven transfer over kernel-paged KV groups.

vLLM may page a group's tensor at a kernel block size smaller than its block
size (``kernel_block_size``): engine block ``b`` then covers tensor pages
``b*k .. b*k+k-1``. FlashInfer before Blackwell, for example, pages
``--block-size 128`` groups at 64 tokens. These tests pin that store and
retrieve move every page of each engine block, that chunk counting (null
chunks, retrieve windows) still runs on engine block IDs, and that a sliding
window of one page keeps only each chunk's last page.

Tensors are ``[2, NB, X, X, HS]`` with the two middle axes equal, so the block
size is the same whichever of NHD / HND the detector resolves.
"""

# Standard
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextResponse
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


def _inline_ctx(
    monkeypatch, kv: dict[str, torch.Tensor], group_infos: list[EngineGroupInfo]
) -> tuple[EngineDrivenTransferContext, EngineDrivenContextInline]:
    """Register ``kv`` and swap in a real inline transfer context."""
    req_client = MagicMock()
    rpc = req_client.register_kv_cache_engine_driven_context
    rpc.return_value.result.return_value = RegisterEngineDrivenContextResponse(
        shm_name="lmcache_l1_pool_x", pool_size=4096
    )
    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        mq_timeout=1.0,
        engine_group_infos=group_infos,
    )
    ictx = EngineDrivenContextInline(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = ictx
    monkeypatch.setattr(ictx, "prepare_store", lambda _k, _i: None)
    return ctx, ictx


def _round_trip(
    monkeypatch,
    src: dict[str, torch.Tensor],
    group_infos: list[EngineGroupInfo],
    src_blocks: list[list[int]],
    dst_blocks: list[list[int]],
    serve: Callable[[Any], Any] = lambda chunks: chunks,
) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    """Store ``src`` from ``src_blocks``, then retrieve into zeroed tensors at
    ``dst_blocks``; the fake server answers with ``serve(stored chunks)``.

    Returns the retrieved tensors and the per-group chunk counts requested.
    """
    ctx, ictx = _inline_ctx(monkeypatch, src, group_infos)
    stored: dict[str, Any] = {}
    requested: list[list[int]] = []

    def _commit(_k: Any, _i: int, chunks: Any) -> bool:
        stored["chunks"] = chunks
        return True

    def _prepare_retrieve(_k: Any, _i: int, expected: Any) -> Any:
        requested.append([count for _, _, count in expected])
        return serve(stored["chunks"])

    monkeypatch.setattr(ictx, "commit_store", _commit)
    monkeypatch.setattr(ictx, "prepare_retrieve_multigroup", _prepare_retrieve)
    monkeypatch.setattr(ictx, "commit_retrieve", lambda _k, _i: True)

    dst = {name: torch.zeros_like(t) for name, t in src.items()}
    assert ctx.submit_store("req", MagicMock(), src, src_blocks, None, 2).result(1)
    assert ctx.submit_retrieve("req", MagicMock(), dst, dst_blocks, None, 2).result(1)
    return dst, requested


def test_store_skips_all_null_chunks_by_engine_block(monkeypatch) -> None:
    """A chunk whose engine blocks are all null is skipped for the paged group
    too: null chunks are counted in engine block IDs, not in pages."""
    ctx, ictx = _inline_ctx(monkeypatch, _kv(), _group_infos())
    committed: list[Any] = []
    gathered: list[list[int]] = []

    def _commit(_k: Any, _i: int, chunks: Any) -> bool:
        committed.append(chunks)
        return True

    def _fake_gather(_kv, _block_ids, _blocks_in_chunk, **kwargs: Any):
        gathered.append(kwargs["chunk_indices"])
        return [torch.zeros(1) for _ in kwargs["chunk_indices"]]

    monkeypatch.setattr(ictx, "commit_store", _commit)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)

    # Chunk 0 is [0, 0] (all null) in both groups; chunk 1 is [3, 1].
    block_ids = [[0, 0, 3, 1], [0, 0, 3, 1]]
    future = ctx.submit_store("req", MagicMock(), _kv(), block_ids, None, 2)

    assert future.result(timeout=1) is True
    assert gathered == [[1], [1]]
    assert [indices for indices, _ in committed[0]] == [[1], [1]]


def test_window_group_round_trip_restores_window_chunks_last_page(
    monkeypatch,
) -> None:
    """Store then retrieve restores every full-attention block, and for the
    one-page window group only the last page of its one-chunk window."""
    src = _kv(seed=1)
    # Two chunks of two engine blocks each: [2, 4] and [3, 1].
    block_ids = [[2, 4, 3, 1], [2, 4, 3, 1]]

    def _serve(chunks: Any) -> Any:
        # The server serves the window group only its last chunk.
        (full_idx, full_chunks), (swa_idx, swa_chunks) = chunks
        return [(full_idx, full_chunks), (swa_idx[-1:], swa_chunks[-1:])]

    dst, requested = _round_trip(
        monkeypatch, src, _group_infos(), block_ids, block_ids, _serve
    )

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


def test_single_group_round_trip_moves_every_page(monkeypatch) -> None:
    """A single kernel-paged group is not read as compressed: retrieving into
    other engine blocks moves both pages of each block, and pages outside
    the request stay untouched."""
    src = {"kv": _kv(seed=1)["swa"]}
    src_blocks, dst_blocks = [2, 4, 3, 1], [1, 3, 4, 2]

    dst, requested = _round_trip(
        monkeypatch, src, _group_infos()[:1], [src_blocks], [dst_blocks]
    )

    assert requested == [[2]]
    k = PAGES_PER_BLOCK
    for s, d in zip(src_blocks, dst_blocks, strict=True):
        assert torch.equal(
            dst["kv"][:, d * k : (d + 1) * k], src["kv"][:, s * k : (s + 1) * k]
        )
    assert not dst["kv"][:, :k].any()

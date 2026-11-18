# SPDX-License-Identifier: Apache-2.0
"""Sparse recurrent-state stores (``lmcache.mp.sparse_state_every_n_chunks``).

The STORE keeps a recurrent object group's full chunk only at multiples of N
chunks (absolute) and at the prompt's last full chunk boundary. The
window-aware lookup must then serve a prefix exactly at a kept boundary, fall
back to the nearest earlier kept boundary, or miss -- never end a hit at a
chunk without its recurrent state.
"""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.common.sparse_state import (
    PROMPT_TOKENS_HINT,
    SPARSE_STATE_HINT,
    keeps_windowed_chunk,
)
from lmcache.v1.distributed.api import FULL_ATTENTION_WINDOW_CHUNKS
from lmcache.v1.distributed.bitmap_ops import fold_unfold_grouped
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    sparse_recurrent_chunk_masks,
)

C = 9216
# One recurrent object group, a full-attention group, a sliding-window group.
GROUPS = [
    ObjectGroupInfo(kernel_group_indices=[0, 1, 2], sw_size_chunks=1, recurrent=True),
    ObjectGroupInfo(kernel_group_indices=[3, 4]),
    ObjectGroupInfo(kernel_group_indices=[5], sw_size_chunks=1),
    ObjectGroupInfo(kernel_group_indices=[6], sw_size_chunks=1, aux=True),
]


def _key(start: int, end: int, configs: dict | None) -> Any:
    return SimpleNamespace(start=start, end=end, request_configs=configs)


def _row0(start: int, end: int, cfg: dict) -> list[bool]:
    mask = sparse_recurrent_chunk_masks(_key(start, end, cfg), GROUPS, 1, C)
    assert mask is not None
    return mask[0]


def _kept(mask_row: list[bool]) -> list[int]:
    return [i for i, skip in enumerate(mask_row) if not skip]


def test_no_hint_no_mask() -> None:
    assert sparse_recurrent_chunk_masks(_key(0, 10 * C, None), GROUPS, 10, C) is None
    assert (
        sparse_recurrent_chunk_masks(
            _key(0, 10 * C, {SPARSE_STATE_HINT: 0}), GROUPS, 10, C
        )
        is None
    )


def test_every_n_and_prompt_end() -> None:
    # 95,000-token prompt: last full chunk ends at 10 * C.
    cfg = {SPARSE_STATE_HINT: 4, PROMPT_TOKENS_HINT: 95_000}
    mask = sparse_recurrent_chunk_masks(_key(0, 10 * C, cfg), GROUPS, 10, C)
    assert mask is not None
    assert _kept(mask[0]) == [3, 7, 9]  # ends 4C, 8C, prompt end 10C
    assert _kept(mask[1]) == list(range(10))  # attention: every chunk
    assert _kept(mask[2]) == [3, 7, 9]  # draft SWA (1-chunk window): same rule
    assert _kept(mask[3]) == list(range(10))  # aux: never masked


def test_window_reach() -> None:
    # A 2-chunk window keeps the chunk before each kept boundary too.
    kept = [i for i in range(10) if keeps_windowed_chunk((i + 1) * C, C, 4, 95_000, 2)]
    assert kept == [2, 3, 6, 7, 8, 9]
    assert all(keeps_windowed_chunk((i + 1) * C, C, 0, 0) for i in range(10))
    # Prompt shorter than one chunk: only multiples of N.
    assert not keeps_windowed_chunk(C, C, 4, 5000)


def test_absolute_chunk_end_across_steps() -> None:
    # Chunked prefill stores one chunk per step; boundaries follow the absolute
    # chunk end, so a STORE of chunk 3 alone keeps it and one of chunk 4 skips it.
    cfg = {SPARSE_STATE_HINT: 4, PROMPT_TOKENS_HINT: 95_000}
    assert _kept(_row0(3 * C, 4 * C, cfg)) == [0]
    assert _kept(_row0(4 * C, 5 * C, cfg)) == []
    # Prompt end inside a later step's STORE.
    assert _kept(_row0(9 * C, 10 * C, cfg)) == [0]


def _rows(num_chunks: int, present: list[list[int]]) -> list[Bitmap]:
    rows = []
    for idx in present:
        b = Bitmap(num_chunks)
        for i in idx:
            b.set(i)
        rows.append(b)
    return rows


WINDOWS = [1, FULL_ATTENTION_WINDOW_CHUNKS, 1]


@pytest.mark.parametrize(
    "kda, attn, expected_hit",
    [
        ([3, 7, 9], list(range(10)), 10),  # hit at the stored prompt end
        ([3, 7], list(range(10)), 8),  # prompt-end state evicted: nearest kept (8C)
        ([3, 7, 9], list(range(6)), 4),  # attention ends at 6C: fall back to 4C
        ([7, 9], list(range(6)), 0),  # no kept state inside the attention prefix: miss
        ([], list(range(10)), 0),
    ],
)
def test_lookup_lands_on_a_kept_state(kda, attn, expected_hit) -> None:
    swa = list(range(10))
    hit, retain = fold_unfold_grouped(_rows(10, [kda, attn, swa]), WINDOWS)
    assert hit == expected_hit
    if hit:
        # The hit's last chunk carries the recurrent state, and the retrieve
        # reads only that chunk of the recurrent group.
        assert hit - 1 in kda
        assert retain[0].get_indices_list() == [hit - 1]
        assert retain[1].get_indices_list() == list(range(hit))


def test_fence_follows_the_mask() -> None:
    """wait_for_save fences exactly the STOREs that read windowed blocks."""
    connector_mod = pytest.importorskip("lmcache.integration.vllm.lmcache_mp_connector")
    conn = object.__new__(connector_mod.LMCacheMPConnector)
    conn._has_windowed_groups = True
    conn.worker_adapter = SimpleNamespace(
        lmcache_tokens_per_chunk=C,
        engine_group_infos=[
            SimpleNamespace(sw_size_tokens=-1),
            SimpleNamespace(sw_size_tokens=1152),
            SimpleNamespace(sw_size_tokens=2048),
        ],
    )
    cfg = {SPARSE_STATE_HINT: 4, PROMPT_TOKENS_HINT: 95_000}

    def fence(start, end, configs=cfg):
        meta = SimpleNamespace(
            request_configs=configs, op=SimpleNamespace(start=start, end=end)
        )
        return connector_mod.LMCacheMPConnector._store_reads_windowed_blocks(conn, meta)

    assert fence(3 * C, 4 * C)  # chunk ending at 4C is kept
    assert not fence(4 * C, 5 * C)
    assert fence(4 * C, 6 * C + 0) is False
    assert fence(4 * C, 8 * C)  # range contains 8C
    assert fence(9 * C, 10 * C)  # prompt end
    assert fence(10 * C, 10 * C + 500)  # unfull tail: always
    assert fence(4 * C, 5 * C, configs=None)  # option off: every STORE
    conn._has_windowed_groups = False
    assert not fence(3 * C, 4 * C)


def test_pins_follow_the_mask() -> None:
    """With sparse stores the scheduler pins windowed blocks of kept chunks only."""
    connector_mod = pytest.importorskip("lmcache.integration.vllm.lmcache_mp_connector")
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    conn = connector_mod.LMCacheMPConnector.__new__(connector_mod.LMCacheMPConnector)
    conn._pin_windowed_stores = True
    pool = MagicMock()
    pool.blocks = {i: f"block-{i}" for i in range(100)}
    conn._gpu_block_pool = pool
    # Chunk 8 tokens; group 0 full attention, group 1 Mamba (2-token blocks).
    conn.scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=8)
    conn._group_tokens_per_block = [8, 2]
    conn._windowed_group_tokens = {1: 2}
    conn._store_pins = {}
    conn._store_pin_seq = 0
    conn._sched_step = 0
    conn._store_pin_steps = {}
    # Chunks 1..4 (ends 8..32); N = 2 keeps ends 16, 32; prompt 30 -> end 24 kept.
    r_meta = LMCacheMPRequestMetadata(
        request_id="req-0",
        direction="STORE",
        op=LoadStoreOp(
            token_bytes=b"",
            block_ids=[[1, 2, 3, 4], list(range(10, 26))],
            start=0,
            end=32,
        ),
        request_configs={SPARSE_STATE_HINT: 2, PROMPT_TOKENS_HINT: 30},
    )
    conn._pin_store_blocks(r_meta)
    # Last state block of chunks 2, 3, 4 (ends 16, 24, 32); not chunk 1.
    pool.touch.assert_called_once_with([f"block-{i}" for i in (17, 21, 25)])

# SPDX-License-Identifier: Apache-2.0
"""Sparse recurrent-state stores (``lmcache.mp.sparse_state_every_n_chunks``).

The engine connector keeps a full chunk's recurrent state only when at least
N chunks passed since the request's last kept state or cache hit, or at the
prompt's last full chunk boundary, and names the kept chunk ends in the
STORE's ``lmcache.hint.sparse_state_keep``. The server skips the windowed
objects of the other chunks. The window-aware lookup must then serve a prefix
exactly at a kept end, fall back to the nearest earlier kept end, or miss --
never end a hit at a chunk without its recurrent state.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.common.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.common.sparse_state import (
    SPARSE_STATE_KEEP_HINT,
    encode_kept_chunk_ends,
    keeps_state_at,
    kept_chunk_ends,
    merge_sparse_state_keep,
)
from lmcache.v1.distributed.api import FULL_ATTENTION_WINDOW_CHUNKS
from lmcache.v1.distributed.bitmap_ops import fold_unfold_grouped
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    sparse_recurrent_chunk_masks,
)

C = 9216
# A recurrent group, full attention, a one-chunk sliding window, an aux
# group and a two-chunk sliding window.
GROUPS = [
    ObjectGroupInfo(kernel_group_indices=[0, 1, 2], sw_size_chunks=1, recurrent=True),
    ObjectGroupInfo(kernel_group_indices=[3, 4]),
    ObjectGroupInfo(kernel_group_indices=[5], sw_size_chunks=1),
    ObjectGroupInfo(kernel_group_indices=[6], sw_size_chunks=1, aux=True),
    ObjectGroupInfo(kernel_group_indices=[7], sw_size_chunks=2),
]


def _key(start: int, end: int, configs: dict | None) -> Any:
    return SimpleNamespace(start=start, end=end, request_configs=configs)


def _kept(mask_row: list[bool]) -> list[int]:
    return [i for i, skip in enumerate(mask_row) if not skip]


# ------------------------------------------------------------------ #
#  Keep rule and hint                                                  #
# ------------------------------------------------------------------ #


def test_keep_rule_counts_chunks_since_the_last_kept_state() -> None:
    # N = 4: three chunks after the last kept state is too early, four is not.
    assert not keeps_state_at(7 * C, 4 * C, C, 4, 200_000)
    assert keeps_state_at(8 * C, 4 * C, C, 4, 200_000)
    assert keeps_state_at(11 * C, 4 * C, C, 4, 200_000)
    # From the start of the request, or from a cache hit at 4C.
    assert not keeps_state_at(3 * C, 0, C, 4, 200_000)
    assert keeps_state_at(4 * C, 0, C, 4, 200_000)


def test_keep_rule_keeps_the_prompts_last_full_chunk() -> None:
    # 95,000-token prompt: its last full chunk ends at 10C.
    assert keeps_state_at(10 * C, 8 * C, C, 8, 95_000)
    assert not keeps_state_at(9 * C, 8 * C, C, 8, 95_000)
    # Unknown prompt length: only the distance counts.
    assert not keeps_state_at(10 * C, 8 * C, C, 8, 0)


def test_keep_rule_keeps_every_state_when_off() -> None:
    assert all(keeps_state_at(end * C, 0, C, 0, 95_000) for end in range(1, 11))


def test_hint_round_trip() -> None:
    assert kept_chunk_ends(None) is None
    assert kept_chunk_ends({"lmcache.tag.x": "y"}) is None
    assert kept_chunk_ends({SPARSE_STATE_KEEP_HINT: ""}) == frozenset()
    value = encode_kept_chunk_ends([8 * C, 4 * C, 8 * C])
    assert value == f"{4 * C},{8 * C}"
    assert kept_chunk_ends({SPARSE_STATE_KEEP_HINT: value}) == {4 * C, 8 * C}


def test_coalesced_stores_keep_every_stores_ends() -> None:
    first = {"lmcache.tag.x": "y", SPARSE_STATE_KEEP_HINT: ""}
    merged = merge_sparse_state_keep(
        [first, {SPARSE_STATE_KEEP_HINT: str(8 * C)}, {SPARSE_STATE_KEEP_HINT: ""}]
    )
    assert merged == {"lmcache.tag.x": "y", SPARSE_STATE_KEEP_HINT: str(8 * C)}
    assert first[SPARSE_STATE_KEEP_HINT] == ""
    # Without the hint the first STORE's configs pass through unchanged.
    plain = {"lmcache.tag.x": "y"}
    assert merge_sparse_state_keep([plain, None]) is plain
    assert merge_sparse_state_keep([None, None]) is None


def test_lazy_offload_coalescing_merges_the_hint() -> None:
    pytest.importorskip("vllm", reason="the lazy offload manager imports vLLM")
    # First Party
    from lmcache.integration.vllm.lazy_offload_manager import (
        _coalesce_store_metadata,
    )
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    def store(start: int, end: int, kept: list[int]) -> LMCacheMPRequestMetadata:
        return LMCacheMPRequestMetadata(
            request_id="req-0",
            direction="STORE",
            op=LoadStoreOp(
                token_bytes=b"",
                block_ids=[list(range(start // C, end // C))],
                start=start,
                end=end,
            ),
            request_configs={SPARSE_STATE_KEEP_HINT: encode_kept_chunk_ends(kept)},
        )

    merged = _coalesce_store_metadata(
        [store(0, 7 * C, []), store(7 * C, 8 * C, [8 * C]), store(8 * C, 9 * C, [])]
    )

    assert (merged.op.start, merged.op.end) == (0, 9 * C)
    assert kept_chunk_ends(merged.request_configs) == {8 * C}


# ------------------------------------------------------------------ #
#  Server mask                                                         #
# ------------------------------------------------------------------ #


def test_no_hint_no_mask() -> None:
    assert sparse_recurrent_chunk_masks(_key(0, 10 * C, None), GROUPS, 10, C) is None
    assert (
        sparse_recurrent_chunk_masks(
            _key(0, 10 * C, {"lmcache.tag.x": "y"}), GROUPS, 10, C
        )
        is None
    )


def test_mask_keeps_only_the_hinted_chunk_ends() -> None:
    cfg = {SPARSE_STATE_KEEP_HINT: encode_kept_chunk_ends([4 * C, 8 * C, 10 * C])}
    mask = sparse_recurrent_chunk_masks(_key(0, 10 * C, cfg), GROUPS, 10, C)
    assert mask is not None
    assert _kept(mask[0]) == [3, 7, 9]  # recurrent state
    assert _kept(mask[1]) == list(range(10))  # attention: every chunk
    assert _kept(mask[2]) == [3, 7, 9]  # draft SWA (1-chunk window): same ends
    assert _kept(mask[3]) == list(range(10))  # aux: never masked
    # A longer window is read below a kept end too: every chunk is stored.
    assert _kept(mask[4]) == list(range(10))


def test_mask_uses_absolute_chunk_ends() -> None:
    # Chunked prefill stores a few chunks per STORE; the hint names absolute
    # token positions, so a STORE starting at 6C keeps its second chunk.
    cfg = {SPARSE_STATE_KEEP_HINT: str(8 * C)}
    mask = sparse_recurrent_chunk_masks(_key(6 * C, 9 * C, cfg), GROUPS, 3, C)
    assert mask is not None
    assert _kept(mask[0]) == [1]


def test_empty_hint_keeps_no_state() -> None:
    cfg = {SPARSE_STATE_KEEP_HINT: ""}
    mask = sparse_recurrent_chunk_masks(_key(0, 3 * C, cfg), GROUPS, 3, C)
    assert mask is not None
    assert _kept(mask[0]) == []
    assert _kept(mask[1]) == [0, 1, 2]


# ------------------------------------------------------------------ #
#  Lookup                                                              #
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
#  Connector: hint and pins                                            #
# ------------------------------------------------------------------ #


def _scheduler(every_n: int) -> Any:
    """A scheduler connector over 8-token chunks: group 0 full attention,
    group 1 Mamba (2-token blocks), group 2 a 16-token sliding window."""
    connector_mod = pytest.importorskip("lmcache.integration.vllm.lmcache_mp_connector")
    conn = connector_mod.LMCacheMPConnector.__new__(connector_mod.LMCacheMPConnector)
    conn._sparse_state_every_n = every_n
    conn.scheduler_adapter = MagicMock(lmcache_tokens_per_chunk=8)
    conn._group_tokens_per_block = [8, 2, 8]
    conn._windowed_group_tokens = {1: 2, 2: 16}
    pool = MagicMock()
    pool.blocks = {i: f"block-{i}" for i in range(100)}
    conn._gpu_block_pool = pool
    conn._store_pins = {}
    conn._store_pin_seq = 0
    conn._warned_unpinned_stores = False
    return conn


def _store(start: int, end: int, state_blocks: list[int]) -> Any:
    pytest.importorskip("vllm", reason="the connector metadata imports vLLM")
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )
    from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp

    chunks = (end - start) // 8
    return LMCacheMPRequestMetadata(
        request_id="req-0",
        direction="STORE",
        op=LoadStoreOp(
            token_bytes=b"",
            block_ids=[
                list(range(1, chunks + 1)),
                state_blocks,
                list(range(50, 50 + chunks)),
            ],
            start=start,
            end=end,
        ),
        request_configs={"lmcache.tag.x": "y"},
    )


def _tracker(hit_tokens: int = 0, prompt_tokens: int = 30) -> Any:
    return SimpleNamespace(
        last_kept_state_end=None,
        num_lmcache_hit_tokens=hit_tokens,
        num_prompt_tokens=prompt_tokens,
    )


def test_hint_names_the_kept_ends_among_the_chunks_that_hold_a_state() -> None:
    conn = _scheduler(every_n=2)
    tracker = _tracker()
    # Chunks ending at 8, 16, 24 and 32; the one ending at 16 holds no state
    # (its last state block is null). Prompt 30: last full chunk ends at 24.
    r_meta = _store(0, 32, [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 25])

    conn._add_sparse_state_hint(r_meta, tracker)

    # 8 is one chunk after the start; 16 holds no state; 24 is two chunks
    # after the start and the prompt end; 32 is one chunk after 24.
    assert kept_chunk_ends(r_meta.request_configs) == {24}
    assert r_meta.request_configs["lmcache.tag.x"] == "y"
    assert tracker.last_kept_state_end == 24


def test_hint_counts_from_the_cache_hit_and_across_stores() -> None:
    conn = _scheduler(every_n=2)
    # LMCache hit of 20 tokens: the state at 16 is in the cache.
    tracker = _tracker(hit_tokens=20, prompt_tokens=100)

    first = _store(16, 32, [0, 0, 0, 13, 0, 0, 0, 17])
    conn._add_sparse_state_hint(first, tracker)
    assert kept_chunk_ends(first.request_configs) == {32}

    second = _store(32, 48, [0, 0, 0, 21, 0, 0, 0, 25])
    conn._add_sparse_state_hint(second, tracker)
    assert kept_chunk_ends(second.request_configs) == {48}
    assert tracker.last_kept_state_end == 48


def test_no_hint_when_the_option_is_off_or_no_group_has_a_short_window() -> None:
    r_meta = _store(0, 16, [0, 0, 0, 13, 0, 0, 0, 17])
    _scheduler(every_n=0)._add_sparse_state_hint(r_meta, _tracker())
    assert SPARSE_STATE_KEEP_HINT not in r_meta.request_configs

    conn = _scheduler(every_n=2)
    conn._windowed_group_tokens = {2: 16}
    conn._add_sparse_state_hint(r_meta, _tracker())
    assert SPARSE_STATE_KEEP_HINT not in r_meta.request_configs


def test_pins_follow_the_hint() -> None:
    """The scheduler pins the state blocks of kept chunks only, and the
    longer sliding window of every chunk the STORE reads."""
    conn = _scheduler(every_n=2)
    r_meta = _store(0, 32, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + [0] * 4)
    r_meta.request_configs = {SPARSE_STATE_KEEP_HINT: "16,24"}

    conn._pin_store_blocks(r_meta)

    conn._gpu_block_pool.touch.assert_called_once_with(
        [f"block-{i}" for i in (17, 21, 50, 51, 52, 53)]
    )


def test_pins_cover_every_chunk_without_the_hint() -> None:
    conn = _scheduler(every_n=0)
    r_meta = _store(0, 16, [10, 11, 12, 13, 14, 15, 16, 17])

    conn._pin_store_blocks(r_meta)

    conn._gpu_block_pool.touch.assert_called_once_with(
        [f"block-{i}" for i in (13, 17, 50, 51)]
    )

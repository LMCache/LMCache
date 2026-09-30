# SPDX-License-Identifier: Apache-2.0
"""``_sparse_classify`` must not strike chunks that L1 could not stage.

A chunk that misses because the L1 reservation hit OUT_OF_MEMORY is still in
storage. Striking it evicts a live entry from the fingerprint table, and
nothing re-registers it, so the reuse loss is permanent.
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.modules.blend.lookup import LookupMixin

_CHUNK = 256


class _Classifier(LookupMixin):
    """Minimal host for the mixin: only what ``_sparse_classify`` touches."""

    UNRETRIEVED_KEYS_EXTRA = "unretrieved"

    def __init__(self):
        self._pending_fp_hashes = set()
        self._pending_fp_lock = threading.Lock()
        self._stale_strike = {}
        self._STALE_STRIKE_THRESHOLD = 2
        self._token_range_matcher = MagicMock()
        self._event_bus = MagicMock()
        self._ctx = MagicMock()
        self._ctx.session_manager.get_or_create.return_value.extras = {}


def _key(request_id="req-1"):
    key = MagicMock()
    key.request_id = request_id
    key.require_num_kv_readers.return_value = 1
    return key


def _match(h: bytes) -> CBMatchResult:
    return CBMatchResult(old_st=0, old_ed=_CHUNK, cur_st=0, cur_ed=_CHUNK, hash=h)


def _rows(num_cols: int, set_cols: list[int]) -> list[Bitmap]:
    row = Bitmap(num_cols)
    for c in set_cols:
        row.set(c)
    return [row]


def test_capacity_miss_does_not_strike_or_evict():
    """L1 had no room: the chunk is dropped for this request but kept."""
    c = _Classifier()
    match = _match(b"h0")
    found = _rows(1, [])  # nothing landed
    capacity = _rows(1, [0])  # ...because of OOM

    out = c._sparse_classify(
        _key(), [match], found, {b"h0": ["k"]}, {b"h0": 0}, capacity
    )

    assert out == []
    assert c._stale_strike == {}, "capacity miss must not accrue a strike"
    c._token_range_matcher.remove_chunks.assert_not_called()


def test_capacity_miss_survives_repeated_rounds():
    """The strike threshold is 2; a capacity miss must never reach it."""
    c = _Classifier()
    match = _match(b"h0")
    for _ in range(5):
        c._sparse_classify(
            _key(), [match], _rows(1, []), {b"h0": ["k"]}, {b"h0": 0}, _rows(1, [0])
        )
    assert c._stale_strike == {}
    c._token_range_matcher.remove_chunks.assert_not_called()


def test_genuine_miss_still_strikes_and_evicts():
    """No capacity attribution: absent from storage, existing behavior holds."""
    c = _Classifier()
    match = _match(b"h0")
    for _ in range(2):  # threshold is 2
        c._sparse_classify(
            _key(), [match], _rows(1, []), {b"h0": ["k"]}, {b"h0": 0}, _rows(1, [])
        )
    c._token_range_matcher.remove_chunks.assert_called_once_with([b"h0"])


def test_omitted_capacity_rows_preserve_legacy_behavior():
    """Callers that pass no capacity rows keep striking as before."""
    c = _Classifier()
    match = _match(b"h0")
    for _ in range(2):
        c._sparse_classify(_key(), [match], _rows(1, []), {b"h0": ["k"]}, {b"h0": 0})
    c._token_range_matcher.remove_chunks.assert_called_once_with([b"h0"])


def test_found_chunk_unaffected_by_capacity_rows():
    """A chunk that landed is returned even if its bit is set in capacity."""
    c = _Classifier()
    match = _match(b"h0")
    out = c._sparse_classify(
        _key(), [match], _rows(1, [0]), {b"h0": ["k"]}, {b"h0": 0}, _rows(1, [0])
    )
    assert out == [match]
    c._token_range_matcher.remove_chunks.assert_not_called()


def test_mixed_batch_splits_capacity_from_stale():
    """One chunk landed, one OOM'd, one genuinely absent."""
    c = _Classifier()
    landed, oom, absent = _match(b"h0"), _match(b"h1"), _match(b"h2")
    hash_to_col = {b"h0": 0, b"h1": 1, b"h2": 2}
    obj_keys = {b"h0": ["k0"], b"h1": ["k1"], b"h2": ["k2"]}

    out = c._sparse_classify(
        _key(),
        [landed, oom, absent],
        _rows(3, [0]),
        obj_keys,
        hash_to_col,
        _rows(3, [1]),
    )

    assert out == [landed]
    assert set(c._stale_strike) == {b"h2"}, "only the absent chunk is struck"


@pytest.mark.parametrize("pending", [True, False])
def test_in_flight_guard_still_applies(pending):
    """The pre-existing in-flight guard is independent of capacity misses."""
    c = _Classifier()
    match = _match(b"h0")
    if pending:
        c._pending_fp_hashes.add(b"h0")
    for _ in range(2):
        c._sparse_classify(
            _key(), [match], _rows(1, []), {b"h0": ["k"]}, {b"h0": 0}, _rows(1, [])
        )
    if pending:
        c._token_range_matcher.remove_chunks.assert_not_called()
    else:
        c._token_range_matcher.remove_chunks.assert_called_once_with([b"h0"])

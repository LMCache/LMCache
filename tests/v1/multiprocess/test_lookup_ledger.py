# SPDX-License-Identifier: Apache-2.0
"""Tests for LookupLockLedger, the per-request read-lock ownership record."""

# Standard
from unittest.mock import MagicMock
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.lookup_ledger import LookupLockLedger

_OWNER: dict[ObjectKey, int] = {MagicMock(spec=ObjectKey): 1}


def _lookup_key(request_id: str = "req-1", start: int = 0, end: int = 8):
    """Build a scheduler-style (worker_id=None) lookup key."""
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=1,
        worker_id=None,
        token_ids=list(range(16)),
        start=start,
        end=end,
        request_id=request_id,
    )


def _worker_key(worker_id: int = 0, start: int = 0, end: int = 8):
    """Build a worker-specific key inside the lookup's range."""
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=1,
        worker_id=worker_id,
        token_ids=list(range(16)),
        start=start,
        end=end,
        request_id="req-1",
    )


class TestLookupLockLedgerLifetime:
    def test_ensure_records_key_only_once(self) -> None:
        """ensure() must not reset a lookup that already recorded a hit."""
        ledger = LookupLockLedger()
        ledger.ensure("req-1", _lookup_key())
        ledger.begin_lookup("req-1", _lookup_key(start=0, end=16), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,))

        ledger.ensure("req-1", _lookup_key())

        state = ledger.lock_state("req-1")
        assert state is not None
        assert state.hit_chunks == 2

    def test_begin_lookup_resets_the_previous_result(self) -> None:
        """A new lookup invalidates the previous hit length and generation."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 3, (0,), _OWNER)

        ledger.begin_lookup("req-1", _lookup_key(start=0, end=16), (2,))

        state = ledger.lock_state("req-1")
        assert state is not None
        assert state.hit_chunks == -1
        assert state.locked_gids == ()
        assert state.group_windows == (2,)
        assert state.owners == {}
        assert state.generation == 2

    def test_lock_state_is_a_snapshot(self) -> None:
        """Mutating a returned state must not corrupt the ledger."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,), _OWNER)

        state = ledger.lock_state("req-1")
        assert state is not None
        state.owners[MagicMock(spec=ObjectKey)] = 9

        assert ledger.get_prefetch_owners("req-1") == _OWNER

    def test_release_drops_the_record(self) -> None:
        """release() returns the record and removes it from the ledger."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))

        released = ledger.release("req-1")

        assert released is not None
        assert released.generation == 1
        assert ledger.lock_state("req-1") is None
        assert ledger.release("req-1") is None

    def test_unknown_request_has_no_owners(self) -> None:
        """An unrecorded request owns nothing, so no lock may be derived."""
        ledger = LookupLockLedger()
        assert ledger.lock_state("never-looked-up") is None
        assert ledger.get_prefetch_owners("never-looked-up") == {}

    def test_concurrent_begin_and_record(self) -> None:
        """Concurrent lookups must not lose or corrupt a recorded hit."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))

        def worker(hit: int) -> None:
            ledger.record_prefetch_result("req-1", hit, (0,))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        state = ledger.lock_state("req-1")
        assert state is not None
        assert state.hit_chunks in range(8)


class TestFailedRetrieveRelease:
    def _prepared(self, ledger: LookupLockLedger) -> int:
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,))
        prepared = ledger.prepare_failed_retrieve_release("req-1", _worker_key())
        assert prepared is not None
        return prepared[3]

    def test_prepare_returns_the_lock_model(self) -> None:
        """prepare() reports the hit, locked groups, windows and generation."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,), _OWNER)

        prepared = ledger.prepare_failed_retrieve_release("req-1", _worker_key())

        assert prepared == (2, (0,), (-1,), 1)
        assert ledger.get_prefetch_owners("req-1") == _OWNER

    def test_prepare_rejects_a_scheduler_key(self) -> None:
        """Only a worker's own share is released, never the whole reservation."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,))

        assert ledger.prepare_failed_retrieve_release("req-1", _lookup_key()) is None

    def test_prepare_rejects_a_range_outside_the_lookup(self) -> None:
        """A key the lookup never locked must not be derived into a release."""
        ledger = LookupLockLedger()
        ledger.begin_lookup("req-1", _lookup_key(start=0, end=8), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,))

        assert (
            ledger.prepare_failed_retrieve_release("req-1", _worker_key(end=16)) is None
        )

    def test_claim_is_exactly_once_per_instance(self) -> None:
        """A duplicate failed response must not release the same share twice."""
        ledger = LookupLockLedger()
        generation = self._prepared(ledger)
        key = _worker_key()

        assert ledger.claim_failed_retrieve_release("req-1", 7, key, generation)
        assert not ledger.claim_failed_retrieve_release("req-1", 7, key, generation)
        # A different instance still owns its own share.
        assert ledger.claim_failed_retrieve_release("req-1", 8, key, generation)

    def test_claim_rejects_a_stale_generation(self) -> None:
        """A release prepared before a new lookup must not be claimable."""
        ledger = LookupLockLedger()
        generation = self._prepared(ledger)

        ledger.begin_lookup("req-1", _lookup_key(), (-1,))
        ledger.record_prefetch_result("req-1", 2, (0,))

        assert not ledger.claim_failed_retrieve_release(
            "req-1", 7, _worker_key(), generation
        )

    def test_claim_ignores_a_request_without_a_lookup(self) -> None:
        """No ledger entry means no provable ownership, so nothing is claimed."""
        ledger = LookupLockLedger()
        assert not ledger.claim_failed_retrieve_release("req-1", 7, _worker_key(), 1)

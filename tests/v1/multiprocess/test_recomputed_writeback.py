# SPDX-License-Identifier: Apache-2.0
"""Contract tests for independent failed-range persistence tracking."""

# Third Party
# First Party
from lmcache.v1.common.recomputed_writeback import RecomputedWriteback


def test_only_missing_committed_ranges_are_eligible() -> None:
    ledger = RecomputedWriteback()
    ledger.observe(128, 256)
    assert ledger.ready(255) == []
    assert ledger.ready(8064) == [(128, 256)]
    ticket = ledger.begin(128, 256)
    assert ledger.ready(8064) == []
    assert ledger.complete(ticket, True)
    assert ledger.ready(8064) == []


def test_failure_is_retryable_and_old_success_cannot_clear_new_loss() -> None:
    ledger = RecomputedWriteback()
    ledger.observe(128, 256)
    old = ledger.begin(128, 256)
    ledger.observe(128, 256)
    assert ledger.complete(old, True)
    assert ledger.ready(8064) == [(128, 256)]
    current = ledger.begin(128, 256)
    assert not ledger.complete(old, True)
    assert ledger.complete(current, False)
    assert ledger.ready(8064) == [(128, 256)]
    retry = ledger.begin(128, 256)
    assert retry.identity != current.identity
    assert ledger.complete(retry, True)
    assert ledger.ready(8064) == []

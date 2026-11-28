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


def test_groups_keep_independent_ranges_and_completion_revisions() -> None:
    ledger = RecomputedWriteback()
    ledger.observe(128, 256, group_id=0)
    ledger.observe(128, 256, group_id=2)
    assert ledger.group_ids == [0, 2]
    first = ledger.begin(128, 256, group_id=0)
    second = ledger.begin(128, 256, group_id=2)
    assert first.group_id == 0 and second.group_id == 2
    assert ledger.complete(first, True)
    assert ledger.complete(second, False)
    assert ledger.group_ids == [2]
    assert ledger.ready(256, group_id=0) == []
    assert ledger.ready(256, group_id=2) == [(128, 256)]

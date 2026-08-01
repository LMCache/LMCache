# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed cluster view used by KV cache orchestration (#4291).

The event stream the view folds is lossy, so these tests pin the difference
between state that heals after a dropped event and state that does not.
"""

# Standard

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.orchestration.view import (
    FoldKind,
    View,
    ViewFieldError,
)


def test_convergent_field_stays_trusted_after_drops() -> None:
    """A dropped event only leaves a stale value, which the next one fixes."""
    view = View()
    accessed = view.declare("accessed_at", FoldKind.CONVERGENT)
    accessed.set("key-a", 100.0)

    view.observe_dropped_events(42)

    assert accessed.trusted
    assert not view.degraded


def test_accumulative_field_loses_trust_after_a_drop() -> None:
    """A dropped delta is a permanent offset, so the total is not usable."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)
    used_bytes.add("tenant-a", 4096.0)

    assert used_bytes.trusted

    view.observe_dropped_events(1)

    assert not used_bytes.trusted
    assert view.degraded


def test_reconcile_with_absolute_values_restores_trust() -> None:
    """Absolute values from an authoritative source clear the drift."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)
    used_bytes.add("tenant-a", 4096.0)
    view.observe_dropped_events(3)

    used_bytes.reconcile({"tenant-a": 2048.0})

    assert used_bytes.trusted
    assert not view.degraded
    assert used_bytes.get("tenant-a") == 2048.0


def test_reconcile_drops_keys_absent_from_the_snapshot() -> None:
    """Reconciliation replaces the contents rather than merging into them."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)
    used_bytes.add("tenant-a", 4096.0)
    used_bytes.add("tenant-b", 8192.0)

    used_bytes.reconcile({"tenant-a": 4096.0})

    assert used_bytes.get("tenant-b") == 0.0


def test_trigger_over_convergent_state_stays_actionable_while_degraded() -> None:
    """An LRU style policy is safe to run even after events were dropped."""
    view = View()
    accessed = view.declare("accessed_at", FoldKind.CONVERGENT)
    view.declare("used_bytes", FoldKind.ACCUMULATIVE).add("tenant-a", 1.0)
    accessed.set("key-a", 10.0)
    view.observe_dropped_events(5)

    verdict = view.evaluate(lambda v: v.field("accessed_at").get("key-a") < 50.0)

    assert verdict.fired
    assert verdict.untrusted_fields == ()
    assert verdict.actionable


def test_trigger_over_accumulative_state_is_flagged_while_degraded() -> None:
    """A quota style policy must not delete keys off a drifted total."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)
    used_bytes.add("tenant-a", 9000.0)
    view.observe_dropped_events(1)

    verdict = view.evaluate(lambda v: v.field("used_bytes").get("tenant-a") > 8192.0)

    assert verdict.fired
    assert verdict.untrusted_fields == ("used_bytes",)
    assert not verdict.actionable


def test_a_dropped_decrement_is_what_makes_the_quota_trigger_wrong() -> None:
    """The failure this guard exists for, written out end to end."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)

    used_bytes.add("tenant-a", 8192.0)  # key entered L1
    view.observe_dropped_events(1)  # the matching eviction event was dropped

    over_quota = view.evaluate(
        lambda v: v.field("used_bytes").get("tenant-a") > 4096.0
    )

    # The total still says the tenant is over quota and the action attached to
    # this trigger deletes their keys, so firing here would delete a cache the
    # tenant is no longer filling.
    assert over_quota.fired
    assert not over_quota.actionable

    used_bytes.reconcile({"tenant-a": 0.0})
    after = view.evaluate(lambda v: v.field("used_bytes").get("tenant-a") > 4096.0)

    assert not after.fired
    assert after.trustworthy
    assert not after.actionable


def test_each_untrusted_field_is_named_once() -> None:
    """Repeated reads of the same field do not repeat in the verdict."""
    view = View()
    view.declare("used_bytes", FoldKind.ACCUMULATIVE).add("tenant-a", 1.0)
    view.observe_dropped_events(1)

    def trigger(v: View) -> bool:
        left = v.field("used_bytes").get("tenant-a")
        right = v.field("used_bytes").get("tenant-b")
        return left > right

    verdict = view.evaluate(trigger)

    assert verdict.untrusted_fields == ("used_bytes",)


def test_assignment_on_an_accumulative_field_is_rejected() -> None:
    """Assigning would silently discard the running total."""
    view = View()
    used_bytes = view.declare("used_bytes", FoldKind.ACCUMULATIVE)

    with pytest.raises(ViewFieldError):
        used_bytes.set("tenant-a", 1.0)


def test_delta_on_a_convergent_field_is_rejected() -> None:
    """A delta has no defined meaning against a last writer wins field."""
    view = View()
    accessed = view.declare("accessed_at", FoldKind.CONVERGENT)

    with pytest.raises(ViewFieldError):
        accessed.add("key-a", 1.0)


def test_drop_counter_may_not_move_backwards() -> None:
    """A decreasing counter means two buses are folded into one view."""
    view = View()
    view.observe_dropped_events(10)

    with pytest.raises(ValueError):
        view.observe_dropped_events(9)


def test_declaring_the_same_field_twice_is_rejected() -> None:
    """Two owners of one name would disagree about its fold kind."""
    view = View()
    view.declare("used_bytes", FoldKind.ACCUMULATIVE)

    with pytest.raises(ViewFieldError):
        view.declare("used_bytes", FoldKind.CONVERGENT)


def test_reading_an_undeclared_field_is_rejected() -> None:
    """A typo in a trigger should fail loudly rather than read empty state."""
    view = View()

    with pytest.raises(ViewFieldError):
        view.field("used_byte")

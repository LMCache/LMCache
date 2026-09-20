# SPDX-License-Identifier: Apache-2.0
"""Transition-level deadline tests; native buffer lifetimes are covered elsewhere."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
    PrefetchPhase,
    ResourceState,
)


def make_controller_and_request(
    *, deadline, now, phase=None, draining=False, done=True
):
    phase = PrefetchPhase.LOOKUP if phase is None else phase
    ctrl = PrefetchController.__new__(PrefetchController)
    ctrl._clock = Mock(return_value=now)
    calls = []
    for name in (
        "_poll_lookup_results",
        "_poll_load_results",
        "_transition_to_load_phase",
        "_enter_drain_only",
        "_finish_drain",
        "_finish_request",
        "_finalize_completed_load_tasks",
    ):
        setattr(
            ctrl,
            name,
            Mock(side_effect=lambda *args, _name=name: calls.append(_name)),
        )
    req = SimpleNamespace(
        phase=phase,
        resource_state=(ResourceState.DRAINING if draining else ResourceState.ACTIVE),
        deadline_at=deadline,
        published_retained=object(),
        all_lookups_done=lambda: done,
        all_loads_done=lambda: done,
    )
    signaled = {phase_: {0} for phase_ in PrefetchPhase}
    return ctrl, req, signaled, calls


@pytest.mark.parametrize("now", [0.2, 0.21, 2.0])
def test_late_lookup_cannot_submit_new_load(now):
    ctrl, req, signaled, calls = make_controller_and_request(deadline=0.2, now=now)
    ctrl._advance_request(req, signaled)
    assert calls == ["_poll_lookup_results", "_enter_drain_only", "_finish_drain"]
    ctrl._enter_drain_only.assert_called_once_with(req, now)
    ctrl._transition_to_load_phase.assert_not_called()


@pytest.mark.parametrize("deadline", [None, 0.2])
def test_unexpired_or_unarmed_lookup_can_submit(deadline):
    ctrl, req, signaled, calls = make_controller_and_request(deadline=deadline, now=0.1)
    if deadline is None:
        ctrl._clock.side_effect = AssertionError("unarmed path read clock")
    ctrl._advance_request(req, signaled)
    assert calls == ["_poll_lookup_results", "_transition_to_load_phase"]


def test_completed_load_still_wins_the_completion_wake():
    ctrl, req, signaled, calls = make_controller_and_request(
        deadline=0.2,
        now=1.0,
        phase=PrefetchPhase.PLAN_AND_LOAD,
    )
    ctrl._advance_request(req, signaled)
    assert calls == ["_poll_load_results", "_finish_request"]
    ctrl._clock.assert_not_called()


@pytest.mark.parametrize("phase", list(PrefetchPhase))
def test_already_draining_request_does_not_publish_again(phase):
    ctrl, req, signaled, calls = make_controller_and_request(
        deadline=0.2,
        now=1.0,
        phase=phase,
        draining=True,
    )
    ctrl._advance_request(req, signaled)
    assert calls[-1] == "_finish_drain"
    ctrl._clock.assert_not_called()
    ctrl._enter_drain_only.assert_not_called()


def test_incomplete_lookup_leaves_expiry_to_the_loop():
    ctrl, req, signaled, calls = make_controller_and_request(
        deadline=0.2, now=1.0, done=False
    )
    ctrl._advance_request(req, signaled)
    assert calls == ["_poll_lookup_results"]
    ctrl._clock.assert_not_called()


def test_partial_late_load_still_finalizes_completed_tasks():
    ctrl, req, signaled, calls = make_controller_and_request(
        deadline=0.2,
        now=1.0,
        phase=PrefetchPhase.PLAN_AND_LOAD,
        draining=True,
        done=False,
    )
    ctrl._advance_request(req, signaled)
    assert calls == ["_poll_load_results", "_finalize_completed_load_tasks"]
    ctrl._finish_drain.assert_not_called()


def test_unsignaled_request_remains_untouched():
    ctrl, req, signaled, calls = make_controller_and_request(deadline=0.2, now=1.0)
    signaled[req.phase] = set()
    ctrl._advance_request(req, signaled)
    assert calls == []
    ctrl._clock.assert_not_called()

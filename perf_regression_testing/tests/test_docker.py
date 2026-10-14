# SPDX-License-Identifier: Apache-2.0
# Standard
import itertools

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import docker


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake time: every sleep advances 5 s; the URL never answers."""
    now = itertools.count(0, 5)
    monkeypatch.setattr(docker.time, "time", lambda: next(now))
    monkeypatch.setattr(docker.time, "sleep", lambda s: None)


def test_a_created_container_is_waited_on(
    clock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = iter(["created"] * 30 + ["exited"] * 100)
    monkeypatch.setattr(docker, "status", lambda name: next(states))
    assert not docker.wait_http("http://127.0.0.1:1/", 10_000, "c")
    assert next(states) == "exited"  # gave up only after the exit


def test_an_exited_container_ends_the_wait(
    clock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def status(name: str) -> str:
        calls.append(name)
        return "exited"

    monkeypatch.setattr(docker, "status", status)
    assert not docker.wait_http("http://127.0.0.1:1/", 10_000, "c")
    assert len(calls) == 1


def test_remove_retries_until_a_dying_container_is_gone(
    clock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rm_calls = []

    def run(cmd: list[str], **kw: object) -> None:
        if cmd[1] == "rm":
            rm_calls.append(cmd)

    # Still listed after the first two rm attempts, gone after the third.
    states = iter(["dead", "dead", None])
    monkeypatch.setattr(docker.subprocess, "run", run)
    monkeypatch.setattr(docker, "status", lambda name: next(states))
    assert docker.remove(["c"]) == []
    assert len(rm_calls) == 3


def test_remove_reports_containers_it_could_not_remove(
    clock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docker.subprocess, "run", lambda cmd, **kw: None)
    monkeypatch.setattr(docker, "status", lambda name: "dead")
    assert docker.remove(["a", "b"], timeout_s=30) == ["a", "b"]

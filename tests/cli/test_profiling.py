# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared CLI flame-graph profiler."""

# Standard
from pathlib import Path
import os
import shutil

# Third Party
import pytest

# First Party
from lmcache.cli import profiling
from lmcache.cli.profiling import (
    FlameProfiler,
    ProfileError,
    check_profiling_deps,
    record_attached,
)


def test_check_profiling_deps_missing_tool_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tool names the tool and how to recover.

    Runs on any machine: the toolchain is faked absent via monkeypatch,
    so this never skips and always exercises the error path.
    """
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("on-cpu")

    message = str(excinfo.value)
    assert "perf" in message
    # Actionable: points at the fallback mode.
    assert "off-cpu" in message


def test_check_profiling_deps_missing_py_spy_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The py-spy modes name py-spy and how to install it."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("gil")

    message = str(excinfo.value)
    assert "py-spy" in message
    assert "pip install py-spy" in message


def test_check_profiling_deps_rejects_unknown_mode() -> None:
    """An unknown mode is rejected before any tool lookup."""
    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("wallclock")

    message = str(excinfo.value)
    assert "wallclock" in message
    # The message enumerates what is valid.
    assert "gil" in message


def test_check_profiling_deps_reports_high_perf_paranoid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A paranoid level above 2 fails fast, naming the sysctl."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/perf")
    paranoid = tmp_path / "perf_event_paranoid"
    paranoid.write_text("3\n")
    monkeypatch.setattr(profiling, "_PERF_PARANOID_PATH", str(paranoid))

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("on-cpu")

    message = str(excinfo.value)
    assert "perf_event_paranoid" in message
    assert "sysctl" in message


def test_check_profiling_deps_reports_restrictive_ptrace_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """py-spy cannot trace a non-descendant, so scope > 0 fails fast."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/py-spy")
    scope = tmp_path / "ptrace_scope"
    scope.write_text("1\n")
    monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", str(scope))

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("wall")

    message = str(excinfo.value)
    assert "ptrace_scope" in message
    assert "sysctl" in message


@pytest.mark.parametrize(
    ("mode", "tool"),
    [
        ("off-cpu", "offcputime-bpfcc"),
        ("offwake", "offwaketime-bpfcc"),
        ("wakeup", "wakeuptime-bpfcc"),
    ],
)
def test_check_profiling_deps_bcc_mode_names_its_tool(
    monkeypatch: pytest.MonkeyPatch, mode: str, tool: str
) -> None:
    """Each bcc mode fails fast naming *its* tool, not off-cpu's."""
    # sudo present, the bcc tool absent -> the tool is what is missing.
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None
    )
    monkeypatch.setattr(profiling.os.path, "exists", lambda _p: False)

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps(mode)

    assert tool in str(excinfo.value)


def _raise_no_recorder(*_args: object, **_kwargs: object) -> None:
    """Stand in for ``subprocess.Popen`` to stop before a real recorder."""
    raise RuntimeError("no recorder spawned in test")


def test_attach_without_perf_map_warns_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On-cpu attach to a map-less target warns on stderr, not via log.

    The warning changes how the chart must be read (Python frames will be
    ``[unknown]``), so it must survive a caller that silenced ``log``.
    """
    # Host-independent: skip the real toolchain check, point the map dir
    # at an empty tmp dir (target has no map), and stop before spawning a
    # recorder.
    monkeypatch.setattr(profiling, "check_profiling_deps", lambda _mode: None)
    monkeypatch.setattr(profiling, "_PERF_MAP_DIR", str(tmp_path))
    monkeypatch.setattr(profiling.subprocess, "Popen", _raise_no_recorder)

    prof = FlameProfiler(
        mode="on-cpu",
        output=str(tmp_path / "out.svg"),
        flamegraph_dir=str(tmp_path),
        pid=999_999,  # not our pid -> attach path
        title="test",
    )
    # A no-op logger stands in for a --quiet caller; the warning must not
    # depend on it.
    captured_logs: list[str] = []
    with pytest.raises(RuntimeError):
        prof.start(captured_logs.append)

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "PYTHONPERFSUPPORT=1" in err
    # The warning went to stderr, not through the (silent) logger.
    assert not any("WARNING" in line for line in captured_logs)


def test_attach_with_perf_map_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the target has a perf map, frames resolve and nothing warns."""
    monkeypatch.setattr(profiling, "check_profiling_deps", lambda _mode: None)
    monkeypatch.setattr(profiling, "_PERF_MAP_DIR", str(tmp_path))
    monkeypatch.setattr(profiling.subprocess, "Popen", _raise_no_recorder)
    # Materialise the target's map so the "will resolve" branch is taken.
    (tmp_path / "perf-999999.map").write_text("")

    prof = FlameProfiler(
        mode="on-cpu",
        output=str(tmp_path / "out.svg"),
        flamegraph_dir=str(tmp_path),
        pid=999_999,
        title="test",
    )
    logs: list[str] = []
    with pytest.raises(RuntimeError):
        prof.start(logs.append)

    assert "WARNING" not in capsys.readouterr().err
    assert any("will resolve" in line for line in logs)


def test_record_attached_rejects_dead_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing target is reported before any recorder is spawned."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/py-spy")
    scope = tmp_path / "ptrace_scope"
    scope.write_text("0\n")
    monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", str(scope))

    def _no_such_process(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", _no_such_process)

    with pytest.raises(ProfileError) as excinfo:
        record_attached(
            pid=999999,
            mode="gil",
            output=str(tmp_path / "out.svg"),
            flamegraph_dir="",
            duration=1.0,
            log=lambda _m: None,
        )

    assert "no such process" in str(excinfo.value)


def test_flame_profiler_builds_with_real_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if shutil.which("perf") is None:
        pytest.skip("perf not installed; flame-graph profiling unavailable")

    # Sampling permission is a property of the host, not of this code;
    # pin it so the test asserts on FlameProfiler, not on the kernel.
    paranoid = tmp_path / "perf_event_paranoid"
    paranoid.write_text("1\n")
    monkeypatch.setattr(profiling, "_PERF_PARANOID_PATH", str(paranoid))

    scripts_dir = tmp_path / "FlameGraph"
    scripts_dir.mkdir()
    (scripts_dir / "flamegraph.pl").write_text("#!/usr/bin/perl\n")
    output = tmp_path / "flames" / "out.svg"

    FlameProfiler(
        mode="on-cpu",
        output=str(output),
        flamegraph_dir=str(scripts_dir),
        pid=os.getpid(),
        title="test",
    )

    # The component validated the toolchain and prepared the output dir.
    assert output.parent.is_dir()


def test_flame_profiler_clears_a_stale_svg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A previous run's SVG must not survive to look like a fresh one."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/py-spy")
    scope = tmp_path / "ptrace_scope"
    scope.write_text("0\n")
    monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", str(scope))

    output = tmp_path / "out.svg"
    output.write_text("<svg>stale</svg>")

    FlameProfiler(
        mode="gil",
        output=str(output),
        flamegraph_dir="",
        pid=os.getpid(),
        title="test",
    )

    assert not output.exists()

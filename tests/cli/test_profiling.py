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
from lmcache.cli.profiling import FlameProfiler, ProfileError, check_profiling_deps


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("on-cpu", "perf"), ("gil", "pip install py-spy")],
)
def test_check_profiling_deps_missing_tool_is_actionable(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
) -> None:
    """A missing tool fails fast with a message naming it and how to recover."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps(mode)
    assert expected in str(excinfo.value)


def test_check_profiling_deps_reports_restrictive_ptrace_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without CAP_SYS_PTRACE, a ptrace scope > 0 fails fast for py-spy."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/py-spy")
    # CAP_SYS_PTRACE bypasses the scope, and CI often runs as root with it;
    # force the no-capability case so the check actually fires.
    monkeypatch.setattr(profiling, "_has_cap_sys_ptrace", lambda: False)
    scope = tmp_path / "ptrace_scope"
    scope.write_text("1\n")
    monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", str(scope))

    with pytest.raises(ProfileError) as excinfo:
        check_profiling_deps("wall")

    assert "ptrace_scope" in str(excinfo.value)


def _raise_no_recorder(*_args: object, **_kwargs: object) -> None:
    """Stand in for ``subprocess.Popen`` to stop before a real recorder."""
    raise RuntimeError("no recorder spawned in test")


def test_attach_without_perf_map_warns_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Attaching to a map-less target warns on stderr, surviving --quiet.

    Python frames will be ``[unknown]``, which changes how the chart reads,
    so the warning must not depend on the caller's (silenced) logger.
    """
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
    captured_logs: list[str] = []
    with pytest.raises(RuntimeError):
        prof.start(captured_logs.append)

    err = capsys.readouterr().err
    assert "WARNING" in err and "PYTHONPERFSUPPORT=1" in err
    assert not any("WARNING" in line for line in captured_logs)


def test_flame_profiler_clears_a_stale_svg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A previous run's SVG must not survive to look like a fresh one."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/py-spy")
    monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", "/dev/null")

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

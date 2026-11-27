# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared CLI flame-graph profiler."""

# Standard
from pathlib import Path
from unittest.mock import Mock
import os
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.cli import profiling
from lmcache.cli.profiling import FlameProfiler, ProfileError, check_profiling_deps


@pytest.fixture
def failed_recorder(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Simulate a failed external recorder without attaching to a process."""
    monkeypatch.setattr(shutil, "which", lambda name: name)
    monkeypatch.setattr(profiling, "check_profiling_deps", lambda _mode: None)
    monkeypatch.setattr(profiling.time, "sleep", lambda _duration: None)
    proc = Mock(spec=subprocess.Popen, returncode=7, stdout=Mock())
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=proc))
    return proc


@pytest.mark.parametrize("mode", ["gil", "wall", "off-cpu", "on-cpu"])
def test_record_attached_propagates_capture_failure(
    failed_recorder: Mock, tmp_path: Path, mode: str
) -> None:
    """An unsuccessful recording must reach the caller as ProfileError."""
    # perf script's empty output represents a capture with no samples.
    failed_recorder.stdout.readline.return_value = b""
    with pytest.raises(ProfileError, match="status 7|no stacks|no samples"):
        profiling.record_attached(
            pid=os.getppid(),
            mode=mode,
            output=str(tmp_path / "out.svg"),
            flamegraph_dir=str(tmp_path),
            duration=1,
            log=lambda _message: None,
        )
    assert not list(tmp_path.glob("*.folded"))
    assert not list(tmp_path.glob("*.perf.data"))


@pytest.mark.parametrize("mode", ["gil", "wall", "off-cpu", "on-cpu"])
def test_benchmark_stop_keeps_capture_failure_nonfatal(
    failed_recorder: Mock, tmp_path: Path, mode: str
) -> None:
    """Optional benchmark profiling logs failures and permits cleanup."""
    failed_recorder.stdout.readline.return_value = b""
    profiler = FlameProfiler(
        mode=mode,
        output=str(tmp_path / "out.svg"),
        flamegraph_dir=str(tmp_path),
        pid=os.getppid(),
        title="test",
    )
    logs: list[str] = []
    profiler.start(logs.append)
    profiler.stop(logs.append)
    assert any("failed:" in line for line in logs)
    profiler.stop(logs.append)
    failed_recorder.send_signal.assert_called_once()


def test_record_attached_rejects_missing_svg(
    failed_recorder: Mock, tmp_path: Path
) -> None:
    """Even a clean py-spy exit must produce an SVG to count as success."""
    failed_recorder.returncode = 0
    with pytest.raises(ProfileError, match="wrote no flame graph"):
        profiling.record_attached(
            pid=os.getppid(),
            mode="gil",
            output=str(tmp_path / "out.svg"),
            flamegraph_dir="",
            duration=1,
            log=lambda _message: None,
        )


@pytest.mark.parametrize("interrupted", [False, True])
def test_record_attached_accepts_svg(
    failed_recorder: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
) -> None:
    """An SVG is successful, including after Ctrl-C ends the recording."""
    if interrupted:
        monkeypatch.setattr(
            profiling.time, "sleep", Mock(side_effect=KeyboardInterrupt)
        )
    output = tmp_path / "out.svg"
    failed_recorder.returncode = 0
    failed_recorder.wait.side_effect = lambda **_kwargs: output.write_text("<svg/>")
    logs: list[str] = []
    profiling.record_attached(
        pid=os.getppid(),
        mode="gil",
        output=str(output),
        flamegraph_dir="",
        duration=1,
        log=logs.append,
    )
    assert output.read_text() == "<svg/>"
    assert f"[Profile] wrote {output}" in logs


def test_record_attached_propagates_render_failure_and_cleans_capture(
    failed_recorder: Mock, tmp_path: Path
) -> None:
    """Usable bcc samples do not hide a failed renderer or skip cleanup."""
    capture = tmp_path / "out.svg.folded"
    failed_recorder.wait.side_effect = lambda **_kwargs: capture.write_text(
        "main;work 1\n"
    )
    with pytest.raises(ProfileError, match="flamegraph.pl exited with code 7"):
        profiling.record_attached(
            pid=os.getppid(),
            mode="off-cpu",
            output=str(tmp_path / "out.svg"),
            flamegraph_dir=str(tmp_path),
            duration=1,
            log=lambda _message: None,
        )
    assert not capture.exists()
    assert not (tmp_path / "out.svg.recorder.err").exists()


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

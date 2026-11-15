# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache tool flamegraph`` CLI command."""

# Standard
from pathlib import Path
from unittest.mock import Mock
import argparse
import os
import shutil
import subprocess

# Third Party
import pytest

# First Party
from lmcache.cli.commands.tool import ToolCommand
from lmcache.cli.commands.tool.flamegraph import FlamegraphCommand


def _args(**overrides: object) -> argparse.Namespace:
    base = {
        "pid": 1,
        "mode": "gil",
        "duration": 5.0,
        "output": "",
        "flamegraph_scripts_dir": "",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_flamegraph_is_discovered() -> None:
    """The tool group auto-discovers flamegraph without registry edits."""
    parser = argparse.ArgumentParser()
    ToolCommand().register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["tool", "flamegraph", "--pid", "1"])
    assert hasattr(args, "func")
    assert args.tool_target == "flamegraph"


def test_recorder_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed external recorder must make the CLI fail for shell callers."""
    monkeypatch.setattr(shutil, "which", lambda name: name)
    monkeypatch.setattr(subprocess, "Popen", Mock(return_value=Mock(returncode=7)))
    with pytest.raises(SystemExit) as excinfo:
        FlamegraphCommand().execute(
            _args(pid=os.getpid(), duration=0.001, output=str(tmp_path / "out.svg"))
        )
    assert excinfo.value.code == 2
    assert "py-spy exited with status 7" in capsys.readouterr().err


def test_multiple_modes_record_each_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comma-separated ``--mode`` records each mode in turn, de-duplicated."""
    # First Party
    from lmcache.cli import profiling

    recorded: list[str] = []
    monkeypatch.setattr(profiling, "check_profiling_deps", lambda _mode: None)
    monkeypatch.setattr(profiling, "resolve_flamegraph_dir", lambda *_: "")
    monkeypatch.setattr(
        profiling, "record_attached", lambda **kw: recorded.append(kw["mode"])
    )

    FlamegraphCommand().execute(_args(mode="gil,on-cpu,gil"))

    assert recorded == ["gil", "on-cpu"]

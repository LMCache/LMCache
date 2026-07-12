# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache tool flamegraph`` CLI command.

Covers:
- Auto-discovery under ``lmcache tool``
- Argument registration and defaults
- ``execute`` fail-fast paths (missing target, bad toolchain), without
  ever spawning a real recorder
"""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.tool import ToolCommand
from lmcache.cli.commands.tool.flamegraph import FlamegraphCommand


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    """Parser with the whole ``lmcache tool`` group registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    ToolCommand().register(sub)
    return p


class TestRegistration:
    def test_flamegraph_is_discovered(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """The tool group auto-discovers flamegraph without registry edits."""
        args = parser.parse_args(["tool", "flamegraph", "--pid", "1"])
        assert hasattr(args, "func")
        assert args.tool_target == "flamegraph"

    def test_does_not_inherit_generic_output_flags(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Its own --output owns the SVG path; the metrics --format is not
        added by this command's custom register().
        """
        args = parser.parse_args(["tool", "flamegraph", "--pid", "1"])
        assert not hasattr(args, "format")


class TestExecuteFailFast:
    """``execute`` exits non-zero before spawning a recorder."""

    @staticmethod
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

    def test_missing_toolchain_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # First Party
        from lmcache.cli import profiling

        monkeypatch.setattr(profiling.shutil, "which", lambda _name: None)

        with pytest.raises(SystemExit) as excinfo:
            FlamegraphCommand().execute(self._args(mode="gil"))
        assert excinfo.value.code == 2

    def test_dead_pid_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # First Party
        from lmcache.cli import profiling

        # Toolchain present + permissive ptrace, so failure is the pid.
        monkeypatch.setattr(profiling.shutil, "which", lambda _name: "/usr/bin/py-spy")
        monkeypatch.setattr(profiling, "_YAMA_PTRACE_PATH", "/dev/null")

        with pytest.raises(SystemExit) as excinfo:
            FlamegraphCommand().execute(self._args(mode="gil", pid=2_000_000_000))
        assert excinfo.value.code == 2

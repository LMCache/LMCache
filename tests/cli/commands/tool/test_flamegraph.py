# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache tool flamegraph`` CLI command.

Covers auto-discovery under ``lmcache tool`` and the ``execute`` fail-fast
path, without ever spawning a real recorder.
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

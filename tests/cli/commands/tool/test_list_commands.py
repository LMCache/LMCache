# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache tool list-commands``."""

# Standard
from unittest.mock import patch
import io
import json

# First Party
from lmcache.cli.main import main


def _run_cli_json(argv: list[str]) -> dict:
    stdout = io.StringIO()
    with patch("sys.argv", argv), patch("sys.stdout", stdout):
        main()
    return json.loads(stdout.getvalue())


def test_list_commands_reports_discovered_command_tree() -> None:
    """The command lists top-level and nested auto-discovered commands."""
    output = _run_cli_json(
        ["lmcache", "tool", "list-commands", "--format", "json"],
    )

    assert output["title"] == "LMCache CLI Commands"
    metrics = output["metrics"]
    commands = metrics["commands"]
    paths = {entry["path"] for entry in commands}

    assert metrics["command_count"] == len(commands)
    assert "lmcache ping" in paths
    assert "lmcache tool" in paths
    assert "lmcache tool list-commands" in paths


def test_list_commands_reports_stable_command_fields() -> None:
    """Each command entry exposes stable fields for docs and tooling."""
    output = _run_cli_json(
        ["lmcache", "tool", "list-commands", "--format", "json"],
    )
    commands = output["metrics"]["commands"]

    list_commands = next(
        entry for entry in commands if entry["path"] == "lmcache tool list-commands"
    )

    assert list_commands["name"] == "list-commands"
    assert list_commands["type"] == "command"
    assert list_commands["depth"] == 2
    assert list_commands["help"]

    tool = next(entry for entry in commands if entry["path"] == "lmcache tool")
    assert tool["name"] == "tool"
    assert tool["type"] == "group"
    assert tool["depth"] == 1

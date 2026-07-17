# SPDX-License-Identifier: Apache-2.0
"""``lmcache tool list-commands`` - list discovered CLI commands."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand, CompositeCommand


def _command_type(command: BaseCommand) -> str:
    """Return the command kind shown in CLI output.

    Args:
        command: Command instance to classify.

    Returns:
        ``"group"`` for composite commands, otherwise ``"command"``.
    """
    if isinstance(command, CompositeCommand):
        return "group"
    return "command"


def _iter_command_entries(
    commands: Iterable[BaseCommand],
    prefix: tuple[str, ...] = ("lmcache",),
) -> Iterable[dict[str, object]]:
    """Yield command metadata for a command tree.

    Args:
        commands: Commands at the current tree level.
        prefix: CLI path tokens before the current level.

    Yields:
        Dictionaries containing stable command metadata.
    """
    for command in commands:
        path_tokens = (*prefix, command.name())
        yield {
            "path": " ".join(path_tokens),
            "name": command.name(),
            "help": command.help(),
            "type": _command_type(command),
            "depth": len(path_tokens) - 1,
        }
        if isinstance(command, CompositeCommand):
            yield from _iter_command_entries(
                command.subcommands().values(),
                path_tokens,
            )


def collect_command_entries() -> list[dict[str, object]]:
    """Collect metadata for all auto-discovered LMCache CLI commands.

    Returns:
        Command entries sorted by CLI path for deterministic output.
    """
    # Deferred import avoids a circular dependency while command modules are
    # being imported by the auto-discovery path.
    # First Party
    from lmcache.cli.commands import ALL_COMMANDS

    return sorted(
        _iter_command_entries(ALL_COMMANDS),
        key=lambda entry: str(entry["path"]),
    )


class ListCommandsCommand(BaseCommand):
    """List the command tree discovered by the LMCache CLI."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"list-commands"``.
        """
        return "list-commands"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache tool -h``.
        """
        return "List auto-discovered CLI commands."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments.

        Args:
            _parser: The ``ArgumentParser`` for this subcommand.
        """

    def execute(self, args: argparse.Namespace) -> None:
        """List all discovered CLI commands.

        Args:
            args: Parsed CLI arguments.
        """
        entries = collect_command_entries()
        metrics = self.create_metrics("LMCache CLI Commands", args)
        metrics.add("command_count", "Command count", len(entries))
        for index, entry in enumerate(entries):
            section = metrics.add_list_section(
                "commands",
                f"command_{index}",
                str(entry["path"]),
            )
            section.add("path", "Path", entry["path"])
            section.add("name", "Name", entry["name"])
            section.add("help", "Help", entry["help"])
            section.add("type", "Type", entry["type"])
            section.add("depth", "Depth", entry["depth"])
        metrics.emit()

# SPDX-License-Identifier: Apache-2.0
"""Unified ``lmcache`` CLI entry point."""

# Standard
import argparse
import sys

# First Party
from lmcache.cli.server import register_server_command


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand.

    This is the entry point for the unified ``lmcache`` console script.
    If no subcommand is given, the help message is printed and the
    process exits with code 1.

    Raises:
        SystemExit: When no subcommand is provided.
    """
    parser = argparse.ArgumentParser(prog="lmcache", description="LMCache CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register subcommands
    register_server_command(subparsers)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)

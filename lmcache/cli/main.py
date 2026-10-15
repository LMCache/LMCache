# SPDX-License-Identifier: Apache-2.0
"""LMCache CLI entry point.

``main`` is the only reader of ambient ``sys.argv`` and the only place the
invocation route is chosen; importing this module imports no concrete
command wrapper modules.  See section 1 of
``docs/design/cli/framework-and-metrics.md``.
"""

# Standard
from collections.abc import Sequence
from typing import Optional
import argparse
import sys

# First Party
from lmcache.banner import print_banner_once
from lmcache.cli.commands import register_commands
from lmcache.logging import init_logger

logger = init_logger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point registered as ``lmcache`` in *pyproject.toml*.

    Args:
        argv: Invocation tokens after the program name.  ``None`` (the
            console-script path) reads ``sys.argv[1:]``.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)

    print_banner_once(sys.stderr)
    parser = argparse.ArgumentParser(
        prog="lmcache",
        description="LMCache — KV cache management for LLM serving",
    )
    subparsers = parser.add_subparsers(dest="command")

    register_commands(subparsers, tokens)

    args = parser.parse_args(tokens)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:  # noqa: BLE001
        logger.exception("Command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

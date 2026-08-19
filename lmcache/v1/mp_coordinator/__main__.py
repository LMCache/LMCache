# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the mp coordinator process.

Run with ``python -m lmcache.v1.mp_coordinator``. Accepts the same flags as
``lmcache coordinator`` (see :class:`CoordinatorCommand`); an unset flag leaves
the corresponding :class:`MPCoordinatorConfig` default.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.coordinator import CoordinatorCommand


def main() -> None:
    """Parse coordinator flags and serve the app.

    Delegates to :class:`CoordinatorCommand` so the module entrypoint and the
    ``lmcache coordinator`` subcommand share one flag set and one config path.
    """
    command = CoordinatorCommand()
    parser = argparse.ArgumentParser(
        prog="python -m lmcache.v1.mp_coordinator",
        description=command.help(),
    )
    command.add_arguments(parser)
    command.execute(parser.parse_args())


if __name__ == "__main__":
    main()

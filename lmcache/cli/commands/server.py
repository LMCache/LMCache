# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` — launch the LMCache server (ZMQ + HTTP)."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class ServerCommand(BaseCommand):
    """CLI command that launches the LMCache server (ZMQ + HTTP)."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"server"``.
        """
        return "server"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Launch the LMCache server (ZMQ + HTTP)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add server-specific arguments to the parser.

        Composes argument groups from the multiprocess, storage manager,
        HTTP frontend, Prometheus, and telemetry config modules.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        # First Party
        from lmcache.v1.distributed.config import add_storage_manager_args
        from lmcache.v1.mp_observability.config import add_observability_args
        from lmcache.v1.multiprocess.config import (
            add_http_frontend_args,
            add_mp_server_args,
        )

        add_mp_server_args(parser)
        add_storage_manager_args(parser)
        add_http_frontend_args(parser)
        add_observability_args(parser)

    def execute(self, args: argparse.Namespace) -> None:
        """Parse CLI arguments into config objects and launch the HTTP server.

        Args:
            args: Parsed CLI arguments.
        """
        # First Party
        from lmcache.v1.distributed.config import parse_args_to_config
        from lmcache.v1.mp_observability.config import (
            parse_args_to_observability_config,
        )
        from lmcache.v1.multiprocess.config import (
            parse_args_to_http_frontend_config,
            parse_args_to_mp_server_config,
        )
        from lmcache.v1.multiprocess.http_server import run_http_server

        run_http_server(
            http_config=parse_args_to_http_frontend_config(args),
            mp_config=parse_args_to_mp_server_config(args),
            storage_manager_config=parse_args_to_config(args),
            obs_config=parse_args_to_observability_config(args),
        )

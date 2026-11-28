# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` — launch the LMCache request server and HTTP API."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand, missing_grpc_bindings_hint
from lmcache.v1.common.logging import init_logger

logger = init_logger(__name__)


class ServerCommand(BaseCommand):
    """CLI command that launches the LMCache request server and HTTP API."""

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
        return "Launch the LMCache request server and HTTP API."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add server-specific arguments to the parser.

        Composes argument groups from the multiprocess, storage manager,
        HTTP frontend, Prometheus, and telemetry config modules.
        When the server runtime cannot be imported, logs the remedy and skips
        argument registration; ``execute`` then prints the same remedy: an
        editable build that skipped the gRPC generator is pointed at it, a
        lightweight ``lmcache-cli`` install at ``pip install lmcache``.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        try:
            # First Party
            from lmcache.v1.distributed.config import add_storage_manager_args
            from lmcache.v1.mp_observability.config import add_observability_args
            from lmcache.v1.multiprocess.config import (
                add_coordinator_args,
                add_http_frontend_args,
                add_mp_server_args,
                add_p2p_args,
            )

            add_mp_server_args(parser)
            add_p2p_args(parser)
            add_storage_manager_args(parser)
            add_http_frontend_args(parser)
            add_observability_args(parser)
            add_coordinator_args(parser)
        except ImportError as e:
            advice = missing_grpc_bindings_hint(e) or (
                "install the full lmcache package to use 'server' and 'bench'."
            )
            logger.warning("Cannot load the 'server' runtime (%s); %s", e, advice)
            return

    def execute(self, args: argparse.Namespace) -> None:
        """Parse CLI arguments into config objects and launch the HTTP server.

        Args:
            args: Parsed CLI arguments.

        Raises:
            SystemExit: When server dependencies are not installed.
        """
        # Standard
        import sys

        try:
            # First Party
            from lmcache.v1.distributed.config import parse_args_to_config
            from lmcache.v1.mp_observability.config import (
                parse_args_to_observability_config,
            )
            from lmcache.v1.multiprocess.config import (
                parse_args_to_coordinator_config,
                parse_args_to_http_frontend_config,
                parse_args_to_mp_server_config,
            )
            from lmcache.v1.multiprocess.http_auth import HTTPAuthenticationConfig
            from lmcache.v1.multiprocess.http_server import run_http_server
        except ImportError as e:
            print(
                f"The 'lmcache server' command cannot load its runtime: {e}",
                missing_grpc_bindings_hint(e)
                or "Install the full lmcache package with: pip install lmcache",
                sep="\n",
                file=sys.stderr,
            )
            sys.exit(1)

        run_http_server(
            http_config=parse_args_to_http_frontend_config(args),
            mp_config=parse_args_to_mp_server_config(args),
            storage_manager_config=parse_args_to_config(args),
            obs_config=parse_args_to_observability_config(args),
            coordinator_config=parse_args_to_coordinator_config(args),
            http_authentication=HTTPAuthenticationConfig.from_sources(
                args.api_key_file
            ),
        )

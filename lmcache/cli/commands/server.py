# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` — start the LMCache cache server.

Wraps the existing multiprocess server startup, exposing it as a CLI
subcommand instead of requiring ``python3 -m lmcache.v1.multiprocess.http_server``.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class ServerCommand(BaseCommand):
    """Start the LMCache cache server (ZMQ + optional HTTP frontend)."""

    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "Start the LMCache cache server."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Reuse the existing argument helpers so that every flag accepted
        # by the standalone entry points is also available here.
        # First Party
        from lmcache.v1.distributed.config import add_storage_manager_args
        from lmcache.v1.mp_observability.config import add_prometheus_args
        from lmcache.v1.mp_observability.telemetry import add_telemetry_args
        from lmcache.v1.multiprocess.config import (
            add_http_frontend_args,
            add_mp_server_args,
        )

        add_mp_server_args(parser)
        add_storage_manager_args(parser)
        add_http_frontend_args(parser)
        add_prometheus_args(parser)
        add_telemetry_args(parser)

        parser.add_argument(
            "--no-http",
            action="store_true",
            default=False,
            help="Disable the HTTP frontend (ZMQ-only mode).",
        )

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Register with deferred argument setup.

        The server's ``add_arguments`` imports heavyweight modules
        (torch, etc.) via the existing ``add_*_args`` helpers.  We
        catch ``ImportError`` so that ``lmcache -h`` and other
        subcommands still work in minimal environments where the
        server dependencies are not installed.
        """
        parser = subparsers.add_parser(self.name(), help=self.help())
        try:
            self.add_arguments(parser)
        except ImportError as exc:
            msg = str(exc)
            logger.debug("Server dependencies not available: %s", msg)
            parser.set_defaults(
                func=lambda _args, _msg=msg: logger.error(
                    "Cannot start server — missing dependency: %s. "
                    "Install the full LMCache package to use 'lmcache server'.",
                    _msg,
                )
            )
            return
        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        # First Party
        from lmcache.v1.distributed.config import parse_args_to_config
        from lmcache.v1.mp_observability.config import (
            parse_args_to_prometheus_config,
        )
        from lmcache.v1.mp_observability.telemetry import (
            parse_args_to_telemetry_config,
        )
        from lmcache.v1.multiprocess.config import (
            parse_args_to_http_frontend_config,
            parse_args_to_mp_server_config,
        )

        mp_config = parse_args_to_mp_server_config(args)
        storage_config = parse_args_to_config(args)
        prometheus_config = parse_args_to_prometheus_config(args)
        telemetry_config = parse_args_to_telemetry_config(args)

        if args.no_http:
            # First Party
            from lmcache.v1.multiprocess.server import run_cache_server

            logger.info(
                "Starting LMCache server (ZMQ-only) at "
                f"{mp_config.host}:{mp_config.port}"
            )
            run_cache_server(
                mp_config,
                storage_config,
                prometheus_config,
                telemetry_config,
            )
        else:
            # First Party
            from lmcache.v1.multiprocess.http_server import run_http_server

            http_config = parse_args_to_http_frontend_config(args)
            logger.info(
                "Starting LMCache server at "
                f"{mp_config.host}:{mp_config.port} "
                f"(HTTP at {http_config.http_host}:{http_config.http_port})"
            )
            run_http_server(
                http_config,
                mp_config,
                storage_config,
                prometheus_config,
                telemetry_config,
            )

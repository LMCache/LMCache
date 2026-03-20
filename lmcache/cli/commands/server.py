# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` — launch the LMCache server (ZMQ + HTTP)."""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.v1.distributed.config import (
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.mp_observability.config import (
    add_prometheus_args,
    parse_args_to_prometheus_config,
)
from lmcache.v1.mp_observability.telemetry import (
    add_telemetry_args,
    parse_args_to_telemetry_config,
)
from lmcache.v1.multiprocess.config import (
    add_http_frontend_args,
    add_mp_server_args,
    parse_args_to_http_frontend_config,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.http_server import run_http_server


class ServerCommand(BaseCommand):
    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "Launch the LMCache server (ZMQ + HTTP)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_mp_server_args(parser)
        add_storage_manager_args(parser)
        add_http_frontend_args(parser)
        add_prometheus_args(parser)
        add_telemetry_args(parser)

    def execute(self, args: argparse.Namespace) -> None:
        run_http_server(
            http_config=parse_args_to_http_frontend_config(args),
            mp_config=parse_args_to_mp_server_config(args),
            storage_manager_config=parse_args_to_config(args),
            prometheus_config=parse_args_to_prometheus_config(args),
            telemetry_config=parse_args_to_telemetry_config(args),
        )

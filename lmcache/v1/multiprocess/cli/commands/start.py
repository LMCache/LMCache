# SPDX-License-Identifier: Apache-2.0

"""
``start`` sub-command — launches the LMCache server.

Supports two modes controlled by ``--engine-type``:

* ``default`` — standard :class:`MPCacheEngine`
* ``blend``   — cross-request :class:`BlendEngineV2`

And optionally ``--with-http`` to enable the HTTP frontend.
"""

# Standard
import argparse

# First Party
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


def _run(args: argparse.Namespace) -> None:
    """Entry-point called by the CLI dispatcher."""
    mp_config = parse_args_to_mp_server_config(args)
    storage_config = parse_args_to_config(args)
    prom_config = parse_args_to_prometheus_config(args)
    tele_config = parse_args_to_telemetry_config(args)

    if getattr(args, "with_http", False):
        # First Party
        from lmcache.v1.multiprocess.http_server import (
            run_http_server,
        )

        http_config = parse_args_to_http_frontend_config(args)
        run_http_server(
            http_config=http_config,
            mp_config=mp_config,
            storage_manager_config=storage_config,
            prometheus_config=prom_config,
            telemetry_config=tele_config,
        )
    else:
        if mp_config.engine_type == "blend":
            # First Party
            from lmcache.v1.multiprocess.blend_server_v2 import (
                run_cache_server,
            )
        else:
            # First Party
            from lmcache.v1.multiprocess.server import (
                run_cache_server,
            )

        run_cache_server(
            mp_config=mp_config,
            storage_manager_config=storage_config,
            prometheus_config=prom_config,
            telemetry_config=tele_config,
        )


def register_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``start`` sub-command."""
    parser = subparsers.add_parser(
        "start",
        help="Start the LMCache server",
        description="Start the LMCache multiprocess cache server.",
    )
    parser.add_argument(
        "--with-http",
        action="store_true",
        default=False,
        help="Enable the HTTP frontend (uvicorn/FastAPI).",
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_prometheus_args(parser)
    add_telemetry_args(parser)
    add_http_frontend_args(parser)
    parser.set_defaults(func=_run)

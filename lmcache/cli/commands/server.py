# SPDX-License-Identifier: Apache-2.0
"""``lmcache server`` — launch the LMCache server (ZMQ + HTTP)."""

# Standard
import argparse
import os

# First Party
from lmcache.cli.commands.base import BaseCommand


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


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
        Silently skips argument registration when server dependencies
        (e.g. CUDA extensions) are not installed; ``execute`` will then
        print an actionable error.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        try:
            # First Party
            from lmcache.v1.distributed.config import add_storage_manager_args
            from lmcache.v1.mp_observability.config import add_observability_args
            from lmcache.v1.multiprocess.config import (
                add_http_frontend_args,
                add_mp_server_args,
            )

            add_mp_server_args(parser)
            add_storage_manager_args(parser, require_core_args=False)
            add_http_frontend_args(parser)
            add_observability_args(parser)
            parser.add_argument(
                "--config-file",
                type=str,
                default=None,
                help=(
                    "Load LMCache engine config values for supported server "
                    "startup fields. Equivalent to LMCACHE_CONFIG_FILE for "
                    "lmcache server."
                ),
            )
            native_group = parser.add_argument_group(
                "Native MP Server",
                "Experimental native C++ MP server launcher.",
            )
            native_group.add_argument(
                "--native",
                action="store_true",
                help=("Launch the native C++ MP server."),
            )
            native_group.add_argument(
                "--native-cuda",
                action="store_true",
                help=(
                    "Launch the CUDA-enabled native C++ MP server build. "
                    "Implies --native."
                ),
            )
            native_group.add_argument(
                "--native-no-cuda",
                action="store_true",
                help=(
                    "Launch the no-CUDA native C++ MP server build for "
                    "controller-only checks. STORE/RETRIEVE cannot move vLLM "
                    "CUDA KV bytes in this mode. Implies --native."
                ),
            )
            native_group.add_argument(
                "--python",
                action="store_true",
                help="Force the Python MP server even if native mode is enabled.",
            )
            native_group.add_argument(
                "--native-disk-path",
                type=str,
                default=None,
                help="Disk spill path for the native C++ MP server.",
            )
        except ImportError as e:
            print(
                f"Failed to import server dependencies: {e}. "
                "Install the full lmcache package to use 'lmcache server'."
            )
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
                parse_args_to_http_frontend_config,
                parse_args_to_mp_server_config,
            )
            from lmcache.v1.multiprocess.http_server import run_http_server
            from lmcache.v1.multiprocess.native_launcher import (
                apply_lmcache_engine_config_to_args,
            )
        except ImportError:
            print(
                "The 'lmcache server' command requires the full lmcache "
                "installation with CUDA extensions.\n"
                "Install with: pip install lmcache",
                file=sys.stderr,
            )
            sys.exit(1)

        native_requested = (
            bool(getattr(args, "native", False))
            or bool(getattr(args, "native_cuda", False))
            or bool(getattr(args, "native_no_cuda", False))
            or _env_flag_enabled("LMCACHE_MP_NATIVE")
            or _env_flag_enabled("LMCACHE_MP_NATIVE_CUDA")
            or _env_flag_enabled("LMCACHE_MP_NATIVE_NO_CUDA")
        )
        use_native = native_requested and not bool(getattr(args, "python", False))
        try:
            apply_lmcache_engine_config_to_args(args, validate_native=use_native)
        except ValueError as exc:
            print(f"invalid server config: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

        if args.l1_size_gb is None:
            print(
                "invalid server config: --l1-size-gb is required when it is "
                "not provided by --config-file or LMCACHE_CONFIG_FILE",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if args.eviction_policy is None:
            print(
                "invalid server config: --eviction-policy is required when it "
                "is not provided by --config-file or LMCACHE_CONFIG_FILE",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if use_native:
            # First Party
            from lmcache.v1.multiprocess.native_launcher import run_native_server

            run_native_server(args)
            return

        run_http_server(
            http_config=parse_args_to_http_frontend_config(args),
            mp_config=parse_args_to_mp_server_config(args),
            storage_manager_config=parse_args_to_config(args),
            obs_config=parse_args_to_observability_config(args),
        )

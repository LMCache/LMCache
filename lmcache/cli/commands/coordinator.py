# SPDX-License-Identifier: Apache-2.0
"""``lmcache coordinator`` — launch the LMCache mp coordinator (HTTP).

The coordinator tracks mp server instances via a registry and evicts those
whose heartbeats lapse. Configuration comes from CLI flags only; an unset flag
leaves the corresponding :class:`MPCoordinatorConfig` default.
"""

# Standard
import argparse
import json

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class CoordinatorCommand(BaseCommand):
    """CLI command that launches the LMCache mp coordinator (HTTP)."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            The string ``"coordinator"``.
        """
        return "coordinator"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Launch the LMCache mp coordinator (HTTP)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add coordinator-specific arguments to the parser.

        Each flag defaults to ``None`` so that :meth:`execute` can tell an
        unset flag from an explicit one and leave the corresponding
        :class:`MPCoordinatorConfig` default in place.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Host the coordinator's HTTP server binds to (default: 0.0.0.0).",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port the coordinator's HTTP server binds to (default: 9300).",
        )
        parser.add_argument(
            "--instance-timeout",
            type=float,
            default=None,
            help=(
                "Seconds without a heartbeat after which an instance is evicted "
                "(default: 30)."
            ),
        )
        parser.add_argument(
            "--health-check-interval",
            type=float,
            default=None,
            help=(
                "Seconds between health-check sweeps; 0 disables the loop "
                "(default: 10)."
            ),
        )
        parser.add_argument(
            "--eviction-check-interval",
            type=float,
            default=None,
            help=(
                "Seconds between L2 eviction sweeps; 0 disables the loop (default: 5)."
            ),
        )
        parser.add_argument(
            "--eviction-ratio",
            type=float,
            default=None,
            help=(
                "Fraction of tracked keys (by count) to evict per cycle, "
                "0.0 to 1.0 (default: 0.2)."
            ),
        )
        parser.add_argument(
            "--trigger-watermark",
            type=float,
            default=None,
            help=(
                "Eviction fires when usage reaches this fraction of the "
                "quota, 0.0 (exclusive) to 1.0 (default: 1.0)."
            ),
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=None,
            help=(
                "Tokens per KV chunk: the CacheBlend match unit and the unit used "
                "to resolve pin token_ids to keys. Must equal the MP servers' "
                "--chunk-size (default: 256)."
            ),
        )
        parser.add_argument(
            "--hash-algorithm",
            type=str,
            default=None,
            help=(
                "Token hash algorithm for pin key resolution; must equal the MP "
                "servers' --hash-algorithm. 'blake3' (default) is self-contained; "
                "other algorithms require vLLM importable in the coordinator."
            ),
        )
        parser.add_argument(
            "--enable-blend-lookup",
            action="store_true",
            default=None,
            help=(
                "Index stored chunk content so POST /directory/blend-lookup "
                "can serve fleet CacheBlend reuse. Off by default: hashing "
                "content costs CPU on every store and is useless without "
                "CacheBlend."
            ),
        )
        parser.add_argument(
            "--blend-probe-stride",
            type=int,
            default=None,
            help=(
                "Positions between CacheBlend match probes; 1 probes every "
                "offset for full recall (default: 1)."
            ),
        )
        parser.add_argument(
            "--checkpoint-path",
            type=str,
            default=None,
            help=(
                "File to checkpoint the coordinator's derived state to, so a "
                "restart resumes instead of starting cold. Unset disables it."
            ),
        )
        parser.add_argument(
            "--checkpoint-interval",
            type=float,
            default=None,
            help=(
                "Seconds between checkpoint writes; 0 writes only on a clean "
                "stop (default: 60). Ignored without --checkpoint-path."
            ),
        )
        parser.add_argument(
            "--extra-config",
            type=str,
            default=None,
            help=(
                "JSON object of settings the core flags do not name, read by "
                "whichever view or controller looks for them."
            ),
        )
        parser.add_argument(
            "--metadata-path",
            type=str,
            default=None,
            help=(
                "File to store operator-set state (L2 pins and per-cache_salt "
                "quotas) in. Unset means that state is lost on restart."
            ),
        )
        parser.add_argument(
            "--timeout-keep-alive",
            type=int,
            default=None,
            help=(
                "Seconds the HTTP server keeps idle connections open "
                "before closing them (default: 10)."
            ),
        )
        parser.add_argument(
            "--disable-metrics",
            action="store_true",
            default=None,
            help="Disable OpenTelemetry metrics (enabled by default).",
        )
        parser.add_argument(
            "--otlp-endpoint",
            type=str,
            default=None,
            help=(
                "OTLP gRPC endpoint for metrics push mode. When unset, "
                "Prometheus scrapes /metrics on the coordinator HTTP port."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Build the coordinator config and serve the app with uvicorn.

        Builds the config from the supplied flags; every flag left unset keeps
        its :class:`MPCoordinatorConfig` default.

        Args:
            args: Parsed CLI arguments.

        Raises:
            SystemExit: When coordinator dependencies are not installed.
        """
        # Standard
        import sys

        try:
            # Third Party
            import uvicorn

            # First Party
            from lmcache.v1.mp_coordinator.app import create_app
            from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
            from lmcache.v1.mp_coordinator.observability import (
                init_coordinator_metrics,
            )
        except ImportError:
            print(
                "The 'lmcache coordinator' command requires the full lmcache "
                "installation.\nInstall with: pip install lmcache",
                file=sys.stderr,
            )
            sys.exit(1)

        fields = {
            field: value
            for field, value in (
                ("host", args.host),
                ("port", args.port),
                ("instance_timeout", args.instance_timeout),
                ("health_check_interval", args.health_check_interval),
                ("eviction_check_interval", args.eviction_check_interval),
                ("eviction_ratio", args.eviction_ratio),
                ("trigger_watermark", args.trigger_watermark),
                ("chunk_size", args.chunk_size),
                ("hash_algorithm", args.hash_algorithm),
                ("enable_blend_lookup", args.enable_blend_lookup),
                ("blend_probe_stride", args.blend_probe_stride),
                ("checkpoint_path", args.checkpoint_path),
                ("checkpoint_interval", args.checkpoint_interval),
                ("metadata_path", args.metadata_path),
                ("timeout_keep_alive", args.timeout_keep_alive),
                ("otlp_endpoint", args.otlp_endpoint),
            )
            if value is not None
        }
        if args.disable_metrics is not None:
            fields["metrics_enabled"] = not args.disable_metrics
        extra_config = _parse_extra_config(args.extra_config)
        if extra_config is not None:
            fields["extra_config"] = extra_config
        config = MPCoordinatorConfig(**fields)

        init_coordinator_metrics(config)
        app = create_app(config)
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level="info",
            timeout_keep_alive=config.timeout_keep_alive,
        )


def _parse_extra_config(raw: str | None) -> dict[str, object] | None:
    """Parse ``--extra-config``.

    Args:
        raw: The JSON object as given, or ``None`` when the flag is unset.

    Returns:
        The parsed settings, or ``None`` so an unset flag leaves the
        config default alone.

    Raises:
        ValueError: If it is not JSON, or is not an object -- a list or a
            bare string would fail far from here, on the first lookup.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--extra-config is not valid JSON: {e}") from None
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--extra-config must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed

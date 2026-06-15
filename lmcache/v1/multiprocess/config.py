# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the multiprocess (ZMQ) server and HTTP frontend.
"""

# Standard
from dataclasses import dataclass, field
import argparse
import json
import math
import os
import uuid


@dataclass
class MPServerConfig:
    """Configuration for the ZMQ-based multiprocess cache server."""

    host: str = "localhost"
    """ZMQ server host."""

    port: int = 5555
    """ZMQ server port."""

    chunk_size: int = 256
    """Chunk size for KV cache operations."""

    max_workers: int = 1
    """Base number of worker threads. Sets default for both GPU and CPU pools."""

    max_gpu_workers: int = 1
    """Worker threads for the GPU affinity pool (STORE/RETRIEVE).
    Resolved from --max-gpu-workers or --max-workers."""

    max_cpu_workers: int = 1
    """Worker threads for the normal (CPU) pool (LOOKUP, END_SESSION, etc.).
    Resolved from --max-cpu-workers or --max-workers."""

    hash_algorithm: str = "blake3"
    """Hash algorithm for token-based operations (builtin, sha256_cbor, blake3)."""

    engine_type: str = "default"
    """Cache engine backend type
    ('default' for standard prefix caching, 'blend' when cacheblend is enabled).
    """

    supported_transfer_mode: str = "auto"
    """Transfer mode: 'lmcache_driven' for server-driven transfer
    (STORE/RETRIEVE, supports CUDA IPC and CPU SHM), 'engine_driven' for
    engine-driven transfer (PREPARE/COMMIT), or 'auto' to enable both."""

    runtime_plugin_config: "RuntimePluginConfig" = field(
        default_factory=lambda: RuntimePluginConfig()
    )
    """Runtime plugin configuration (locations + extra config)."""

    shm_name: str | None = None
    """SHM segment name for engine-driven KV transfer.
    None: auto-allocate (default). "": force pickle. Other: use that name."""

    script_allowed_imports: list[str] = field(default_factory=list)
    """Modules that /run_script endpoint is allowed to import."""

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Stable identity of this MP server, the single source of truth for who
    this server is. Used as the coordinator membership key and projected onto
    the OTel ``service.instance.id`` resource attribute (see
    ``run_cache_server``) so metrics, traces, and coordinator state all key on
    the same id. Set via ``--instance-id``; defaults to a random UUID v4."""


@dataclass
class RuntimePluginConfig:
    """Configuration for runtime plugins."""

    locations: list[str] = field(default_factory=list)
    """Paths to runtime plugin scripts or directories."""

    extra_config: dict = field(default_factory=dict)
    """Extra key-value config forwarded to runtime plugins
    via the JSON config blob.
    Accepts a JSON string on the command line.
    """


DEFAULT_MP_SERVER_CONFIG = MPServerConfig()


@dataclass
class HTTPFrontendConfig:
    """Configuration for the HTTP frontend (uvicorn/FastAPI)."""

    http_host: str = "0.0.0.0"
    """HTTP server host."""

    http_port: int = 8080
    """HTTP server port."""


DEFAULT_HTTP_FRONTEND_CONFIG = HTTPFrontendConfig()


@dataclass
class CoordinatorConfig:
    """Configuration for joining an MP coordinator (registrant side).

    Consumed by the HTTP server's lifespan to start the registration task.
    When :attr:`url` is empty, the server registers with no coordinator and
    runs exactly as before.
    """

    url: str = ""
    """Coordinator base URL, e.g. ``http://coordinator:9300``. Empty disables
    registration."""

    advertise_ip: str = ""
    """IP the coordinator should reach this server at. Empty defers to the
    server's outbound IP (resolved by the registrar)."""

    heartbeat_interval: float = 5.0
    """Seconds between heartbeats. Must be strictly positive and kept well below
    the coordinator's ``INSTANCE_TIMEOUT``."""

    l2_event_reporting: bool = False
    """When ``True``, report L2 store/lookup events to the coordinator for
    fleet-wide usage tracking and eviction."""

    l2_event_flush_interval: float = 1.0
    """Seconds between L2 event flush attempts to the coordinator."""


DEFAULT_COORDINATOR_CONFIG = CoordinatorConfig()


def add_mp_server_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add MP server configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with MP server arguments added.
    """
    mp_group = parser.add_argument_group(
        "MP Server", "Configuration for the ZMQ multiprocess cache server"
    )
    mp_group.add_argument(
        "--instance-id",
        type=str,
        default=None,
        help="Stable identity of this MP server. Used as the coordinator "
        "membership key and as the OTel 'service.instance.id' resource "
        "attribute on every metric and span. Defaults to a random UUID v4 "
        "minted at startup.",
    )
    mp_group.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the ZMQ server. Default is localhost.",
    )
    mp_group.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Port to bind the ZMQ server. Default is 5555.",
    )
    mp_group.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for KV cache operations. Default is 256.",
    )
    mp_group.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Base number of worker threads for both GPU and CPU pools. "
        "Default is 1. Can be overridden per-pool with "
        "--max-gpu-workers and --max-cpu-workers.",
    )
    mp_group.add_argument(
        "--max-gpu-workers",
        type=int,
        default=None,
        help="Worker threads for the GPU affinity pool (STORE/RETRIEVE). "
        "Defaults to --max-workers if not specified.",
    )
    mp_group.add_argument(
        "--max-cpu-workers",
        type=int,
        default=None,
        help="Worker threads for the normal CPU pool (LOOKUP, etc.). "
        "Defaults to --max-workers if not specified.",
    )
    mp_group.add_argument(
        "--hash-algorithm",
        type=str,
        default="blake3",
        help="Hash algorithm for token-based operations "
        "(builtin, sha256_cbor, blake3). Default is blake3.",
    )
    mp_group.add_argument(
        "--engine-type",
        type=str,
        default="default",
        choices=["default", "blend", "blend_legacy"],
        help="Cache engine backend type. 'default' uses standard prefix caching; "
        "'blend' selects CacheBlend V3 (the current implementation); "
        "'blend_legacy' selects the original CacheBlend. Default is 'default'.",
    )
    mp_group.add_argument(
        "--supported-transfer-mode",
        type=str,
        default="auto",
        choices=["lmcache_driven", "engine_driven", "auto"],
        help="Supported transfer mode: 'lmcache_driven' for server-driven "
        "transfer (STORE/RETRIEVE, supports CUDA IPC and CPU SHM), "
        "'engine_driven' for engine-driven transfer (PREPARE/COMMIT), "
        "or 'auto' to enable both transfer paths. Default is 'auto'.",
    )
    mp_group.add_argument(
        "--runtime-plugin-locations",
        type=str,
        nargs="*",
        default=[],
        help="Paths to runtime plugin scripts or "
        "directories to launch alongside the server.",
    )
    mp_group.add_argument(
        "--runtime-plugin-config",
        type=str,
        default="{}",
        help="JSON string of extra key-value config forwarded to runtime "
        "plugins via LMCACHE_RUNTIME_PLUGIN_EXTRA_CONFIG. "
        'Example: \'{"plugin.frontend.heartbeat_url": '
        '"http://localhost:5000/heartbeat"}\'',
    )
    mp_group.add_argument(
        "--shm-name",
        type=str,
        default=None,
        help="SHM segment name for engine-driven KV transfer. "
        "Default (not specified): auto-allocate. "
        'Set to "" to force pickle path (disable SHM). '
        "Set to a name to use that specific SHM segment.",
    )
    mp_group.add_argument(
        "--script-allowed-imports",
        type=str,
        nargs="*",
        default=[],
        help="Python modules that the /run_script endpoint is allowed to "
        "import. Example: --script-allowed-imports numpy pandas",
    )
    return parser


def parse_args_to_mp_server_config(
    args: argparse.Namespace,
) -> MPServerConfig:
    """
    Convert parsed command line arguments to an MPServerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        MPServerConfig: The configuration object.
    """
    base = args.max_workers
    max_gpu = args.max_gpu_workers if args.max_gpu_workers is not None else base
    max_cpu = args.max_cpu_workers if args.max_cpu_workers is not None else base
    try:
        plugin_extra = json.loads(getattr(args, "runtime_plugin_config", None) or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("--runtime-plugin-config is not valid JSON: %s" % exc) from exc
    return MPServerConfig(
        instance_id=args.instance_id or str(uuid.uuid4()),
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        max_workers=base,
        max_gpu_workers=max_gpu,
        max_cpu_workers=max_cpu,
        hash_algorithm=args.hash_algorithm,
        engine_type=args.engine_type,
        supported_transfer_mode=args.supported_transfer_mode,
        runtime_plugin_config=RuntimePluginConfig(
            locations=(args.runtime_plugin_locations or []),
            extra_config=plugin_extra,
        ),
        shm_name=args.shm_name,
        script_allowed_imports=args.script_allowed_imports or [],
    )


def add_http_frontend_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add HTTP frontend configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with HTTP frontend arguments added.
    """
    http_group = parser.add_argument_group(
        "HTTP Frontend", "Configuration for the HTTP frontend server"
    )
    http_group.add_argument(
        "--http-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server. Default is 0.0.0.0.",
    )
    http_group.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port to bind the HTTP server. Default is 8080.",
    )
    return parser


def parse_args_to_http_frontend_config(
    args: argparse.Namespace,
) -> HTTPFrontendConfig:
    """
    Convert parsed command line arguments to an HTTPFrontendConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        HTTPFrontendConfig: The configuration object.
    """
    return HTTPFrontendConfig(
        http_host=args.http_host,
        http_port=args.http_port,
    )


def add_coordinator_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add MP coordinator registration arguments to an existing parser.

    Each flag falls back to its ``LMCACHE_COORDINATOR_*`` environment variable
    so the server can be configured either way (the env var is convenient for
    the Kubernetes downward API); an explicit flag wins over the env var.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with coordinator arguments added.
    """
    group = parser.add_argument_group(
        "Coordinator", "Configuration for joining an MP coordinator"
    )
    group.add_argument(
        "--coordinator-url",
        type=str,
        default=None,
        help="Coordinator base URL (e.g. http://coordinator:9300). When set, "
        "this server registers, heartbeats, and deregisters on shutdown. "
        "Defaults to LMCACHE_COORDINATOR_URL; unset disables registration.",
    )
    group.add_argument(
        "--coordinator-advertise-ip",
        type=str,
        default=None,
        help="IP the coordinator should reach this server at. Defaults to "
        "LMCACHE_COORDINATOR_ADVERTISE_IP, then the server's outbound IP.",
    )
    group.add_argument(
        "--coordinator-heartbeat-interval",
        type=float,
        default=None,
        help="Seconds between heartbeats (must be > 0). Defaults to "
        "LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL, then 5.0.",
    )
    group.add_argument(
        "--coordinator-l2-event-reporting",
        action="store_true",
        default=None,
        help="Report L2 store/lookup events to the coordinator for "
        "fleet-wide usage tracking and eviction. Defaults to "
        "LMCACHE_COORDINATOR_L2_EVENT_REPORTING; unset disables.",
    )
    group.add_argument(
        "--coordinator-l2-event-flush-interval",
        type=float,
        default=None,
        help="Seconds between L2 event flush attempts (must be > 0). "
        "Defaults to LMCACHE_COORDINATOR_L2_EVENT_FLUSH_INTERVAL, then 1.0.",
    )
    return parser


def parse_args_to_coordinator_config(
    args: argparse.Namespace,
) -> CoordinatorConfig:
    """Convert parsed command line arguments to a CoordinatorConfig.

    A flag value takes precedence over its environment variable. The heartbeat
    interval is validated here so a malformed value fails fast at startup
    (runtime best-effort only covers coordinator *reachability*, not config).

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        The configuration object.

    Raises:
        ValueError: If the heartbeat interval is not a positive number.
    """
    url = (
        args.coordinator_url
        if args.coordinator_url is not None
        else os.getenv("LMCACHE_COORDINATOR_URL", "")
    )
    advertise_ip = (
        args.coordinator_advertise_ip
        if args.coordinator_advertise_ip is not None
        else os.getenv("LMCACHE_COORDINATOR_ADVERTISE_IP", "")
    )
    if args.coordinator_heartbeat_interval is not None:
        heartbeat_interval = args.coordinator_heartbeat_interval
    else:
        raw = os.getenv("LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL")
        if raw:
            try:
                heartbeat_interval = float(raw)
            except ValueError as exc:
                raise ValueError(
                    "LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL is not a number: %r" % raw
                ) from exc
        else:
            heartbeat_interval = 5.0
    if not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
        # Reject inf/nan too: inf would register once then sleep forever
        # (never heartbeat), and nan has undefined sleep behavior.
        raise ValueError(
            "coordinator heartbeat interval must be a finite number > 0, "
            "got %s" % heartbeat_interval
        )
    if args.coordinator_l2_event_reporting is not None:
        l2_event_reporting = args.coordinator_l2_event_reporting
    else:
        l2_event_reporting = os.getenv(
            "LMCACHE_COORDINATOR_L2_EVENT_REPORTING", ""
        ).lower() in ("1", "true", "yes")

    if args.coordinator_l2_event_flush_interval is not None:
        l2_event_flush_interval = args.coordinator_l2_event_flush_interval
    else:
        raw = os.getenv("LMCACHE_COORDINATOR_L2_EVENT_FLUSH_INTERVAL")
        if raw:
            try:
                l2_event_flush_interval = float(raw)
            except ValueError as exc:
                raise ValueError(
                    "LMCACHE_COORDINATOR_L2_EVENT_FLUSH_INTERVAL is not a number: %r"
                    % raw
                ) from exc
        else:
            l2_event_flush_interval = 1.0
    if not math.isfinite(l2_event_flush_interval) or l2_event_flush_interval <= 0:
        raise ValueError(
            "coordinator L2 event flush interval must be a finite number > 0, "
            "got %s" % l2_event_flush_interval
        )

    return CoordinatorConfig(
        url=url,
        advertise_ip=advertise_ip,
        heartbeat_interval=heartbeat_interval,
        l2_event_reporting=l2_event_reporting,
        l2_event_flush_interval=l2_event_flush_interval,
    )

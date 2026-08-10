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

try:
    # Standard
    import tomllib  # Python 3.11+ stdlib
except ModuleNotFoundError:  # pragma: no cover (Python 3.10 only)
    import tomli as tomllib  # type: ignore[no-redef, import-not-found]

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


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

    separate_object_groups: bool = False
    """When True, split kernel groups into one object group per
    sliding-window size at KV-cache registration (hybrid models). When False
    (default), all kernel groups share a single full-attention object group."""

    enable_segmented_prefix: bool = False
    """CacheBlend only (engine_type='blend'): on a mid-prefix L2 retrieve
    failure, retain the gapped contiguous prefix so the post-gap chunks stay
    L1-resident (served by the sparse leg as L1 hits, the hole recomputed)
    instead of truncating the prefix at the gap. No effect for other engines."""

    supported_transfer_mode: str = "auto"
    """Transfer mode: 'lmcache_driven' for server-driven transfer
    (STORE/RETRIEVE, supports CUDA IPC and CPU SHM), 'engine_driven' for
    engine-driven transfer (PREPARE/COMMIT), or 'auto' to enable both."""

    runtime_plugin_config: "RuntimePluginConfig" = field(
        default_factory=lambda: RuntimePluginConfig()
    )
    """Runtime plugin configuration (locations + extra config)."""

    p2p_config: "P2PConfig" = field(default_factory=lambda: P2PConfig())
    """Peer-to-peer configuration. P2P is enabled when its advertise URL is
    set."""

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

    worker_reap_timeout_seconds: float = 120.0
    """Silence budget (seconds) after which a ping-proven worker's KV cache
    registration is reaped. 0 disables worker reaping. Keep it >= 3 x the
    engine adapter's heartbeat interval so a few missed pings never reap a live
    worker."""

    worker_registration_grace_seconds: float = 3600.0
    """Silence budget (seconds) for a worker that registered but has never
    sent a PING (model warmup, or death before its first request). Must be
    >= worker_reap_timeout_seconds."""

    enable: list[str] = field(default_factory=list)
    """List of experimental transfer modules to enable. Options: transfer_query
    (see lmcache.v1.multiprocess.modules.experimental.__init___.py)."""

    def __post_init__(self) -> None:
        """Validate the worker-reaping timeouts.

        Raises:
            ValueError: If a timeout is non-finite, the reap timeout is
                negative or a non-zero value below the 30 s floor, or the
                registration grace is below the reap timeout.
        """
        reap = self.worker_reap_timeout_seconds
        grace = self.worker_registration_grace_seconds
        if not math.isfinite(reap) or reap < 0 or (reap != 0 and reap < 30.0):
            raise ValueError(
                "worker reap timeout must be 0 (disabled) or >= 30s; keep it "
                ">= 3 x your configured lmcache.mp.heartbeat_interval "
                f"(default 10s); got {reap}"
            )
        if not math.isfinite(grace) or grace < reap:
            raise ValueError(
                "worker registration grace must be >= the worker reap timeout "
                f"({reap}s); got {grace}"
            )


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


@dataclass
class P2PConfig:
    """Configuration for peer-to-peer KV transfer.

    P2P is enabled when :attr:`advertise_url` is non-empty. It additionally
    requires a coordinator URL for peer discovery (validated at startup).
    """

    advertise_url: str = ""
    """Transfer-channel server ``host:port`` this instance advertises to peers.
    Empty disables P2P."""

    listen_url: str = ""
    """Transfer-channel server ``host:port`` to bind and listen on. Empty
    defers to :attr:`advertise_url`."""

    lookup_timeout: float = 30.0
    """Seconds before a peer lookup result counts as a miss."""

    load_timeout: float = 30.0
    """Seconds before a peer load counts as a failure."""

    transfer_engine: str = "nixl"
    """Transfer-channel implementation to use."""

    @property
    def enabled(self) -> bool:
        """Whether P2P is enabled (an advertise URL is configured)."""
        return bool(self.advertise_url)

    @property
    def effective_listen_url(self) -> str:
        """The listen URL, defaulting to the advertise URL when unset."""
        return self.listen_url or self.advertise_url


DEFAULT_P2P_CONFIG = P2PConfig()


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

    event_reporting: bool = False
    """When ``True``, stream cache store/access/delete events to the
    coordinator, feeding the key directory (fleet-wide placement
    tracking) and, for L2 events, usage/quota tracking and eviction."""

    event_flush_interval: float = 1.0
    """Seconds between cache-event flush attempts to the coordinator."""


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
    mp_group.add_argument(
        "--separate-object-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split kernel groups into one object group per sliding-window size "
        "at KV-cache registration (for hybrid models). (Default is False)",
    )
    mp_group.add_argument(
        "--worker-reap-timeout-seconds",
        type=float,
        default=120.0,
        help="Silence budget (s) before a ping-proven worker's KV cache "
        "registration is reaped. 0 disables reaping. Must be >= 3 x the "
        "engine adapter's heartbeat interval. Default is 120.",
    )
    mp_group.add_argument(
        "--worker-registration-grace-seconds",
        type=float,
        default=3600.0,
        help="Silence budget (s) for a worker that registered but never "
        "pinged (model warmup or early death). Must be >= the worker reap "
        "timeout. Default is 3600.",
    )
    mp_group.add_argument(
        "--enable-segmented-prefix",
        action="store_true",
        help="CacheBlend (--engine-type blend) only: on a mid-prefix L2 "
        "retrieve failure, retain the gapped prefix so post-gap chunks stay "
        "L1-resident instead of truncating at the gap. No effect otherwise.",
    )
    mp_group.add_argument(
        "--enable",
        type=str,
        nargs="*",
        default=[],
        help="List of experimental transfer modules to enable. "
        "Options: transfer_query (see lmcache.v1.multiprocess.modules."
        "experimental.__init___.py).",
    )
    mp_group.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a TOML file providing MP server defaults (any flag set "
        "explicitly on the command line overrides the file). Also read from "
        "the LMCACHE_MP_CONFIG environment variable; the flag wins over the "
        "env var. Missing or unparsable files warn and fall back to "
        "argparse-only behavior (never aborts startup).",
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
        separate_object_groups=args.separate_object_groups,
        enable_segmented_prefix=args.enable_segmented_prefix,
        supported_transfer_mode=args.supported_transfer_mode,
        runtime_plugin_config=RuntimePluginConfig(
            locations=(args.runtime_plugin_locations or []),
            extra_config=plugin_extra,
        ),
        p2p_config=parse_args_to_p2p_config(args),
        shm_name=args.shm_name,
        script_allowed_imports=args.script_allowed_imports or [],
        worker_reap_timeout_seconds=args.worker_reap_timeout_seconds,
        worker_registration_grace_seconds=args.worker_registration_grace_seconds,
        enable=args.enable or [],
    )


# Dest names the TOML file must not set (the config-file path itself; setting
# it from inside the file would be circular and meaningless).
_MP_CONFIG_FILE_EXCLUDED_DESTS = frozenset({"config_file"})

# Memoized argparse defaults for the MP-server group; built lazily by
# _mp_server_arg_defaults() and only read by the merge layer.
_MP_SERVER_DEFAULTS: argparse.Namespace | None = None


def _mp_server_arg_defaults() -> argparse.Namespace:
    """Return a Namespace of the MP-server argparse defaults.

    Built by parsing an empty argv through a fresh parser with
    ``add_mp_server_args``, so the defaults stay in lockstep with the
    argparse declarations (single source of truth). Memoized; the merge
    layer only reads from it.
    """
    global _MP_SERVER_DEFAULTS
    if _MP_SERVER_DEFAULTS is None:
        parser = argparse.ArgumentParser(add_help=False)
        add_mp_server_args(parser)
        _MP_SERVER_DEFAULTS = parser.parse_args([])
    return _MP_SERVER_DEFAULTS


def load_mp_config_from_toml(path: str) -> dict:
    """Load MP-server defaults from a TOML file.

    Keys are normalized to argparse dest names (``max-gpu-workers`` and
    ``max_gpu_workers`` both map to ``max_gpu_workers``), and only dests that
    ``add_mp_server_args`` actually registers are kept. The returned dict is
    keyed by dest name with values shaped to match what argparse would store
    (e.g. ``runtime_plugin_config`` is flattened to a JSON string when given
    as a TOML table).

    Missing files, unreadable files, and invalid TOML all warn and return an
    empty dict — the caller falls back to argparse-only behavior. This never
    raises so server startup is never aborted by a config-file problem.

    Args:
        path: Path to the TOML config file.

    Returns:
        A dest-keyed dict of MP-server config values (empty on any failure).
    """
    try:
        with open(path, "rb") as fin:
            data = tomllib.load(fin)
    except FileNotFoundError:
        logger.warning("MP config file %s not found; using CLI defaults", path)
        return {}
    except tomllib.TOMLDecodeError as exc:
        logger.warning(
            "MP config file %s is invalid TOML (%s); using CLI defaults",
            path,
            exc,
        )
        return {}
    except OSError as exc:
        logger.warning(
            "MP config file %s could not be read (%s); using CLI defaults",
            path,
            exc,
        )
        return {}

    if not isinstance(data, dict):
        logger.warning("MP config file %s is not a mapping; using CLI defaults", path)
        return {}

    defaults = _mp_server_arg_defaults()
    out: dict = {}
    for raw_key, value in data.items():
        if not isinstance(raw_key, str):
            continue
        dest = raw_key.strip().lower().replace("-", "_")
        if dest in _MP_CONFIG_FILE_EXCLUDED_DESTS:
            logger.warning(
                "Ignoring MP config key '%s' from %s (reserved)", raw_key, path
            )
            continue
        if not hasattr(defaults, dest):
            logger.warning("Ignoring unknown MP config key '%s' from %s", raw_key, path)
            continue
        # runtime_plugin_config is a JSON string on the CLI; accept a TOML
        # table too by serializing it back to JSON so parse_args_to_mp_server
        # _config's json.loads keeps working.
        if dest == "runtime_plugin_config" and isinstance(value, dict):
            value = json.dumps(value)
        out[dest] = value
    return out


def merge_mp_config_file_into_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Merge an MP TOML config file into parsed CLI args (in place).

    The file supplies defaults; a CLI flag wins when its value differs from
    the argparse default (i.e. it was passed explicitly). An absent flag lets
    the file win over the argparse default. This preserves backward
    compatibility: with no file, ``args`` is returned unchanged.

    The file path is resolved from ``--config-file`` (registered by
    ``add_mp_server_args``) or the ``LMCACHE_MP_CONFIG`` environment variable;
    the flag wins over the env var. A missing/unreadable file warns and leaves
    ``args`` untouched.

    Args:
        args: Parsed argparse Namespace. Merged in place and returned.

    Returns:
        The same Namespace, with file defaults applied under CLI overrides.
    """
    path = getattr(args, "config_file", None) or os.getenv("LMCACHE_MP_CONFIG")
    if not path:
        return args
    file_config = load_mp_config_from_toml(path)
    if not file_config:
        return args
    defaults = _mp_server_arg_defaults()
    for dest, value in file_config.items():
        cli_value = getattr(args, dest, getattr(defaults, dest))
        if cli_value == getattr(defaults, dest):
            # CLI flag was absent (or equal to default) -> let the file win.
            setattr(args, dest, value)
    return args


def add_p2p_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add peer-to-peer configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with P2P arguments added.
    """
    group = parser.add_argument_group(
        "P2P", "Configuration for peer-to-peer KV transfer"
    )
    group.add_argument(
        "--p2p-advertise-url",
        type=str,
        default="",
        help="Transfer-channel server host:port this instance advertises to "
        "peers. Setting it enables P2P (also requires --coordinator-url).",
    )
    group.add_argument(
        "--p2p-listen-url",
        type=str,
        default="",
        help="Transfer-channel server host:port to bind. Defaults to "
        "--p2p-advertise-url.",
    )
    group.add_argument(
        "--p2p-lookup-timeout",
        type=float,
        default=30.0,
        help="Seconds before a peer lookup result counts as a miss. Default is 30.",
    )
    group.add_argument(
        "--p2p-load-timeout",
        type=float,
        default=30.0,
        help="Seconds before a peer load counts as a failure. Default is 30.",
    )
    group.add_argument(
        "--p2p-transfer-engine",
        type=str,
        default="nixl",
        help="Transfer-channel implementation to use. Default is nixl.",
    )
    return parser


def parse_args_to_p2p_config(
    args: argparse.Namespace,
) -> P2PConfig:
    """Convert parsed command line arguments to a P2PConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        The configuration object.
    """
    return P2PConfig(
        advertise_url=getattr(args, "p2p_advertise_url", "") or "",
        listen_url=getattr(args, "p2p_listen_url", "") or "",
        lookup_timeout=getattr(args, "p2p_lookup_timeout", 30.0),
        load_timeout=getattr(args, "p2p_load_timeout", 30.0),
        transfer_engine=getattr(args, "p2p_transfer_engine", "nixl"),
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
        "--coordinator-event-reporting",
        action="store_true",
        default=None,
        help="Stream cache store/access/delete events to the coordinator, "
        "feeding the key directory (placement tracking) and, for L2 "
        "events, usage/quota tracking and eviction. Defaults to "
        "LMCACHE_COORDINATOR_EVENT_REPORTING; unset disables.",
    )
    group.add_argument(
        "--coordinator-event-flush-interval",
        type=float,
        default=None,
        help="Seconds between cache-event flush attempts (must be > 0). "
        "Defaults to LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL, then 1.0.",
    )
    # Deprecated pre-v0.5.3 aliases, hidden from --help. Released deployers
    # (operator <= v0.5.2, charts) still render these names into server args,
    # and rejecting them crashes the pod on startup. Remove once those
    # deployers are out of the support matrix.
    group.add_argument(
        "--coordinator-l2-event-reporting",
        dest="coordinator_l2_event_reporting",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--coordinator-l2-event-flush-interval",
        dest="coordinator_l2_event_flush_interval",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def parse_args_to_coordinator_config(
    args: argparse.Namespace,
) -> CoordinatorConfig:
    """Convert parsed command line arguments to a CoordinatorConfig.

    A flag value takes precedence over its environment variable. The heartbeat
    interval is validated here so a malformed value fails fast at startup
    (runtime best-effort only covers coordinator *reachability*, not config).

    The event-reporting flags also accept their deprecated pre-v0.5.3
    spellings (``--coordinator-l2-event-*``), logging a deprecation warning
    when used. Precedence per setting: new flag > deprecated flag > env var
    > default.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        The configuration object.

    Raises:
        ValueError: If the heartbeat interval or the event flush interval
            is not a positive finite number.
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
    # Deprecated pre-v0.5.3 flag spellings. Precedence within each setting:
    # new flag > deprecated flag > env var > default.
    deprecated_reporting = getattr(args, "coordinator_l2_event_reporting", None)
    if args.coordinator_event_reporting is not None:
        event_reporting = args.coordinator_event_reporting
    elif deprecated_reporting is not None:
        logger.warning(
            "--coordinator-l2-event-reporting is deprecated, "
            "use --coordinator-event-reporting instead"
        )
        event_reporting = deprecated_reporting
    else:
        event_reporting = os.getenv(
            "LMCACHE_COORDINATOR_EVENT_REPORTING", ""
        ).lower() in ("1", "true", "yes")

    deprecated_flush = getattr(args, "coordinator_l2_event_flush_interval", None)
    if args.coordinator_event_flush_interval is not None:
        event_flush_interval = args.coordinator_event_flush_interval
    elif deprecated_flush is not None:
        logger.warning(
            "--coordinator-l2-event-flush-interval is deprecated, "
            "use --coordinator-event-flush-interval instead"
        )
        event_flush_interval = deprecated_flush
    else:
        raw = os.getenv("LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL")
        if raw:
            try:
                event_flush_interval = float(raw)
            except ValueError as exc:
                raise ValueError(
                    "LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL is not a number: %r" % raw
                ) from exc
        else:
            event_flush_interval = 1.0
    if not math.isfinite(event_flush_interval) or event_flush_interval <= 0:
        raise ValueError(
            "coordinator event flush interval must be a finite number > 0, "
            "got %s" % event_flush_interval
        )

    return CoordinatorConfig(
        url=url,
        advertise_ip=advertise_ip,
        heartbeat_interval=heartbeat_interval,
        event_reporting=event_reporting,
        event_flush_interval=event_flush_interval,
    )

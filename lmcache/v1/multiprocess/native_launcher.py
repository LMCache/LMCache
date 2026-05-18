# SPDX-License-Identifier: Apache-2.0
"""Helpers for launching the native C++ LMCache MP server."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast
import argparse
import json
import os
import shutil
import subprocess
import sys

_CONFIG_FILE_ENV = "LMCACHE_CONFIG_FILE"
_SERVER_ARG_DEFAULTS = {
    "chunk_size": 256,
}
_NATIVE_ENGINE_CONFIG_ENV_NAMES = {
    "LMCACHE_CHUNK_SIZE",
    "LMCACHE_MAX_LOCAL_CPU_SIZE",
    "LMCACHE_CACHE_POLICY",
    "LMCACHE_LOCAL_DISK",
}
_NATIVE_UNSUPPORTED_ENGINE_CONFIG_KEYS = {
    "enable_blending": "blend engine mode",
    "enable_p2p": "P2P mode",
    "enable_pd": "PD mode",
    "external_lookup_client": "external lookup clients",
    "gds_path": "GDS storage",
    "internal_api_server_enabled": "internal API server",
    "maru_path": "Maru storage",
    "nixl_backends": "NIXL storage",
    "remote_config_url": "remote config service",
    "remote_storage_plugins": "remote storage plugins",
    "remote_url": "remote storage",
    "runtime_plugin_locations": "runtime plugins",
    "storage_plugins": "storage plugins",
    "transfer_channel": "transfer channels",
}
_NATIVE_ENGINE_CONFIG_ENV_NAMES.update(
    f"LMCACHE_{key.upper()}" for key in _NATIVE_UNSUPPORTED_ENGINE_CONFIG_KEYS
)
_NATIVE_ENGINE_CONFIG_ENV_NAMES.add("LMCACHE_LOCAL_CPU")


class _EngineConfigLike(Protocol):
    _config_definitions: Mapping[str, Mapping[str, object]]
    _user_set_keys: set[str]

    def to_dict(self) -> dict[str, object]: ...


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def native_source_dir() -> Path:
    return repo_root() / "LMCache-mp-cpp"


def native_build_dir(*, enable_cuda: bool = False) -> Path:
    build_name = "cmake-cuda" if enable_cuda else "cmake"
    return native_source_dir() / ".build" / build_name


def native_binary_path(*, enable_cuda: bool = False) -> Path:
    name = "lmcache-mp-server-native"
    suffix = ".exe" if sys.platform == "win32" else ""
    return native_build_dir(enable_cuda=enable_cuda) / f"{name}{suffix}"


def packaged_native_binary_path(*, enable_cuda: bool = False) -> Path:
    name = (
        "lmcache-mp-server-native-cuda" if enable_cuda else "lmcache-mp-server-native"
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    return repo_root() / "lmcache" / "bin" / f"{name}{suffix}"


def _source_is_newer(binary: Path, source_dir: Path) -> bool:
    if not binary.exists():
        return True
    binary_mtime = binary.stat().st_mtime
    for path in source_dir.rglob("*"):
        if ".build" in path.parts or not path.is_file():
            continue
        if (
            path.suffix in {".cpp", ".h", ".txt"}
            and path.stat().st_mtime > binary_mtime
        ):
            return True
    return False


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _config_file_path(args: argparse.Namespace) -> str | None:
    path = getattr(args, "config_file", None) or os.environ.get(_CONFIG_FILE_ENV)
    return str(path) if path else None


@contextmanager
def _config_file_precedence_env(config_file_path: str | None) -> Iterator[None]:
    """Keep engine config env vars from overriding an explicit config file."""
    if not config_file_path:
        yield
        return

    saved = {
        name: value for name, value in os.environ.items() if name.startswith("LMCACHE_")
    }
    for name in saved:
        os.environ.pop(name, None)

    try:
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("LMCACHE_") and name not in saved:
                os.environ.pop(name, None)
        os.environ.update(saved)


def _native_config_requested(args: argparse.Namespace) -> bool:
    if _config_file_path(args):
        return True
    return any(
        os.environ.get(name) is not None for name in _NATIVE_ENGINE_CONFIG_ENV_NAMES
    )


def _set_arg_if_missing_or_default(
    args: argparse.Namespace,
    name: str,
    value: object,
    default: object | None = None,
) -> None:
    current = getattr(args, name, None)
    if current is None or current == default:
        setattr(args, name, value)


def _assert_native_engine_config_supported(config: object) -> None:
    engine_config = cast(_EngineConfigLike, config)
    user_set_keys = set(getattr(engine_config, "_user_set_keys", set()))
    config_dict = engine_config.to_dict()
    definitions = engine_config._config_definitions

    for key, reason in _NATIVE_UNSUPPORTED_ENGINE_CONFIG_KEYS.items():
        if key not in user_set_keys:
            continue
        value = config_dict.get(key)
        default = definitions.get(key, {}).get("default")
        if value != default:
            raise ValueError(
                f"native MP does not support {reason} from {_CONFIG_FILE_ENV} "
                f"yet; key {key!r} was set to {value!r}. Use --python for "
                "this server mode"
            )

    if "local_cpu" in user_set_keys and config_dict.get("local_cpu") is False:
        raise ValueError(
            "native MP requires a local CPU/DRAM tier; "
            "local_cpu=false from LMCache config is not supported yet. "
            "Use --python for this server mode"
        )


def _fs_l2_adapter_jsons(local_disk: str | None) -> list[str]:
    if not local_disk:
        return []
    paths = [part.strip() for part in str(local_disk).split(",") if part.strip()]
    return [
        json.dumps({"type": "fs", "base_path": path}, sort_keys=True) for path in paths
    ]


def apply_lmcache_engine_config_to_args(
    args: argparse.Namespace,
    *,
    validate_native: bool = False,
) -> None:
    """Seed supported ``lmcache server`` args from LMCache engine config.

    Native mode still validates unsupported server behaviors before exec. This
    helper only translates config values that have a direct native argv
    equivalent today.
    """
    if not _native_config_requested(args):
        return

    # First Party
    from lmcache.v1.config import load_engine_config_with_overrides

    config_file_path = _config_file_path(args)
    with _config_file_precedence_env(config_file_path):
        config = cast(
            _EngineConfigLike,
            load_engine_config_with_overrides(config_file_path=config_file_path),
        )
    if validate_native:
        _assert_native_engine_config_supported(config)

    config_dict = config.to_dict()
    user_set_keys = set(getattr(config, "_user_set_keys", set()))
    if "chunk_size" in user_set_keys:
        _set_arg_if_missing_or_default(
            args,
            "chunk_size",
            config_dict["chunk_size"],
            _SERVER_ARG_DEFAULTS["chunk_size"],
        )
    if (
        "max_local_cpu_size" in user_set_keys
        or getattr(args, "l1_size_gb", None) is None
    ):
        _set_arg_if_missing_or_default(
            args,
            "l1_size_gb",
            config_dict["max_local_cpu_size"],
        )
    if (
        "cache_policy" in user_set_keys
        or getattr(args, "eviction_policy", None) is None
    ):
        _set_arg_if_missing_or_default(
            args,
            "eviction_policy",
            config_dict["cache_policy"],
        )

    if not getattr(args, "l2_adapter", None):
        local_disk = config_dict.get("local_disk")
        adapters = _fs_l2_adapter_jsons(
            local_disk if isinstance(local_disk, str) else None
        )
        if adapters:
            args.l2_adapter = adapters


def ensure_native_binary(force: bool = False, *, enable_cuda: bool = False) -> Path:
    configured_binary = os.environ.get(
        "LMCACHE_MP_NATIVE_CUDA_BINARY" if enable_cuda else "LMCACHE_MP_NATIVE_BINARY"
    )
    if configured_binary:
        path = Path(configured_binary)
        if not path.exists():
            raise FileNotFoundError(
                f"configured native MP binary points to missing file: {path}"
            )
        return path

    packaged_binary = packaged_native_binary_path(enable_cuda=enable_cuda)
    if packaged_binary.exists():
        return packaged_binary

    source_dir = native_source_dir()
    if not source_dir.exists():
        found = shutil.which("lmcache-mp-server-native")
        if found is not None:
            return Path(found)
        raise FileNotFoundError(
            "Native MP source directory is unavailable and "
            "lmcache-mp-server-native is not on PATH."
        )

    binary = native_binary_path(enable_cuda=enable_cuda)
    if force or _source_is_newer(binary, source_dir):
        build_dir = native_build_dir(enable_cuda=enable_cuda)
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_args = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLMCACHE_BUILD_NATIVE_MP=ON",
        ]
        if enable_cuda:
            cmake_args.append("-DLMCACHE_ENABLE_CUDA=ON")
        subprocess.run(
            cmake_args,
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "lmcache-mp-server-native",
            ],
            check=True,
        )
    return binary


def _validate_native_supported_args(args: argparse.Namespace) -> None:
    def reject_non_default(name: str, default: object, flag: str) -> None:
        value = getattr(args, name, default)
        if value != default:
            raise ValueError(
                f"native MP does not support {flag}={value!r} yet; "
                f"only the Python default {default!r} is accepted. "
                "Use --python for this mode"
            )

    if getattr(args, "l1_size_gb", None) is None:
        raise ValueError(
            "native MP requires --l1-size-gb or a supported "
            "max_local_cpu_size value from --config-file/LMCACHE_CONFIG_FILE"
        )
    if getattr(args, "eviction_policy", None) is None:
        raise ValueError(
            "native MP requires --eviction-policy or a supported cache_policy "
            "value from --config-file/LMCACHE_CONFIG_FILE"
        )

    hash_algorithm = getattr(args, "hash_algorithm", "blake3")
    if hash_algorithm != "blake3":
        raise ValueError(
            "native MP currently supports only --hash-algorithm blake3; "
            f"got {hash_algorithm!r}"
        )

    engine_type = getattr(args, "engine_type", "default")
    if engine_type != "default":
        raise ValueError(
            "native MP currently supports only --engine-type default; "
            f"got {engine_type!r}"
        )

    eviction_policy = getattr(args, "eviction_policy", "LRU")
    if eviction_policy != "LRU":
        raise ValueError(
            "native MP currently supports only --eviction-policy LRU; "
            f"got {eviction_policy!r}"
        )

    runtime_plugin_locations = getattr(args, "runtime_plugin_locations", []) or []
    if runtime_plugin_locations:
        raise ValueError(
            "native MP does not support --runtime-plugin-locations yet; "
            "use --python for runtime plugins"
        )

    runtime_plugin_config = getattr(args, "runtime_plugin_config", "{}")
    if runtime_plugin_config not in (None, "", "{}"):
        raise ValueError(
            "native MP does not support --runtime-plugin-config yet; "
            "use --python for runtime plugins"
        )

    max_workers = getattr(args, "max_workers", 1)
    max_cpu_workers = getattr(args, "max_cpu_workers", None)
    if max_cpu_workers is not None and max_cpu_workers != max_workers:
        raise ValueError(
            "native MP currently uses one worker pool and does not support "
            "separate --max-cpu-workers; set it equal to --max-workers or "
            "use --python"
        )
    max_gpu_workers = getattr(args, "max_gpu_workers", None)
    if max_gpu_workers is not None and max_gpu_workers != max_workers:
        raise ValueError(
            "native MP currently uses one worker pool and does not support "
            "separate --max-gpu-workers; set it equal to --max-workers or "
            "use --python"
        )

    reject_non_default("l1_use_lazy", True, "--no-l1-use-lazy")
    reject_non_default("l1_init_size_gb", 20, "--l1-init-size-gb")
    reject_non_default("l1_align_bytes", 4096, "--l1-align-bytes")
    reject_non_default("l1_write_ttl_seconds", 600, "--l1-write-ttl-seconds")
    reject_non_default("l1_read_ttl_seconds", 300, "--l1-read-ttl-seconds")
    reject_non_default(
        "eviction_trigger_watermark", 0.8, "--eviction-trigger-watermark"
    )
    reject_non_default("eviction_ratio", 0.2, "--eviction-ratio")
    reject_non_default("l2_store_policy", "default", "--l2-store-policy")
    reject_non_default("l2_prefetch_policy", "default", "--l2-prefetch-policy")
    reject_non_default("l2_prefetch_max_in_flight", 8, "--l2-prefetch-max-in-flight")
    reject_non_default("disable_observability", False, "--disable-observability")
    reject_non_default("disable_metrics", False, "--disable-metrics")
    reject_non_default("disable_logging", False, "--disable-logging")
    reject_non_default("enable_tracing", False, "--enable-tracing")
    reject_non_default("otlp_endpoint", None, "--otlp-endpoint")
    reject_non_default("event_bus_queue_size", 10_000, "--event-bus-queue-size")
    reject_non_default("prometheus_port", 9090, "--prometheus-port")
    reject_non_default("metrics_sample_rate", 0.01, "--metrics-sample-rate")
    reject_non_default("service_instance_id", None, "--service-instance-id")
    reject_non_default("lookup_hash_log_dir", "", "--lookup-hash-log-dir")
    reject_non_default(
        "lookup_hash_log_rotation_interval",
        6 * 3600,
        "--lookup-hash-log-rotation-interval",
    )
    reject_non_default(
        "lookup_hash_log_rotation_max_size",
        100 * 1024 * 1024,
        "--lookup-hash-log-rotation-max-size",
    )
    reject_non_default("lookup_hash_log_max_files", 100, "--lookup-hash-log-max-files")
    reject_non_default("trace_level", None, "--trace-level")
    reject_non_default("trace_output", None, "--trace-output")


def native_argv_from_args(args: argparse.Namespace) -> list[str]:
    apply_lmcache_engine_config_to_args(args, validate_native=True)
    _validate_native_supported_args(args)
    force_no_cuda = bool(getattr(args, "native_no_cuda", False)) or _env_flag_enabled(
        "LMCACHE_MP_NATIVE_NO_CUDA"
    )
    force_cuda = bool(getattr(args, "native_cuda", False)) or _env_flag_enabled(
        "LMCACHE_MP_NATIVE_CUDA"
    )
    if force_no_cuda and force_cuda:
        raise ValueError(
            "native MP received both CUDA and no-CUDA native launch requests; "
            "choose either --native-cuda or --native-no-cuda"
        )
    enable_cuda = not force_no_cuda
    if force_cuda:
        enable_cuda = True
    binary = ensure_native_binary(enable_cuda=enable_cuda)
    argv = [
        str(binary),
        "--host",
        str(args.host),
        "--port",
        str(args.port),
        "--http-host",
        str(args.http_host),
        "--http-port",
        str(args.http_port),
        "--chunk-size",
        str(args.chunk_size),
        "--l1-size-gb",
        str(args.l1_size_gb),
        "--eviction-policy",
        str(args.eviction_policy),
        "--max-workers",
        str(args.max_workers),
    ]
    if getattr(args, "max_cpu_workers", None) is not None:
        argv.extend(["--max-cpu-workers", str(args.max_cpu_workers)])
    if getattr(args, "max_gpu_workers", None) is not None:
        argv.extend(["--max-gpu-workers", str(args.max_gpu_workers)])

    log_level = getattr(args, "log_level", None)
    if log_level:
        argv.extend(["--log-level", str(log_level)])

    native_disk_path = getattr(args, "native_disk_path", None)
    if native_disk_path:
        argv.extend(["--native-disk-path", str(native_disk_path)])

    for adapter in getattr(args, "l2_adapter", []) or []:
        argv.extend(["--l2-adapter", adapter])
    return argv


def run_native_server(args: argparse.Namespace) -> None:
    try:
        argv = native_argv_from_args(args)
    except ValueError as exc:
        print(f"invalid native MP arguments: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    # First Party
    from lmcache import utils

    env = os.environ.copy()
    env.pop(_CONFIG_FILE_ENV, None)
    env["LMCACHE_NATIVE_VERSION"] = utils.VERSION
    env["LMCACHE_NATIVE_COMMIT_ID"] = utils.COMMIT_ID
    os.execve(argv[0], argv, env)

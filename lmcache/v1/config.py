# SPDX-License-Identifier: Apache-2.0
"""
LMCache Engine Configuration

Configuration system for LMCache Engine that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Provides convenient access to configuration values
"""

# Standard
from typing import Any, Dict, Optional
import json
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config_base import (
    _parse_local_disk,
    _parse_quoted_string,
    _resolve_config_aliases,
    _to_bool,
    _to_float_list,
    _to_int_list,
    _to_str_list,
    create_config_class,
    load_config_with_overrides,
)

logger = init_logger(__name__)


# Configuration aliases and deprecated mappings
_CONFIG_ALIASES = {
    # Maps deprecated names to current names
    "enable_xpyd": "enable_pd",
    "nixl_peer_host": "pd_peer_host",
    "nixl_peer_init_port": "pd_peer_init_port",
    "nixl_peer_alloc_port": "pd_peer_alloc_port",
    "nixl_proxy_host": "pd_proxy_host",
    "nixl_proxy_port": "pd_proxy_port",
    "nixl_buffer_size": "pd_buffer_size",
    "nixl_role": "pd_role",
    "controller_url": "controller_pull_url",
    "lmcache_worker_port": "lmcache_worker_ports",
    "plugin_locations": "runtime_plugin_locations",
    "external_backends": "storage_plugins",
}

_DEPRECATED_CONFIGS = {
    # Maps deprecated names to warning messages
    "nixl_peer_port": "nixl_peer_port is deprecated, use nixl_receiver_port instead",
    "plugin_locations": (
        "plugin_locations is deprecated, use runtime_plugin_locations instead"
    ),
    "external_backends": (
        "external_backends is deprecated, use storage_plugins instead"
    ),
}

# Single configuration definition center - add new config items only here
_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    # Basic configurations
    "chunk_size": {"type": int, "default": 256, "env_converter": int},
    "local_cpu": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "max_local_cpu_size": {"type": float, "default": 5.0, "env_converter": float},
    "reserve_local_cpu_size": {"type": float, "default": 0.0, "env_converter": float},
    "local_disk": {
        "type": Optional[str],
        "default": None,
        "env_converter": _parse_local_disk,
    },
    "max_local_disk_size": {"type": float, "default": 0.0, "env_converter": float},
    "remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "remote_serde": {"type": Optional[str], "default": "naive", "env_converter": str},
    # Feature toggles
    "use_layerwise": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "save_decode_cache": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pre_caching_hash_algorithm": {
        "type": str,
        "default": "builtin",
        "env_converter": str,
    },
    # Blending configurations
    "enable_blending": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "blend_recompute_ratios": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_thresholds": {
        "type": Optional[list[float]],
        "default": None,
        "env_converter": _to_float_list,
    },
    "blend_check_layers": {
        "type": list[int],
        "default": None,
        "env_converter": _to_int_list,
    },
    "blend_min_tokens": {"type": int, "default": 256, "env_converter": int},
    "blend_special_str": {"type": str, "default": " # # ", "env_converter": str},
    # P2P configurations
    "enable_p2p": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "p2p_host": {"type": Optional[str], "default": None, "env_converter": str},
    "p2p_init_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "p2p_lookup_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # Controller configurations
    "enable_controller": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "lmcache_instance_id": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_pull_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "controller_reply_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "lmcache_worker_ports": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "lmcache_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    # LMCache Worker heartbeat
    # the lmcache_worker_heartbeat_delay_time means that delay a period of time
    # before starting, ensures that the heartbeat starts working only after the
    # service is fully ready(such as, waiting register).
    "lmcache_worker_heartbeat_delay_time": {
        "type": int,
        "default": 10,
        "env_converter": int,
    },
    # the lmcache_worker_heartbeat_time means that sending heartbeat periodically.
    "lmcache_worker_heartbeat_time": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # PD-related configurations
    "enable_pd": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "pd_role": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_buffer_size": {"type": Optional[int], "default": None, "env_converter": int},
    "pd_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "pd_peer_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_peer_init_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_peer_alloc_port": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "pd_proxy_host": {"type": Optional[str], "default": None, "env_converter": str},
    "pd_proxy_port": {"type": Optional[int], "default": None, "env_converter": int},
    # Transfer-related configurations
    "transfer_channel": {"type": Optional[str], "default": None, "env_converter": str},
    # Nixl-related configurations
    "nixl_backends": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    "nixl_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "nixl_buffer_device": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    # Storage paths
    "gds_path": {"type": Optional[str], "default": None, "env_converter": str},
    "cufile_buffer_size": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    # Other configurations
    # (Deprecated) The url of the actual remote lmcache instance for auditing.
    # Please use extra_config['audit_actual_remote_url'] instead.
    "audit_actual_remote_url": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "internal_api_server_host": {
        "type": str,
        "default": "0.0.0.0",
        "env_converter": str,
    },
    "extra_config": {
        "type": Optional[dict],
        "default": None,
        "env_converter": lambda x: x
        if isinstance(x, dict)
        else json.loads(x)
        if x
        else None,
    },
    "save_unfull_chunk": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "blocking_timeout_secs": {"type": int, "default": 10, "env_converter": int},
    "external_lookup_client": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "py_enable_gc": {
        "type": bool,
        "default": True,
        "env_converter": _to_bool,
    },
    "cache_policy": {
        "type": str,
        "default": "LRU",
        "env_converter": str,
    },
    "numa_mode": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "enable_async_loading": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_enabled": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "internal_api_server_port_start": {
        "type": int,
        "default": 6999,
        "env_converter": int,
    },
    "priority_limit": {
        "type": Optional[int],
        "default": None,
        "env_converter": int,
    },
    "internal_api_server_include_index_list": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "internal_api_server_socket_path_prefix": {
        "type": Optional[str],
        "default": None,
        "env_converter": str,
    },
    "runtime_plugin_locations": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": lambda x: x if isinstance(x, list) else [x] if x else [],
    },
    "storage_plugins": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lookup client configurations
    "lookup_timeout_ms": {
        "type": int,
        "default": 3000,
        "env_converter": int,
    },
    "hit_miss_ratio": {
        "type": Optional[float],
        "default": None,
        "env_converter": float,
    },
    "lookup_server_worker_ids": {
        "type": Optional[list[int]],
        "default": None,
        "env_converter": _to_int_list,
    },
    "enable_scheduler_bypass_lookup": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "script_allowed_imports": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": _to_str_list,
    },
    # Lazy memory allocator configurations
    "enable_lazy_memory_allocator": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": (
            "Enable lazy memory allocator to reduce initial memory footprint. "
            "Memory is allocated on-demand and expanded automatically when needed."
        ),
    },
    "lazy_memory_initial_ratio": {
        "type": float,
        "default": 0.2,
        "env_converter": float,
        "description": (
            "Initial memory allocation ratio (0.0-1.0). "
            "Determines the percentage of target memory size to allocate at startup. "
            "Default is 0.2 (20%)."
        ),
    },
    "lazy_memory_expand_trigger_ratio": {
        "type": float,
        "default": 0.5,
        "env_converter": float,
        "description": (
            "Memory usage ratio (0.0-1.0) that triggers automatic expansion. "
            "When memory usage exceeds this threshold, expansion is triggered. "
            "Default is 0.5 (50%)."
        ),
    },
    "lazy_memory_step_ratio": {
        "type": float,
        "default": 0.1,
        "env_converter": float,
        "description": (
            "Memory expansion step ratio (0.0-1.0). "
            "Determines the percentage of target memory size to add in each expansion. "
            "Default is 0.1 (10%)."
        ),
    },
    "lazy_memory_safe_size": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": (
            "Safe threshold size in GB. Lazy allocator is only enabled when "
            "max_local_cpu_size exceeds this value. Default is 0.0 GB (always enabled)."
        ),
    },
    # Chunk statistics configurations
    "enable_chunk_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Enable chunk statistics tracking.",
    },
    "chunk_statistics_auto_start_statistics": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
        "description": "Auto-start statistics on init.",
    },
    "chunk_statistics_auto_exit_timeout_hours": {
        "type": float,
        "default": 0.0,
        "env_converter": float,
        "description": "Auto-stop timeout in hours (0=disabled).",
    },
    "chunk_statistics_auto_exit_target_unique_chunks": {
        "type": int,
        "default": 0,
        "env_converter": int,
        "description": "Auto-stop at target unique chunks.",
    },
    "chunk_statistics_strategy": {
        "type": str,
        "default": "memory_bloom_filter",
        "env_converter": str,
        "description": "Recording strategy: memory_bloom_filter or file_hash.",
    },
    # KV events configuration
    "enable_kv_events": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # TODO(chunxiaozheng): remove this after VLLMPagedMemGPUConnectorV3 is stable
    "use_gpu_connector_v3": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    # Memory management configurations
    "pin_timeout_sec": {
        "type": int,
        "default": 300,
        "env_converter": int,
        "description": (
            "Maximum duration in seconds that a memory object can remain pinned. "
            "If a pinned object exceeds this timeout, it will be forcibly unpinned "
            "by the PinMonitor to prevent memory leaks. Default is 300 seconds."
        ),
    },
    "pin_check_interval_sec": {
        "type": int,
        "default": 30,
        "env_converter": int,
        "description": (
            "Interval in seconds between PinMonitor timeout checks. "
            "The background thread periodically scans all pinned objects at this "
            "interval to detect and handle timeouts. Default is 30 seconds."
        ),
    },
}


# Specialized methods that are unique to LMCacheEngineConfig
def _validate_config(self):
    """Validate configuration"""

    # needed for the old async serializer implementation
    # # auto-adjust save_unfull_chunk for async loading to prevent CPU fragmentation
    # if self.enable_async_loading:
    #     logger.warning(
    #         "Automatically setting save_unfull_chunk=False because "
    #         "enable_async_loading=True or use_layerwise=True to prevent "
    #         "CPU memory fragmentation"
    #     )
    #     self.save_unfull_chunk = False

    if self.enable_blending:
        if not self.save_unfull_chunk:
            logger.warning(
                "Automatically setting save_unfull_chunk=True because "
                "enable_blending=True"
            )
            self.save_unfull_chunk = True

    if self.enable_p2p:
        assert self.enable_controller
        assert self.controller_pull_url is not None
        assert self.controller_reply_url is not None
        assert self.lmcache_worker_ports is not None
        assert self.p2p_host is not None
        assert self.p2p_init_ports is not None
        assert self.p2p_lookup_ports is not None
        assert self.transfer_channel is not None

    enable_nixl_storage = self.extra_config is not None and self.extra_config.get(
        "enable_nixl_storage"
    )
    if self.enable_pd:
        assert self.pd_role is not None
        assert self.pd_buffer_size is not None
        assert self.pd_buffer_device is not None

        assert self.remote_url is None, "PD only supports remote_url=None"
        assert self.save_decode_cache is False, (
            "PD only supports save_decode_cache=False"
        )
        assert self.enable_p2p is False, "PD only supports enable_p2p=False"

        # PD requires save_unfull_chunk=True for complete KV cache transfer
        # from prefill node to decode node. Without this, partial chunks would
        # be discarded, causing incomplete KV cache transfer and wrong results
        # on the decode node.
        if not self.save_unfull_chunk:
            logger.warning(
                "PD (Peer-to-Peer Disaggregation) requires save_unfull_chunk=True "
                "for complete KV cache transfer. Automatically setting "
                "save_unfull_chunk=True."
            )
            self.save_unfull_chunk = True
        else:
            logger.info(
                "PD mode enabled with save_unfull_chunk=True - all KV cache "
                "including partial chunks will be transferred to decode node"
            )

    if enable_nixl_storage:
        assert self.extra_config.get("nixl_backend") is not None
        assert self.extra_config.get("nixl_pool_size") is not None
        assert self.nixl_buffer_size is not None
        assert self.nixl_buffer_device is not None

    return self


def _log_config(self):
    """Log configuration"""
    config_dict = {}
    for name in _CONFIG_DEFINITIONS:
        value = getattr(self, name)
        if name in ["max_local_cpu_size", "max_local_disk_size"]:
            value = f"{value} GB"
        config_dict[name] = value

    logger.info(f"LMCache Configuration: {config_dict}")
    return self


def _get_extra_config_value(self, key, default_value=None):
    if hasattr(self, "extra_config") and self.extra_config is not None:
        return self.extra_config.get(key, default_value)
    else:
        return default_value


def _get_lmcache_worker_ids(self, use_mla, world_size):
    if not self.lmcache_worker_ids:
        # if mla is not enabled, return all worker ids, which means start
        # lmcache worker on all ranks as default;
        # if mla is enabled, return [0], which means start lmcache
        # worker on worker 0 as default.
        return [0] if use_mla else list(range(world_size))

    # check the input
    for worker_id in self.lmcache_worker_ids:
        assert -1 < worker_id < world_size
    return self.lmcache_worker_ids


def _get_lookup_server_worker_ids(self, use_mla, world_size):
    if not self.lookup_server_worker_ids:
        # if mla is not enabled, return all worker ids, which means start
        # lookup server on all worker as default;
        # if mla is enabled, return [0], which means start lookup
        # server on worker 0 as default.
        return [0] if use_mla else list(range(world_size))

    # check the input
    for worker_id in self.lookup_server_worker_ids:
        assert -1 < worker_id < world_size
    return self.lookup_server_worker_ids


def _from_legacy(cls, **kwargs):
    """Create configuration from legacy format"""
    backend = kwargs.pop("backend", "cpu")

    # Define backend mappings
    backend_configs = {
        "cpu": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
            "max_local_disk_size": 0,
            "remote_url": None,
        },
        "local_disk": {
            "local_cpu": False,
            "max_local_cpu_size": 3,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 2,
            "remote_url": None,
        },
        "local_cpu_disk": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
            "remote_url": None,
        },
        "remote": {"local_cpu": False, "max_local_cpu_size": 2, "local_disk": None},
        "local_cpu_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": None,
        },
        "local_disk_remote": {
            "local_cpu": False,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
        "local_cpu_disk_remote": {
            "local_cpu": True,
            "max_local_cpu_size": 2,
            "local_disk": "local/disk_test/local_disk/",
            "max_local_disk_size": 5,
        },
    }

    if backend not in backend_configs:
        raise ValueError(f"Invalid backend: {backend}")

    # Merge configurations
    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in backend_configs[backend]:
            config_values[name] = backend_configs[backend][name]
        elif name in kwargs:
            config_values[name] = kwargs[name]
        else:
            config_values[name] = config["default"]

    instance = cls(**config_values)
    instance.validate()
    return instance


def _update_config_from_env(self):
    """Update an existing config object with environment variable configurations."""

    def get_env_name(attr_name: str) -> str:
        return f"LMCACHE_{attr_name.upper()}"

    # Collect environment variables
    env_config = {}
    for name in _CONFIG_DEFINITIONS:
        env_name = get_env_name(name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[name] = env_value

    # Handle deprecated environment variables
    for deprecated_name, new_name in _CONFIG_ALIASES.items():
        env_name = get_env_name(deprecated_name)
        env_value = os.getenv(env_name)
        if env_value is not None:
            env_config[deprecated_name] = env_value

    # Resolve aliases and handle deprecated configurations
    resolved_config = _resolve_config_aliases(env_config, "environment variables")

    # Update config object with environment values
    for name, config in _CONFIG_DEFINITIONS.items():
        if name in resolved_config:
            try:
                # Parse quoted strings and handle escape characters
                raw_value = resolved_config[name]  # Keep original value for logging
                value = _parse_quoted_string(raw_value)
                converted_value = config["env_converter"](value)
                setattr(self, name, converted_value)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to parse {get_env_name(name)}={raw_value!r}: {e}"
                )
                # Keep existing value if conversion fails
    self.validate()
    return self


def _validate_and_set_config_value(config, config_key, value):
    """Validate and set configuration value"""
    if not hasattr(config, config_key):
        logger.warning(f"Config key '{config_key}' does not exist in configuration")
        return False

    try:
        setattr(config, config_key, value)
        return True
    except Exception as e:
        logger.error(
            f"Failed to set config item '{config_key}' with value {value}: {e}"
        )
        return False


# Create configuration class using the base utility
LMCacheEngineConfig = create_config_class(
    config_name="LMCacheEngineConfig",
    config_definitions=_CONFIG_DEFINITIONS,
    config_aliases=_CONFIG_ALIASES,
    deprecated_configs=_DEPRECATED_CONFIGS,
    namespace_extras={
        "validate": _validate_config,
        "log_config": _log_config,
        "get_extra_config_value": _get_extra_config_value,
        "get_lmcache_worker_ids": _get_lmcache_worker_ids,
        "get_lookup_server_worker_ids": _get_lookup_server_worker_ids,
        "from_legacy": classmethod(_from_legacy),
    },
)


def load_engine_config_with_overrides(
    config_file_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "LMCacheEngineConfig":  # type: ignore[valid-type]
    """
    Load engine configuration with support for file, env vars, and overrides.

    This function uses the generic load_config_with_overrides utility from
    config_base.py to reduce code duplication.

    Args:
        config_file_path: Optional direct path to config file
        overrides: Optional dictionary of configuration overrides

    Returns:
        Loaded and validated LMCacheEngineConfig instance
    """

    return load_config_with_overrides(
        config_class=LMCacheEngineConfig,
        config_file_env_var="LMCACHE_CONFIG_FILE",
        config_file_path=config_file_path,
        overrides=overrides,
    )

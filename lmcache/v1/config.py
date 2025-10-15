# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Optional, Union
import ast
import json
import os
import re
import uuid

# Third Party
import yaml

# First Party
from lmcache.logging import init_logger
import lmcache.config as orig_config

logger = init_logger(__name__)


def _parse_local_disk(local_disk) -> Optional[str]:
    match local_disk:
        case None:
            local_disk_path = None
        case path if re.match(r"file://(.*)/", path):
            local_disk_path = path[7:]
        case _:
            local_disk_path = local_disk
    return local_disk_path


def _to_int_list(
    value: Optional[Union[str, int, list[Any]]],
) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, int):
        return [value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def _to_float_list(
    value: Optional[Union[str, float, list[Any]]],
) -> Optional[list[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, float):
        return [value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [float(p) for p in parts]


def _to_str_list(
    value: Optional[Union[str, list[str]]],
) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [p for p in parts]


def _to_bool(
    value: Optional[Union[bool, int, str]],
) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1"]


def _parse_quoted_string(value: str) -> str:
    """Parse a string that may be surrounded by quotes and handle escape characters.

    Args:
        value: The input string that may be quoted

    Returns:
        The unquoted string with escape characters properly handled
    """
    if not value:
        return value

    value = value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        try:
            evaluated = ast.literal_eval(value)
            if isinstance(evaluated, str):
                return evaluated
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, it's not a valid Python literal.
            # Fall back to simply stripping the outer quotes.
            return value[1:-1]

    return value


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
}

_DEPRECATED_CONFIGS = {
    # Maps deprecated names to warning messages
    "nixl_peer_port": "nixl_peer_port is deprecated, use nixl_receiver_port instead",
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
    "weka_path": {"type": Optional[str], "default": None, "env_converter": str},
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
        "default": True,
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
    "plugin_locations": {
        "type": Optional[list[str]],
        "default": None,
        "env_converter": lambda x: x if isinstance(x, list) else [x] if x else [],
    },
    "external_backends": {
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
}


def _resolve_config_aliases(config_dict: dict, source: str) -> dict:
    """Resolve configuration aliases and handle deprecated configurations."""
    resolved = {}

    # Process each key in the input
    for key, value in config_dict.items():
        if key in _DEPRECATED_CONFIGS:
            # Log deprecation warning
            logger.warning(f"{_DEPRECATED_CONFIGS[key]} (source: {source})")

            # Map to new key if alias exists
            if key in _CONFIG_ALIASES:
                new_key = _CONFIG_ALIASES[key]
                resolved[new_key] = value
            else:
                # Keep deprecated key for backward compatibility
                resolved[key] = value
        elif key in _CONFIG_DEFINITIONS:
            # Valid configuration key
            resolved[key] = value
        else:
            # Unknown configuration key
            logger.warning(f"Unknown configuration key: {key} (source: {source})")

    return resolved


# Dynamically create configuration class
def _create_config_class():
    """Dynamically create configuration class"""
    # Extract fields from configuration definitions
    fields_dict = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        fields_dict[name] = (config["type"], config["default"])

    # Create class using make_dataclass
    # Standard
    from dataclasses import make_dataclass

    def _post_init(self):
        # Generate random instance ID if not set
        if not self.lmcache_instance_id:
            self.lmcache_instance_id = f"lmcache_instance_{uuid.uuid4().hex}"
        self.validate()

    cls = make_dataclass(
        "LMCacheEngineConfig",
        [(name, type_, default) for name, (type_, default) in fields_dict.items()],
        namespace={
            "__post_init__": _post_init,
            "validate": _validate_config,
            "log_config": _log_config,
            "to_original_config": _to_original_config,
            "get_extra_config_value": _get_extra_config_value,
            "from_defaults": classmethod(_from_defaults),
            "from_legacy": classmethod(_from_legacy),
            "from_file": classmethod(_from_file),
            "from_env": classmethod(_from_env),
            "update_config_from_env": _update_config_from_env,
            "__str__": lambda self: str(
                {name: getattr(self, name) for name in _CONFIG_DEFINITIONS}
            ),
            "from_dict": classmethod(_from_dict),
            "to_dict": _to_dict,
            "to_json": _to_json,
            "from_json": classmethod(_from_json),
        },
    )
    return cls


def _validate_config(self):
    """Validate configuration"""
    # auto-adjust save_unfull_chunk for async loading to prevent CPU fragmentation
    if self.enable_async_loading or self.use_layerwise:
        logger.warning(
            "Automatically setting save_unfull_chunk=False because "
            "enable_async_loading=True or use_layerwise=True to prevent "
            "CPU memory fragmentation"
        )
        self.save_unfull_chunk = False

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

    if enable_nixl_storage:
        assert self.extra_config.get("nixl_backend") is not None
        assert self.extra_config.get("nixl_path") is not None
        assert self.extra_config.get("nixl_file_pool_size") is not None
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


def _to_original_config(self):
    """Convert to original configuration format"""
    return orig_config.LMCacheEngineConfig(
        chunk_size=self.chunk_size,
        local_device="cpu" if self.local_cpu else "cuda",
        max_local_cache_size=int(self.max_local_cpu_size),
        remote_url=None,
        remote_serde=None,
        pipelined_backend=False,
        save_decode_cache=self.save_decode_cache,
        enable_blending=self.enable_blending,
        blend_recompute_ratio=0.15,
        blend_min_tokens=self.blend_min_tokens,
        blend_separator="[BLEND_SEP]",
        blend_add_special_in_precomp=False,
    )


def _get_extra_config_value(self, key, default_value=None):
    if hasattr(self, "extra_config") and self.extra_config is not None:
        return self.extra_config.get(key, default_value)
    else:
        return default_value


def _from_defaults(cls, **kwargs):
    """Create configuration from defaults"""
    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        config_values[name] = kwargs.get(name, config["default"])

    instance = cls(**config_values)
    return instance


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
    return instance


def _from_file(cls, file_path: str):
    """Load configuration from file"""
    with open(file_path, "r") as fin:
        file_config = yaml.safe_load(fin) or {}

    # Resolve aliases and handle deprecated configurations
    resolved_config = _resolve_config_aliases(file_config, f"file: {file_path}")

    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        value = resolved_config.get(name, config["default"])
        if value is not None:
            value = config["env_converter"](value)

        # Handle local_disk parsing
        if name == "local_disk":
            value = _parse_local_disk(value)

        # Validate remote_url format
        if name == "remote_url" and value is not None:
            if not re.match(r"(.*)://(.*)", value):
                raise ValueError(f"Invalid remote storage url: {value}")

        config_values[name] = value

    instance = cls(**config_values)
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

    return self


def _from_env(cls):
    """Load configuration from environment variables"""
    instance = cls.from_defaults()
    _update_config_from_env(instance)
    return instance


def _from_dict(cls, config_dict: dict):
    """Create configuration from a dictionary."""
    resolved_config = _resolve_config_aliases(config_dict, "dictionary input")
    config_values = {}
    for name, config in _CONFIG_DEFINITIONS.items():
        value = resolved_config.get(name, config["default"])
        if value is not None:
            value = config["env_converter"](value)
        config_values[name] = value
    instance = cls(**config_values)
    return instance


def _to_dict(self):
    """Convert the configuration object into a dictionary."""
    return {name: getattr(self, name) for name in _CONFIG_DEFINITIONS}


def _from_json(cls, json_str: str):
    """Deserialize a JSON string into a configuration object."""
    try:
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        raise


def _to_json(self):
    """Serialize the configuration object to a JSON string."""
    return json.dumps(self.to_dict(), indent=2)


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


# Create configuration class
LMCacheEngineConfig = _create_config_class()

# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional, Tuple
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.request import Request

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig, _validate_and_set_config_value

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"

# Thread-safe singleton storage
_config_instance: Optional[LMCacheEngineConfig] = None
_config_lock = threading.Lock()


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def _fetch_remote_config(
    remote_config_url: str,
    lmcache_app_id: Optional[str],
    config: LMCacheEngineConfig,
    timeout: int = 10,
) -> Optional[dict]:
    """Fetch configuration from remote config service.

    Args:
        remote_config_url: URL of the remote config service.
        lmcache_app_id: Optional app ID to send to the config service.
        config: Current LMCacheEngineConfig to send to the config service.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response from the config service, or None if failed.
    """
    try:
        # Build request payload with current config and env variables
        payload: dict[str, Any] = {
            "current_config": config.to_dict(),
            "env_variables": {},
        }

        # Add lmcache_appId if provided
        if lmcache_app_id:
            payload["appId"] = lmcache_app_id

        # Collect all environment variables
        for key, value in os.environ.items():
            payload["env_variables"][key] = value

        # Prepare and send request
        request_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            remote_config_url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)

    except urllib.error.URLError as e:
        logger.warning(f"Failed to fetch remote config from {remote_config_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse remote config response: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching remote config: {e}")
        return None


def _apply_remote_configs(
    config: LMCacheEngineConfig, remote_response: dict
) -> LMCacheEngineConfig:
    """Apply remote configuration to LMCacheEngineConfig.

    This function extracts the 'configs' field from the remote response
    and applies each config item to the LMCacheEngineConfig instance.

    The expected format of remote_response['configs'] is:
    [
        {"override": true, "key": "config_key", "value": "config_value"},
        ...
    ]

    Args:
        config: LMCacheEngineConfig instance to update.
        remote_response: Response from the remote config service.

    Returns:
        Updated LMCacheEngineConfig instance.
    """
    configs = remote_response.get("configs", [])
    if not configs:
        logger.info("No configs found in remote response")
        return config

    applied_count = 0
    for config_item in configs:
        if not isinstance(config_item, dict):
            logger.warning(f"Invalid config item format: {config_item}")
            continue

        key = config_item.get("key")
        value = config_item.get("value")
        override = config_item.get("override", True)

        if not key:
            logger.warning(f"Config item missing 'key': {config_item}")
            continue

        # Check if the config attribute exists
        if not hasattr(config, key):
            # If the key doesn't exist as a direct attribute, try to store it
            # in extra_config
            if config.extra_config is None:
                config.extra_config = {}
            if override or key not in config.extra_config:
                config.extra_config[key] = value
                logger.info(f"Applied remote config to extra_config: {key}={value}")
                applied_count += 1
            continue

        # Get current value
        current_value = getattr(config, key)

        # Skip if override is False and current value is not None/default
        if not override and current_value is not None:
            logger.info(
                f"Skipping remote config {key} (override=False, "
                f"current value={current_value})"
            )
            continue

        # Try to convert value to appropriate type
        if _validate_and_set_config_value(config, key, value):
            logger.info(f"Applied remote config: {key}={value}")
            applied_count += 1
        else:
            logger.warning(
                f"Failed to apply remote config {key}={value}. Using default value."
            )

    logger.info(f"Applied {applied_count} remote configuration items")
    return config


def lmcache_get_or_create_config() -> LMCacheEngineConfig:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.

    This function is thread-safe and implements singleton pattern,
    ensuring the configuration is loaded only once.

    After loading the configuration, if 'remote_config_url' is configured,
    this function will attempt to fetch additional configuration from the
    remote config service. The current config and LMCACHE environment
    variables will be sent to the service, along with 'lmcache_appId' if set.
    """
    global _config_instance

    # Double-checked locking for thread-safe singleton
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:  # Check again within lock
                if "LMCACHE_CONFIG_FILE" not in os.environ:
                    logger.warning(
                        "No LMCache configuration file is set. Trying to read"
                        " configurations from the environment variables."
                    )
                    logger.warning(
                        "You can set the configuration file through "
                        "the environment variable: LMCACHE_CONFIG_FILE"
                    )
                    _config_instance = LMCacheEngineConfig.from_env()
                else:
                    config_file = os.environ["LMCACHE_CONFIG_FILE"]
                    logger.info(f"Loading LMCache config file {config_file}")
                    _config_instance = LMCacheEngineConfig.from_file(config_file)
                    # Update config from environment variables
                    _config_instance.update_config_from_env()

                # Fetch and apply remote configuration if configured
                remote_config_url = _config_instance.remote_config_url
                if remote_config_url:
                    logger.info(
                        "Fetching remote configuration from %s", remote_config_url
                    )
                    lmcache_app_id = _config_instance.lmcache_app_id
                    remote_response = _fetch_remote_config(
                        remote_config_url, lmcache_app_id, _config_instance
                    )
                    if remote_response:
                        _config_instance = _apply_remote_configs(
                            _config_instance, remote_response
                        )
                    else:
                        logger.warning(
                            "Failed to fetch remote configuration from %s. "
                            "Using local configuration only.",
                            remote_config_url,
                        )
    return _config_instance


def hex_hash_to_int16(s: str) -> int:
    """
    Convert a hex hash string to a 16-bit integer.
    """
    return int(s, 16) & 0xFFFF


def apply_mm_hashes_to_token_ids(
    token_ids: torch.Tensor,
    mm_hashes: list[str],
    mm_positions: list["PlaceholderRange"],
) -> torch.Tensor:
    """
    Overwrite token_ids in-place for multimodal placeholders using
    efficient slice assignments.
    """
    n = token_ids.size(0)
    for hash_str, placeholder in zip(mm_hashes, mm_positions, strict=False):
        start, length = placeholder.offset, placeholder.length
        if start >= n:
            continue
        end = min(start + length, n)
        token_ids[start:end] = hex_hash_to_int16(hash_str)
    return token_ids


def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )


def create_lmcache_metadata(
    vllm_config=None,
    model_config=None,
    parallel_config=None,
    cache_config=None,
    role=None,
):
    """
    Create LMCacheEngineMetadata from vLLM configuration.

    This function extracts common metadata creation logic that was duplicated
    across multiple files.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and
                    cache configs (alternative to individual config parameters)
        model_config: Model configuration (alternative to vllm_config)
        parallel_config: Parallel configuration (alternative to vllm_config)
        cache_config: Cache configuration (alternative to vllm_config)

    Returns:
        tuple: (LMCacheEngineMetadata, LMCacheEngineConfig)
    """
    # Third Party
    # Try to import from old location before merged https://github.com/vllm-project/vllm/pull/26908
    try:
        # Third Party
        from vllm.utils.torch_utils import get_kv_cache_torch_dtype
    except ImportError:
        # Third Party
        from vllm.utils import get_kv_cache_torch_dtype
    # First Party
    from lmcache.config import LMCacheEngineMetadata

    config = lmcache_get_or_create_config()
    # Support both vllm_config object and individual config parameters
    if vllm_config is not None:
        model_cfg = vllm_config.model_config
        parallel_cfg = vllm_config.parallel_config
        cache_cfg = vllm_config.cache_config
    else:
        model_cfg = model_config
        parallel_cfg = parallel_config
        cache_cfg = cache_config

    # Get KV cache dtype
    kv_dtype = get_kv_cache_torch_dtype(cache_cfg.cache_dtype, model_cfg.dtype)

    # Check if MLA is enabled
    use_mla = mla_enabled(model_cfg)

    # Construct KV shape (for memory pool)
    num_layer = model_cfg.get_num_layers(parallel_cfg)
    chunk_size = config.chunk_size
    num_kv_head = model_cfg.get_num_kv_heads(parallel_cfg)
    head_size = model_cfg.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)

    # Create metadata
    metadata = LMCacheEngineMetadata(
        model_cfg.model,
        parallel_cfg.world_size,
        parallel_cfg.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla,
        role,
        served_model_name=model_cfg.served_model_name,
    )

    return metadata, config


def extract_mm_features(
    request: "Request", modify: bool = False
) -> Tuple[list[str], list["PlaceholderRange"]]:
    """
    Normalize multimodal information from a Request into parallel lists.

    This helper reads either:
      1) `request.mm_features` (objects each exposing `.identifier` and
      `.mm_position`), or
      2) legacy fields `request.mm_hashes` and `request.mm_positions`.

    It returns two equally sized lists: the multimodal hash identifiers and their
    corresponding positions. If the request contains no multimodal info, it returns
    `([], [])`.

    Args:
        request (Request): The source object.
        modify (bool):
            Controls copy semantics for the legacy-path return values.
            - If True and legacy fields are used, shallow-copies are returned so
              the caller can mutate the lists without affecting `request`.
            - If False, the original legacy sequences are returned as-is
              (zero-copy); treat them as read-only.

    Returns:
        Tuple[list[str], list[PlaceholderRange]]: (`mm_hashes`, `mm_positions`).
        May be `([], [])` when no multimodal data is present.
    """
    if getattr(request, "mm_features", None):
        mm_hashes, mm_positions = zip(
            *((f.identifier, f.mm_position) for f in request.mm_features), strict=False
        )
        return (list(mm_hashes), list(mm_positions))
    elif getattr(request, "mm_hashes", None):
        if modify:
            return (request.mm_hashes.copy(), request.mm_positions.copy())
        else:
            return (request.mm_hashes, request.mm_positions)
    else:
        return ([], [])


def get_size_bytes(shapes: list[torch.Size], kv_dtypes: list[torch.dtype]):
    """
    Calculate the size in bytes with the given shapes and dtypes.
    """
    assert len(shapes) == len(kv_dtypes), (
        f"shapes and dtypes must have the same length, "
        f"but got {len(shapes)} and {len(kv_dtypes)}"
    )
    return sum(
        shape.numel() * kv_dtype.itemsize
        for shape, kv_dtype in zip(shapes, kv_dtypes, strict=True)
    )

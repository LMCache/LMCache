import os
from typing import Union

from lmcache.config import \
    LMCacheEngineConfig as Config  # type: ignore[assignment]
from lmcache.experimental.config import \
    LMCacheEngineConfig as ExperimentalConfig  # type: ignore[assignment]
from lmcache.logging import init_logger

logger = init_logger(__name__)
ENGINE_NAME = "vllm-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> Union[Config, ExperimentalConfig]:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """

    if is_false(os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")):
        logger.warning("Detected LMCACHE_USE_EXPERIMENTAL is set to False. "
                       "Using legacy configuration is deprecated and will "
                       "be remove soon! Please set LMCACHE_USE_EXPERIMENTAL "
                       "to True.")
        LMCacheEngineConfig = Config  # type: ignore[assignment]
    else:
        LMCacheEngineConfig = ExperimentalConfig  # type: ignore[assignment]

    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn("No LMCache configuration file is set. Trying to read"
                    " configurations from the environment variables.")
        logger.warn("You can set the configuration file through "
                    "the environment variable: LMCACHE_CONFIG_FILE")
        config = LMCacheEngineConfig.from_env()
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    return config

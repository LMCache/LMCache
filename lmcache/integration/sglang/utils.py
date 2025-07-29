# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union
import os

# First Party
from lmcache.config import LMCacheEngineConfig as Config
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig as V1Config

logger = init_logger(__name__)
ENGINE_NAME = "sglang-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> Union[Config, V1Config]:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """
    logger.info(f"LMCACHE_USE_EXPERIMENTAL: {os.getenv('LMCACHE_USE_EXPERIMENTAL')}")
    logger.info(f"LMCACHE_CONFIG_FILE: {os.getenv('LMCACHE_CONFIG_FILE')}")
    if is_false(os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")):
        logger.warning(
            "Detected LMCACHE_USE_EXPERIMENTAL is set to False. "
            "Using legacy configuration is deprecated and will "
            "be remove soon! Please set LMCACHE_USE_EXPERIMENTAL "
            "to True."
        )
        LMCacheEngineConfig = Config  # type: ignore[assignment]
    else:
        LMCacheEngineConfig = V1Config  # type: ignore[assignment]

    if "LMCACHE_CONFIG_FILE" not in os.environ:
        logger.warn(
            "No LMCache configuration file is set. Trying to read"
            " configurations from the environment variables."
        )
        logger.warn(
            "You can set the configuration file through "
            "the environment variable: LMCACHE_CONFIG_FILE"
        )
        config = LMCacheEngineConfig.from_env()
    else:
        config_file = os.environ["LMCACHE_CONFIG_FILE"]
        logger.info(f"Loading LMCache config file {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)

    return config

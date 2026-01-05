# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)
ENGINE_NAME = "sglang-instance"


def is_false(value: str) -> bool:
    """Check if the given string value is equivalent to 'false'."""
    return value.lower() in ("false", "0", "no", "n", "off")


def lmcache_get_config() -> LMCacheEngineConfig:
    """Get the LMCache configuration from the environment variable
    `LMCACHE_CONFIG_FILE`. If the environment variable is not set, this
    function will return the default configuration.
    """
    logger.info(f"LMCACHE_CONFIG_FILE: {os.getenv('LMCACHE_CONFIG_FILE')}")
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

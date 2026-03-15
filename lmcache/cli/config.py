# SPDX-License-Identifier: Apache-2.0
"""CLI configuration using the centralized config system.

All CLI-related environment variables (``LMCACHE_CLI_*``) are defined here.
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.config_base import create_config_class

_CLI_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    "metrics_style": {
        "type": str,
        "default": "vllm",
        "env_converter": str,
    },
}

CLIConfig = create_config_class(
    config_name="CLIConfig",
    config_definitions=_CLI_CONFIG_DEFINITIONS,
    env_prefix="LMCACHE_CLI_",
)


def get_cli_config() -> "CLIConfig":  # type: ignore[valid-type]
    """Load CLI configuration from environment variables.

    Returns:
        A ``CLIConfig`` instance populated from ``LMCACHE_CLI_*`` env vars.
    """
    return CLIConfig.from_env()

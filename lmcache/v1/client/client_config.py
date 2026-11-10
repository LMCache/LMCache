# SPDX-License-Identifier: Apache-2.0
"""Configuration used by clients of the multiprocess LMCache server."""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import os

# Third Party
import yaml


@dataclass(frozen=True)
class MPClientConfig:
    """The subset of LMCache configuration required by MP clients."""

    mp_host: Optional[str] = None
    mp_port: int = 5555
    extra_config: dict[str, Any] = field(default_factory=dict)

    def get_extra_config_value(self, key: str, default: Any = None) -> Any:
        return self.extra_config.get(key, default)


def load_mp_client_config(config_file_path: Optional[str] = None) -> MPClientConfig:
    """Load MP client settings from YAML and matching environment variables."""
    resolved_path = config_file_path or os.getenv("LMCACHE_CONFIG_FILE")
    values: dict[str, Any] = {}
    if resolved_path:
        with Path(resolved_path).open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) or {}
        if not isinstance(loaded, dict):
            raise ValueError("LMCache config file must contain a mapping")
        values = loaded

    host = os.getenv("LMCACHE_MP_HOST", values.get("mp_host"))
    port = int(os.getenv("LMCACHE_MP_PORT", values.get("mp_port", 5555)))
    extra_config = values.get("extra_config") or {}
    if "LMCACHE_EXTRA_CONFIG" in os.environ:
        extra_config = json.loads(os.environ["LMCACHE_EXTRA_CONFIG"])
    if not isinstance(extra_config, dict):
        raise ValueError("extra_config must be a mapping")

    return MPClientConfig(
        mp_host=host,
        mp_port=port,
        extra_config=extra_config,
    )

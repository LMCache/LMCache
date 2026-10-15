# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenDropSpec:
    """The algorithm and its opaque request-local configuration."""

    algorithm: str
    config: dict[str, Any]


def parse_token_drop_spec(
    request_configs: Mapping[str, Any] | None,
) -> TokenDropSpec | None:
    """Return None for normal requests; validate the token-drop envelope."""
    if not request_configs or "lmcache.token_drop" not in request_configs:
        return None

    raw = request_configs["lmcache.token_drop"]
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.token_drop must be a mapping")

    unknown = set(raw) - {"algorithm", "config"}
    if unknown:
        raise ValueError(
            f"Unsupported lmcache.token_drop keys: {sorted(unknown, key=str)}"
        )

    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("lmcache.token_drop.algorithm must be a non-empty string")

    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("lmcache.token_drop.config must be a mapping")

    return TokenDropSpec(algorithm=algorithm, config=dict(config))

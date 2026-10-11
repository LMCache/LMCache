# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenDropSpec:
    algorithm: str
    config: dict[str, Any]


def parse_token_drop_spec(
    request_configs: Mapping[str, Any] | None,
) -> TokenDropSpec | None:
    """Parse a request's token-dropping selection; absent means normal serving."""
    if not request_configs or "lmcache.token_drop" not in request_configs:
        return None

    raw = request_configs["lmcache.token_drop"]
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.token_drop must be an object")
    if set(raw) != {"algorithm", "config"}:
        raise ValueError("lmcache.token_drop requires only algorithm and config")

    algorithm = raw["algorithm"]
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("lmcache.token_drop.algorithm must be a non-empty string")

    config = raw["config"]
    if not isinstance(config, Mapping):
        raise ValueError("lmcache.token_drop.config must be an object")

    return TokenDropSpec(algorithm=algorithm, config=dict(config))

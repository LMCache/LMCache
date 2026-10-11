# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration."""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any


@dataclass(frozen=True)
class TokenDropSpec:
    algorithm: str
    config: dict[str, Any]


def parse_token_drop_config(
    request_configs: Mapping[str, Any] | None,
) -> TokenDropSpec | None:
    """Parse per-request token-dropping config; return None for normal requests."""
    if not request_configs or "lmcache.token_drop" not in request_configs:
        return None

    raw = request_configs["lmcache.token_drop"]
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.token_drop must be a mapping")
    if set(raw) != {"algorithm", "config"}:
        raise ValueError("lmcache.token_drop requires only algorithm and config")

    algorithm = raw["algorithm"]
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("lmcache.token_drop.algorithm must be a non-empty string")

    config = raw["config"]
    if not isinstance(config, Mapping):
        raise ValueError("lmcache.token_drop.config must be a mapping")

    return TokenDropSpec(algorithm=algorithm, config=dict(config))


def build_token_drop_algorithm(spec: TokenDropSpec) -> Any:
    """Load an algorithm plugin for this request only."""
    matches = [
        ep
        for ep in entry_points(group="lmcache.token_drop_algorithms")
        if ep.name == spec.algorithm
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one token-dropping algorithm named {spec.algorithm!r}, "
            f"found {len(matches)}"
        )
    return matches[0].load()(dict(spec.config))

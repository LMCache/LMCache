# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal


@dataclass(frozen=True)
class TokenDropSpec:
    """The algorithm and its opaque request-local configuration."""

    algorithm: str
    config: dict[str, Any]


def build_token_drop_algorithm(spec: TokenDropSpec) -> Any:
    """Construct the request's selected external token-drop algorithm."""
    matches = [
        entry_point
        for entry_point in entry_points(group="lmcache.token_drop_algorithms")
        if entry_point.name == spec.algorithm
    ]
    if not matches:
        raise ValueError(
            f"No token-drop algorithm plugin registered for {spec.algorithm!r}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Multiple token-drop algorithm plugins registered for {spec.algorithm!r}"
        )
    factory = matches[0].load()
    return factory(dict(spec.config))


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


@dataclass
class LMCacheMPTokenDropRequestState:
    """Scheduler-provided request data; the worker assigns its batch row."""

    request_id: str
    algorithm: str
    config: dict[str, Any]
    resident_kv_tokens: int
    physical_block_ids: list[int] | None
    # The worker uses this with token counts to decide when to observe queries
    # and compact KV.
    phase: Literal["prefill", "decode"]
    num_decoded_tokens: int
    num_new_tokens: int
    worker_row: int = -1

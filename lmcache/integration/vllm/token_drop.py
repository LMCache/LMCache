# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration for the vLLM integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True)
class TokenDropSpec:
    """Generic per-request token-dropping envelope."""

    algorithm: str
    config: dict[str, Any]


def build_token_drop_algorithm(spec: TokenDropSpec) -> Any:
    """Construct the selected token-dropping algorithm at the dispatch boundary."""
    if spec.algorithm == "rkv":
        try:
            from rkv import R1KV
        except ImportError as exc:
            raise ImportError(
                "R-KV is enabled but the optional 'rkv' package is not installed"
            ) from exc
        return R1KV.from_serving_config(spec.config)

    raise ValueError(f"Unsupported token-drop algorithm: {spec.algorithm!r}")


def parse_token_drop_spec(
    request_configs: Mapping[str, Any] | None,
) -> TokenDropSpec | None:
    """Parse lmcache.token_drop; absence means vanilla request behavior."""
    if not request_configs or "lmcache.token_drop" not in request_configs:
        return None

    raw = request_configs["lmcache.token_drop"]
    if not isinstance(raw, Mapping):
        raise ValueError("lmcache.token_drop must be a JSON object")

    allowed = {"algorithm", "config"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported lmcache.token_drop keys: {sorted(unknown)}")

    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("lmcache.token_drop.algorithm must be a non-empty string")

    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("lmcache.token_drop.config must be a JSON object")

    return TokenDropSpec(
        algorithm=algorithm,
        config=dict(config),
    )


def token_drop_spec_from_request(request: "Request") -> TokenDropSpec | None:
    """Return this request's token-drop config, if any."""
    return parse_token_drop_spec(extract_request_configs_from_request(request))

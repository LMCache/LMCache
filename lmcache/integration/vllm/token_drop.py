# SPDX-License-Identifier: Apache-2.0
"""Per-request token-dropping configuration and algorithm discovery."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

# First Party
from lmcache.integration.vllm.utils import extract_request_configs_from_request

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.request import Request
    import torch

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_worker import KVView


_TOKEN_DROP_ALGORITHM_ENTRYPOINT_GROUP = "lmcache.token_drop_algorithms"


@dataclass(frozen=True)
class TokenDropSpec:
    """Generic per-request token-dropping envelope."""

    algorithm: str
    config: dict[str, Any]


class TokenDropAlgorithm(Protocol):
    """Methods required of a request-local token-dropping algorithm."""

    def should_observe_token_queries(
        self, phase: Literal["prefill", "decode"], decoded_tokens_before_step: int
    ) -> int: ...

    def observe_token_queries(
        self, queries_by_layer: Mapping[str, torch.Tensor]
    ) -> None: ...

    def should_compact_kv(
        self,
        phase: Literal["prefill", "decode"],
        resident_kv_tokens: int,
        decoded_tokens_before_step: int,
    ) -> bool: ...

    def select_kept_token_positions(
        self, kv_by_layer: Mapping[str, KVView]
    ) -> Mapping[str, torch.Tensor]: ...


def build_token_drop_algorithm(spec: TokenDropSpec) -> TokenDropAlgorithm:
    """Construct an algorithm from an external token-drop plugin."""
    matches = [
        entry_point
        for entry_point in entry_points(group=_TOKEN_DROP_ALGORITHM_ENTRYPOINT_GROUP)
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

    factory = cast(
        "Callable[[Mapping[str, Any]], TokenDropAlgorithm]", matches[0].load()
    )
    return factory(dict(spec.config))


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

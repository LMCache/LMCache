# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# Third Party
from vllm.forward_context import ForwardContext
import torch

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPTokenDropRequestState,
)
from lmcache.integration.vllm.token_drop import (
    TokenDropSpec,
    build_token_drop_algorithm,
)


class TokenDropWorker:
    """Manage request-local algorithm instances and selective post-RoPE Q."""

    def __init__(self) -> None:
        self._algorithms: dict[str, Any] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._indices: torch.Tensor | None = None
        self._slices: list[tuple[str, int, int]] = []
        self._observed: dict[str, torch.Tensor] = {}
        self._original_forwards: dict[str, tuple[Any, Any]] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = dict(kv_caches)

    def _algorithm(self, state: LMCacheMPTokenDropRequestState) -> Any:
        algorithm = self._algorithms.get(state.request_id)
        if algorithm is None:
            algorithm = build_token_drop_algorithm(
                TokenDropSpec(state.algorithm, dict(state.config))
            )
            self._algorithms[state.request_id] = algorithm
        return algorithm

    def is_token_drop_request(self, request_id: str) -> bool:
        return request_id in self._algorithms

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._algorithms.pop(request_id, None)

    def prepare_forward(
        self,
        forward_context: ForwardContext,
        request_states: list[LMCacheMPTokenDropRequestState],
    ) -> None:
        self._remove_hooks()
        self._indices = None
        self._slices.clear()
        self._observed.clear()
        if not request_states:
            return
        if not self._kv_caches:
            raise RuntimeError("Token dropping requires registered GPU KV caches")

        metadata = forward_context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Token dropping requires attention metadata")
        attention = next(iter(metadata.values()))
        query_start = attention.query_start_loc

        active = []
        for state in request_states:
            algorithm = self._algorithm(state)
            phase = "decode" if state.is_genuine_decode else "prefill"
            count = algorithm.should_observe_token_queries(
                phase, int(state.num_decoded_tokens)
            )
            if type(count) is not int or count < 0:
                raise ValueError("Query observation count must be nonnegative")
            count = min(count, int(state.num_new_tokens))
            if count:
                row = int(state.worker_row)
                if row < 0 or row + 1 >= query_start.shape[0]:
                    raise ValueError("Token-drop request has invalid worker row")
                active.append((state.request_id, row, int(state.num_new_tokens), count))
        if not active:
            return

        indices = []
        offset = 0
        for request_id, row, num_new_tokens, count in active:
            start = query_start[row].long()
            indices.append(
                torch.arange(count, device=query_start.device)
                + start
                + num_new_tokens
                - count
            )
            self._slices.append((request_id, offset, offset + count))
            offset += count
        self._indices = torch.cat(indices)
        self._install_hooks(forward_context.no_compile_layers)

    def _install_hooks(self, layers: dict[str, Any]) -> None:
        for name in self._kv_caches:
            if name not in layers or not hasattr(layers[name].impl, "forward"):
                raise RuntimeError(f"Missing attention implementation for {name!r}")

        for name in self._kv_caches:
            impl = layers[name].impl
            original = impl.forward

            def wrapped(
                attn_layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                *args,
                _forward=original,
                **kwargs,
            ):
                count = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
                self.capture_query(attn_layer.layer_name, query[:count])
                return _forward(
                    attn_layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    *args,
                    **kwargs,
                )

            impl.forward = wrapped
            self._original_forwards[name] = (impl, original)

    def _remove_hooks(self) -> None:
        for impl, original in self._original_forwards.values():
            impl.forward = original
        self._original_forwards.clear()

    def capture_query(self, layer_name: str, query: torch.Tensor) -> None:
        if self._indices is None or layer_name not in self._kv_caches:
            return
        head_dim = self._kv_caches[layer_name].shape[-1]
        if query.ndim == 2 and query.shape[1] % head_dim == 0:
            query = query.reshape(query.shape[0], -1, head_dim)
        elif query.ndim != 3 or query.shape[2] != head_dim:
            raise ValueError("Query must be [tokens, q_heads, head_dim]")
        self._observed[layer_name] = query.index_select(0, self._indices)

    def compact(self) -> dict[str, int]:
        """Flush observations after the forward; physical compaction comes later."""
        try:
            for request_id, start, end in self._slices:
                self._algorithms[request_id].observe_token_queries(
                    {name: tensor[start:end] for name, tensor in self._observed.items()}
                )
        finally:
            self._remove_hooks()
            self._indices = None
            self._slices.clear()
            self._observed.clear()
        return {}

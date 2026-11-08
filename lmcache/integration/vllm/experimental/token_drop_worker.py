# SPDX-License-Identifier: Apache-2.0
"""Worker-local query forwarding and paged-KV compaction for token dropping."""

# Standard
from typing import Any
import logging

# Third Party
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop import (
    LMCacheMPTokenDropRequestState,
    TokenDropSpec,
    build_token_drop_algorithm,
)

logger = logging.getLogger(__name__)


class TokenDropWorker:
    """Forward request-local Q and apply algorithm-selected KV compaction."""

    def __init__(self) -> None:
        self._request_algorithms: dict[str, Any] = {}
        self._request_generations: dict[str, int] = {}
        # A plugin may mutate its Q history before a recoverable pre-write error.
        # Disable further compaction for that request generation, not its decoding.
        self._failed_generations: dict[str, int] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._block_size = 0
        self._kv_caches_validated = False

    def _algorithm_from_state(self, state: LMCacheMPTokenDropRequestState) -> Any:
        existing = self._request_algorithms.get(state.request_id)
        if (
            existing is not None
            and self._request_generations.get(state.request_id) == state.generation
        ):
            return existing
        if state.generation < 1:
            raise ValueError("Token dropping requires a request generation")
        self.drop_requests({state.request_id})

        algorithm = build_token_drop_algorithm(
            TokenDropSpec(
                algorithm=state.algorithm,
                config=dict(state.config),
            )
        )
        self._request_algorithms[state.request_id] = algorithm
        self._request_generations[state.request_id] = state.generation
        return algorithm

    def finish_requests(self, request_ids: set[str]) -> set[str]:
        completed = request_ids & self._request_algorithms.keys()
        self.drop_requests(request_ids)
        return completed

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Registration is shared with normal LMCache serving. Validate the
        # token-drop runtime requirements only after a request opts in.
        self._kv_caches = dict(kv_caches)
        self._layer_names = list(kv_caches)
        self._block_size = 0
        self._kv_caches_validated = False

    def _ensure_kv_caches_compatible(self) -> None:
        if self._kv_caches_validated:
            return
        if not self._kv_caches:
            raise ValueError("Token dropping requires KV caches")

        block_size = 0
        for kv_cache in self._kv_caches.values():
            if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
                raise ValueError(
                    "Token dropping requires FlashAttention KV shaped "
                    "[num_blocks, 2, block_size, kv_heads, head_dim]"
                )
            if kv_cache.device.type != "cuda":
                raise ValueError("Token dropping requires CUDA-resident KV cache")
            if kv_cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Token dropping requires FP16 or BF16 KV cache")
            if block_size and kv_cache.shape[2] != block_size:
                raise ValueError("Token dropping requires one shared KV block size")
            block_size = int(kv_cache.shape[2])

        self._block_size = block_size
        self._kv_caches_validated = True

    def drop_requests(self, request_ids: set[str]) -> None:
        for request_id in request_ids:
            self._request_algorithms.pop(request_id, None)
            self._request_generations.pop(request_id, None)
            self._failed_generations.pop(request_id, None)

    def _disable_compaction(self, request_id: str) -> None:
        self._failed_generations[request_id] = self._request_generations[request_id]
        logger.warning(
            "Disabling further token-drop compaction for request %s after a "
            "recoverable pre-write failure",
            request_id,
        )

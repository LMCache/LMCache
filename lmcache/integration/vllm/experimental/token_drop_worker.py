# SPDX-License-Identifier: Apache-2.0
# Third Party
from vllm.forward_context import ForwardContext
import torch

# First Party
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPTokenDropRequestState,
)


class TokenDropWorker:
    """Capture post-RoPE Q; compact KV per layer/KV head; preserve model positions."""

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def prepare_forward(
        self,
        forward_context: ForwardContext,
        request_states: list[LMCacheMPTokenDropRequestState],
    ) -> None:
        raise NotImplementedError

    def capture_query(self, layer_name: str, query: torch.Tensor) -> None:
        raise NotImplementedError

    def compact(self) -> dict[str, int]:
        raise NotImplementedError

    def is_token_drop_request(self, request_id: str) -> bool:
        raise NotImplementedError

    def drop_requests(self, request_ids: set[str]) -> None:
        """Clear request-local algorithm state; normal requests are unaffected."""
        raise NotImplementedError

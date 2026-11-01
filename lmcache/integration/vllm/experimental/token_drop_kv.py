# SPDX-License-Identifier: Apache-2.0
"""Read GPU-resident paged K/V in request token order."""

# Future
from __future__ import annotations

# Third Party
import torch


class PagedKVSequenceView:
    """Read a request's K/V in token order.

    Returns [tokens, kv_heads, head_dim] tensors.
    """

    def __init__(self, cache: torch.Tensor, slots: torch.Tensor):
        self._cache = cache
        block_size = cache.shape[2]
        self._blocks = slots // block_size
        self._offsets = slots % block_size

    def _read(self, kv_index: int) -> torch.Tensor:
        return self._cache[:, kv_index][self._blocks, self._offsets]

    def get_keys(self) -> torch.Tensor:
        return self._read(0)

    def get_values(self) -> torch.Tensor:
        return self._read(1)

    def compact_kv_in_place(self, kept_positions: torch.Tensor) -> int:
        """Pack per-head K/V into this request's private blocks.

        Example: [A B C D], keep [1, 3] -> [B D _ _].
        Positions are [kept_tokens, kv_heads] valid resident indices in output
        order. Caller guarantees exclusive block ownership.

        In-place writes are not rolled back on failure. The caller can only
        fall back if the original KV and request state are still intact.
        """
        if kept_positions.ndim != 2 or kept_positions.shape[1] != self._cache.shape[3]:
            raise ValueError("Expected [kept_tokens, kv_heads] positions")
        kept_len = kept_positions.shape[0]
        if not 0 < kept_len <= self._blocks.numel():
            raise ValueError("Invalid retained token count")

        heads = torch.arange(self._cache.shape[3], device=self._cache.device)
        selected = self._cache[
            self._blocks[kept_positions], :, self._offsets[kept_positions], heads
        ]
        self._cache[
            self._blocks[:kept_len, None], :, self._offsets[:kept_len, None], heads
        ] = selected
        return kept_len

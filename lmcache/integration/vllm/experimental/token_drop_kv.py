# SPDX-License-Identifier: Apache-2.0
"""Read GPU-resident paged K/V in request token order."""

# Future
from __future__ import annotations

# Third Party
import torch


class PagedKVSequenceView:
    """Read a request's K/V in token order.

    Returns contiguous [tokens, kv_heads, head_dim] tensors.
    """

    def __init__(self, cache: torch.Tensor, slots: torch.Tensor):
        self._cache = cache
        block_size = cache.shape[2]
        self._blocks = slots // block_size
        self._offsets = slots % block_size

    def _read(self, kv_index: int) -> torch.Tensor:
        return self._cache[:, kv_index][self._blocks, self._offsets].contiguous()

    def get_keys(self) -> torch.Tensor:
        return self._read(0)

    def get_values(self) -> torch.Tensor:
        return self._read(1)

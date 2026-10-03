# SPDX-License-Identifier: Apache-2.0
"""R-KV token selection over GPU-resident paged KV.

The scoring follows the R-KV reference implementation:
https://github.com/Zefan-Cai/R-KV (commit
6715468b9872442be72e5c97322e4d9c9a2abf55), rkv/compression/r1_kv.py.
"""

# Standard
import math
from collections.abc import Sequence

# Third Party
import torch
import torch.nn.functional as F


def _attention_importance(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """Return attention logits grouped to KV heads."""
    _, q_heads, _, head_dim = queries.shape
    kv_heads = keys.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"q_heads {q_heads} is not a multiple of kv_heads {kv_heads}")

    group = q_heads // kv_heads
    if group == 1:
        return torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(head_dim)

    grouped = queries.view(
        queries.shape[0], kv_heads, group, queries.shape[2], head_dim
    )
    logits = torch.matmul(grouped, keys.unsqueeze(2).transpose(3, 4)) / math.sqrt(
        head_dim
    )
    return logits.max(dim=2).values


def _key_redundancy(keys: torch.Tensor) -> torch.Tensor:
    """Return the R-KV redundancy score of each resident token."""
    if keys.dtype == torch.float16:
        norm = keys.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = keys / norm.to(keys.dtype)
    else:
        normalized = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)

    similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
    similarity.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    seq_len = keys.shape[2]
    positions = torch.arange(seq_len, device=keys.device)
    retained = torch.where(similarity > 0.5, positions, 0).amax(dim=-1)
    similarity.scatter_(-1, retained.unsqueeze(-1), 0)
    return similarity.mean(dim=-2).softmax(dim=-1)


def select_rkv_retained_indices(
    layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    slots: torch.Tensor,
    budget: int,
    *,
    kernel: int = 7,
    mix_lambda: float = 0.07,
) -> torch.Tensor:
    """Select resident token positions to keep using R-KV.

    Each layer is (key_cache, recent_queries). The key cache is paged as
    [num_blocks, block_size, kv_heads, head_dim]; recent queries are
    [window, q_heads, head_dim]. slots contains the physical slot IDs of the
    request's current resident sequence, in sequence order.
    """
    if not layers:
        raise ValueError("R-KV requires at least one layer")

    window = layers[0][1].shape[0]
    length = slots.numel()
    if not 0 < window < budget < length:
        raise ValueError(
            f"R-KV requires 0 < window < budget < length, got "
            f"{window=}, {budget=}, {length=}"
        )

    total_scores: torch.Tensor | None = None
    for key_cache, recent_queries in layers:
        if recent_queries.shape[0] != window:
            raise ValueError("All R-KV query windows must have the same length")

        block_size = key_cache.shape[1]
        blocks = slots // block_size
        offsets = slots % block_size
        keys = key_cache[blocks, offsets].permute(1, 0, 2).unsqueeze(0)
        queries = recent_queries.permute(1, 0, 2).unsqueeze(0)

        logits = _attention_importance(queries, keys)
        importance = (
            F.softmax(logits[:, :, :, :-window], dim=-1, dtype=torch.float32)
            .mean(dim=-2)
            .to(queries.dtype)
        )
        importance = F.max_pool1d(
            importance, kernel_size=kernel, padding=kernel // 2, stride=1
        )
        redundancy = _key_redundancy(keys)[:, :, :-window]
        layer_scores = (importance * mix_lambda - redundancy * (1.0 - mix_lambda)).mean(
            dim=1
        )
        total_scores = (
            layer_scores if total_scores is None else total_scores + layer_scores
        )

    assert total_scores is not None
    if not torch.isfinite(total_scores).all():
        raise RuntimeError("R-KV produced non-finite scores")

    past = total_scores.topk(budget - window, dim=-1).indices[0]
    recent = torch.arange(length - window, length, device=slots.device)
    return torch.sort(torch.cat([past, recent])).values

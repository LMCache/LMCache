# SPDX-License-Identifier: Apache-2.0

# Standard
import math

# Third Party
import pytest
import torch
import torch.nn.functional as F

# First Party
from lmcache.integration.vllm.experimental.rkv import select_rkv_retained_indices

LAYERS = 3
KV_HEADS = 2
Q_HEADS = 4
HEAD_DIM = 16
BLOCK_SIZE = 16
BUDGET = 48
WINDOW = 8
KERNEL = 7
MIX_LAMBDA = 0.07


def _reference_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    _, q_heads, q_len, head_dim = queries.shape
    kv_heads = keys.shape[1]
    group = q_heads // kv_heads
    if group == 1:
        return torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(head_dim)
    queries = queries.view(1, kv_heads, group, q_len, head_dim)
    return (
        (torch.matmul(queries, keys.unsqueeze(2).transpose(3, 4)) / math.sqrt(head_dim))
        .max(dim=2)
        .values
    )


def _reference_redundancy(keys: torch.Tensor) -> torch.Tensor:
    if keys.dtype == torch.float16:
        norm = keys.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = keys / norm.to(keys.dtype)
    else:
        normalized = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
    seq_len = keys.shape[2]
    eye = torch.eye(seq_len, dtype=torch.bool, device=keys.device)
    similarity.masked_fill_(eye.view(1, 1, seq_len, seq_len), 0.0)
    positions = torch.arange(seq_len, device=keys.device).view(1, 1, 1, seq_len)
    retained = torch.where(
        similarity > 0.5,
        positions,
        torch.zeros_like(similarity, dtype=torch.long),
    ).amax(dim=-1)
    similarity.scatter_(-1, retained.unsqueeze(-1), 0)
    return similarity.mean(dim=-2).softmax(dim=-1)


def _reference_select(
    keys_per_layer: list[torch.Tensor],
    queries_per_layer: list[torch.Tensor],
) -> torch.Tensor:
    total = None
    for keys, queries in zip(keys_per_layer, queries_per_layer, strict=True):
        logits = _reference_attention(queries, keys)
        importance = (
            F.softmax(
                logits[:, :, -WINDOW:, :-WINDOW],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(queries.dtype)
        )
        importance = F.max_pool1d(
            importance,
            kernel_size=KERNEL,
            padding=KERNEL // 2,
            stride=1,
        )
        score = (
            importance * MIX_LAMBDA
            - _reference_redundancy(keys)[:, :, :-WINDOW] * (1 - MIX_LAMBDA)
        ).mean(dim=1)
        total = score if total is None else total + score

    past = total.topk(BUDGET - WINDOW, dim=-1).indices[0]
    recent = torch.arange(
        keys_per_layer[0].shape[2] - WINDOW,
        keys_per_layer[0].shape[2],
        device=total.device,
    )
    return torch.sort(torch.cat([past, recent])).values


def _case(
    length: int,
    dtype: torch.dtype,
    clustered: bool,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    blocks_needed = -(-length // BLOCK_SIZE)
    num_blocks = blocks_needed * 2 + 4
    block_ids = (
        torch.randperm(num_blocks - 1, generator=generator)[:blocks_needed] + 1
    ).cuda()
    slots = (
        block_ids.repeat_interleave(BLOCK_SIZE) * BLOCK_SIZE
        + torch.arange(BLOCK_SIZE, device="cuda").repeat(blocks_needed)
    )[:length]

    layers = []
    flat_keys = []
    flat_queries = []
    for _ in range(LAYERS):
        key_cache = torch.zeros(
            num_blocks,
            BLOCK_SIZE,
            KV_HEADS,
            HEAD_DIM,
            dtype=dtype,
            device="cuda",
        )
        if clustered:
            centers = torch.randn(
                4,
                KV_HEADS,
                HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            ).to(device="cuda", dtype=dtype)
            pick = torch.randint(0, 4, (length,), generator=generator).cuda()
            keys = centers[pick] + 0.2 * torch.randn(
                length,
                KV_HEADS,
                HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            ).to(device="cuda", dtype=dtype)
        else:
            keys = torch.randn(
                length,
                KV_HEADS,
                HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            ).to(device="cuda", dtype=dtype)
        queries = torch.randn(
            WINDOW,
            Q_HEADS,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).to(device="cuda", dtype=dtype)

        key_cache[slots // BLOCK_SIZE, slots % BLOCK_SIZE] = keys
        layers.append((key_cache, queries))
        flat_keys.append(keys.permute(1, 0, 2).unsqueeze(0))
        flat_queries.append(queries.permute(1, 0, 2).unsqueeze(0))

    return layers, slots, flat_keys, flat_queries


@pytest.mark.parametrize("length", [64, 65, 80])
@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_rkv_reference(length, clustered, dtype):
    layers, slots, keys, queries = _case(length, dtype, clustered)

    actual = select_rkv_retained_indices(
        layers,
        slots,
        BUDGET,
        kernel=KERNEL,
        mix_lambda=MIX_LAMBDA,
    )
    expected = _reference_select(keys, queries)

    assert torch.equal(actual, expected)
    assert actual.numel() == BUDGET
    assert torch.equal(
        actual[-WINDOW:],
        torch.arange(length - WINDOW, length, device="cuda"),
    )


def test_rejects_nothing_to_drop():
    layers, slots, _, _ = _case(BUDGET, torch.float32, False)
    with pytest.raises(ValueError, match="window < budget < length"):
        select_rkv_retained_indices(layers, slots, BUDGET)

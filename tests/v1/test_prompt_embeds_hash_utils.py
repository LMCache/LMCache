# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt-embeds token substitution.

vLLM puts token ID 0 at every position fed by ``prompt_embeds``, so LMCache
overwrites those positions with values derived from the embeddings before
token-based chunk hashing. Rows are digested in groups of 16 positions.
"""

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import apply_prompt_embeds_to_token_ids


def _embeds(seed: int, num_tokens: int = 32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, 16, generator=gen)


def _substitute(
    embeds: torch.Tensor,
    mask: list[bool] | None = None,
    token_ids: list[int] | None = None,
) -> list[int]:
    ids = torch.tensor(token_ids or [0] * embeds.shape[0])
    return apply_prompt_embeds_to_token_ids(ids, embeds, mask).tolist()


def test_deterministic_and_in_range() -> None:
    values = _substitute(_embeds(0))
    assert values == _substitute(_embeds(0))
    assert all(0 <= v < 2**31 for v in values)


def test_different_embeds_give_different_values() -> None:
    assert _substitute(_embeds(0)) != _substitute(_embeds(1))


def test_shared_groups_keep_shared_values() -> None:
    shared = _embeds(0)
    a = _substitute(torch.cat([shared, _embeds(1, num_tokens=16)]))
    b = _substitute(torch.cat([shared, _embeds(2, num_tokens=16)]))
    assert a[:32] == b[:32]
    assert a[32:] != b[32:]


def test_one_changed_row_changes_its_whole_group() -> None:
    a = _embeds(0)
    b = a.clone()
    b[20, 3] += 1.0
    values_a, values_b = _substitute(a), _substitute(b)
    assert values_a[:16] == values_b[:16]
    assert all(x != y for x, y in zip(values_a[16:], values_b[16:], strict=True))


def test_dtype_is_part_of_identity() -> None:
    fp16 = _embeds(0).to(torch.float16)
    assert _substitute(fp16) != _substitute(fp16.view(torch.bfloat16))


def test_mixed_mode_keeps_real_token_ids() -> None:
    mask = [True, True, False, False, True]
    a = _substitute(_embeds(0, num_tokens=5), mask, [11, 12, 0, 0, 13])
    b = _substitute(_embeds(1, num_tokens=5), mask, [11, 12, 0, 0, 13])
    assert (a[0], a[1], a[4]) == (11, 12, 13)
    assert a[2:4] != [0, 0]
    assert a[2:4] != b[2:4]

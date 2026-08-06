# SPDX-License-Identifier: Apache-2.0
"""Tests for validate_mamba_step_alignment (the relaxed scheduler guard).

Released constraint: align-mode Mamba-hybrid models used to require
``block_size <= max_num_batched_tokens < 2 * block_size``. The store-skip /
retrieve-window logic now makes steps that advance many blocks safe, so only
the lower bound (a step must advance at least one full block) remains.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    validate_mamba_step_alignment,
)


def _cfg(mamba_cache_mode: str, block_size: int, max_batched: int) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=mamba_cache_mode, block_size=block_size
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_batched),
    )


def test_non_align_models_are_unconstrained():
    # Any max_num_batched_tokens passes when not in align mode.
    validate_mamba_step_alignment(_cfg("none", 944, 1))


@pytest.mark.parametrize("max_batched", [944, 1888, 4720, 100_000])
def test_align_allows_at_least_one_block_including_beyond_2x(max_batched):
    # Equal, 2x, 5x, and far beyond the old 2x cap are all accepted now.
    validate_mamba_step_alignment(_cfg("align", 944, max_batched))


@pytest.mark.parametrize("max_batched", [0, 1, 943])
def test_align_rejects_below_block_size(max_batched):
    with pytest.raises(ValueError, match="max_num_batched_tokens >= block_size"):
        validate_mamba_step_alignment(_cfg("align", 944, max_batched))

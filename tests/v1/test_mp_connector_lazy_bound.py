# SPDX-License-Identifier: Apache-2.0
"""The geometry the MP connector hands the lazy-offload danger bound:
align-mode Mamba groups' blocks per step and the prefill chunk alignment."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
import importlib.util

# Third Party
import pytest

requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="requires vLLM"
)

BLOCK_SIZE = 4240


@dataclass
class MambaSpec:
    """Mamba spec double detected by its public class name.

    Attributes:
        block_size: Tokens per block.
        num_prefill_checkpoint_blocks: vLLM's prefill checkpoint blocks.
    """

    block_size: int
    num_prefill_checkpoint_blocks: int = 0


@dataclass
class FullAttentionSpec:
    """Attention spec double.

    Attributes:
        block_size: Tokens per block.
    """

    block_size: int


def _kv_cache_config(
    num_prefill_checkpoint_blocks: int = 0, attention_block_size: int = BLOCK_SIZE
) -> SimpleNamespace:
    """Three Mamba groups and one attention group, as in Qwen3.5 hybrids."""
    mamba = MambaSpec(BLOCK_SIZE, num_prefill_checkpoint_blocks)
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=mamba)] * 3
        + [
            SimpleNamespace(kv_cache_spec=FullAttentionSpec(attention_block_size)),
        ]
    )


def _vllm_config(
    mamba_cache_mode: str = "align",
    max_num_batched_tokens: int = 2 * BLOCK_SIZE - 1,
    long_prefill_token_threshold: int = 0,
) -> SimpleNamespace:
    """Build the cache and scheduler fields the helpers read.

    Args:
        mamba_cache_mode: vLLM's ``mamba_cache_mode``.
        max_num_batched_tokens: vLLM's ``max_num_batched_tokens``.
        long_prefill_token_threshold: vLLM's chunk cap; 0 disables it.

    Returns:
        A ``VllmConfig`` stand-in with those fields.
    """
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            mamba_cache_mode=mamba_cache_mode,
            prefix_match_unit=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=None,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
        ),
    )


def _import_helpers() -> tuple[Callable, Callable]:
    """Import helpers lazily because their module imports vLLM."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import (
        get_mamba_step_extra_blocks,
        get_prefill_alignment,
    )

    return get_mamba_step_extra_blocks, get_prefill_alignment


@requires_vllm
def test_align_mode_mamba_groups_take_one_block_per_step() -> None:
    """Without checkpoints or a finer hash grid, no extra blocks."""
    get_mamba_step_extra_blocks, _ = _import_helpers()

    extra = get_mamba_step_extra_blocks(_vllm_config(), _kv_cache_config())

    assert extra == {0: 0, 1: 0, 2: 0}


@requires_vllm
def test_checkpoints_and_partial_hits_add_blocks() -> None:
    """A checkpoint block, and a copy-on-write block on a finer hash grid."""
    get_mamba_step_extra_blocks, _ = _import_helpers()
    kv_cache_config = _kv_cache_config(
        num_prefill_checkpoint_blocks=1, attention_block_size=BLOCK_SIZE // 4
    )

    extra = get_mamba_step_extra_blocks(_vllm_config(), kv_cache_config)

    assert extra == {0: 2, 1: 2, 2: 2}


@requires_vllm
def test_mamba_outside_align_mode_is_not_capped() -> None:
    """Without align mode, no Mamba group is capped and prefill is unaligned."""
    get_mamba_step_extra_blocks, get_prefill_alignment = _import_helpers()
    config = _vllm_config(mamba_cache_mode="none")
    mamba_groups = get_mamba_step_extra_blocks(config, _kv_cache_config())

    assert mamba_groups == {}
    assert get_prefill_alignment(config, _kv_cache_config(), mamba_groups) == 0


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "long_prefill_token_threshold", "alignment"),
    [
        (2 * BLOCK_SIZE - 1, 0, BLOCK_SIZE),
        (BLOCK_SIZE - 1, 0, 0),
        (2 * BLOCK_SIZE, BLOCK_SIZE - 1, 0),
    ],
    ids=["block-fits-chunk", "block-wider-than-chunk", "threshold-caps-chunk"],
)
@requires_vllm
def test_prefill_aligns_only_when_a_block_fits_a_chunk(
    max_num_batched_tokens: int, long_prefill_token_threshold: int, alignment: int
) -> None:
    """Prefill aligns to a block only when a whole block fits one chunk."""
    get_mamba_step_extra_blocks, get_prefill_alignment = _import_helpers()
    config = _vllm_config(
        max_num_batched_tokens=max_num_batched_tokens,
        long_prefill_token_threshold=long_prefill_token_threshold,
    )
    mamba_groups = get_mamba_step_extra_blocks(config, _kv_cache_config())

    assert get_prefill_alignment(config, _kv_cache_config(), mamba_groups) == alignment

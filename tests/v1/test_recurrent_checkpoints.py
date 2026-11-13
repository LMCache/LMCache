# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.recurrent_checkpoints import publish_recurrent_checkpoints


def test_registers_only_copied_checkpoint_boundaries() -> None:
    pool = MagicMock()
    request = MagicMock()
    block = SimpleNamespace(block_id=17)

    publish_recurrent_checkpoints(pool, request, 16, {32: {2: block}})

    pool.cache_full_blocks.assert_called_once_with(
        request, [pool.null_block, block], 1, 2, 16, 2
    )
    pool.evict_blocks.assert_not_called()


def test_rolls_back_partial_checkpoint_publication() -> None:
    pool = MagicMock()
    pool.cache_full_blocks.side_effect = [None, ValueError("invalid prefix")]
    blocks = {2: SimpleNamespace(block_id=17), 3: SimpleNamespace(block_id=18)}

    with pytest.raises(ValueError, match="invalid prefix"):
        publish_recurrent_checkpoints(pool, MagicMock(), 16, {32: blocks})

    pool.evict_blocks.assert_called_once_with({17, 18})

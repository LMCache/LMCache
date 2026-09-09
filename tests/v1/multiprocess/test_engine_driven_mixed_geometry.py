# SPDX-License-Identifier: Apache-2.0
"""Engine-driven registration when groups have *different* block sizes.

``test_engine_driven_multigroup.py`` builds every group from identically shaped
tensors, so nothing there exercises mixed geometry -- which is where gemma-4
(five sliding groups at 32-token blocks, one global at 64) lost half of every
chunk's KV: the tier stored 440.00 KiB/token where the model's own geometry says
880.00, and the retrieved KV decoded to garbage.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import math

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import compute_kv_layout
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

# Shapes are (kv planes, blocks, block size, heads, head size). gemma-4's real
# situation: the wide group has half the hidden size and twice the block size, so
# the two groups have equal page sizes and *unequal* block sizes.
#
# The two shapes must differ at BOTH dim 2 and dim 3, because which of the two is
# the block size depends on the resolved kv_layout: NHD reads the block size at
# dim 2, HND at dim 3 (VLLM_Detector forces HND whenever torch_device_type is
# "cpu", so the CPU-only unit lane takes the latter). A fixture that differs at
# only one of them collapses to a single block size under the other layout and
# the mixed-geometry premise silently disappears.
_NARROW_SHAPE = (2, 4, 4, 2, 8)
_WIDE_PAGED_SHAPE = (2, 4, 8, 4, 2)


def _block_size_of(shape: tuple[int, ...]) -> int:
    """Tokens one paged block of ``shape`` physically holds.

    Uses the production helper rather than re-deriving it from ``shape`` so the
    test cannot disagree with the code it checks about which dim is the block.
    """
    block_size, *_ = compute_kv_layout({"layer": torch.zeros(shape)})
    return block_size


def test_groups_with_unequal_block_sizes_register_the_whole_chunk() -> None:
    """Mixed block sizes must not shrink the chunk (the gemma-4 halving).

    ``blocks_in_chunk`` counts the engine's scheduling blocks, and the engine
    schedules in units of the LCM of its groups' block sizes -- 8 here, as
    gemma-4 schedules in 64 with groups of 32 and 64. So a chunk is
    ``blocks_in_chunk * lcm`` tokens and every full-attention group must register
    exactly that many. Deriving it from the block size of whichever layer comes
    first registers a fraction of the chunk instead: the rest is never stored.
    """
    narrow_block = _block_size_of(_NARROW_SHAPE)
    wide_block = _block_size_of(_WIDE_PAGED_SHAPE)
    assert wide_block != narrow_block, "fixture must have two different block sizes"
    # 2, not 1: the shrunken chunk must still be divisible by every group's block
    # size, or the existing divisibility guard raises and the loss is loud. It was
    # silent for gemma-4 (128 tokens, divisible by both 32 and 64), and this is
    # the smallest fixture that reproduces that.
    blocks_in_chunk = 2
    expected_chunk = blocks_in_chunk * math.lcm(narrow_block, wide_block)

    sent: list[Any] = []

    def _register(payload: Any) -> Any:
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_mixed", pool_size=4096
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register

    EngineDrivenTransferContext().register(
        instance_id=1,
        kv_caches={
            "layer_0": torch.zeros(_NARROW_SHAPE),
            "layer_1": torch.zeros(_NARROW_SHAPE),
            "layer_2": torch.zeros(_WIDE_PAGED_SHAPE),
        },
        model_name="mixed-geometry",
        world_size=1,
        blocks_in_chunk=blocks_in_chunk,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1),
                tokens_per_block=narrow_block,
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(2,),
                tokens_per_block=wide_block,
            ),
        ],
    )

    window_tokens = [gl.window_tokens for gl in sent[0].group_layouts]
    assert window_tokens == [expected_chunk, expected_chunk], (
        f"groups registered {window_tokens} tokens per chunk, but the chunk is "
        f"{expected_chunk} tokens ({blocks_in_chunk} scheduling block(s) of "
        f"lcm({narrow_block}, {wide_block}))"
    )

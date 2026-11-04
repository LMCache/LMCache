# SPDX-License-Identifier: Apache-2.0
"""Layout coverage for the Mamba unified-state view in kv_cache_group_edits."""

# Standard
from typing import cast

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="kv_cache_group_edits imports vLLM at module top")

# Third Party
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    apply_kv_cache_group_edits,
)
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402

NUM_BLOCKS = 3
BLOCK_SIZE = 16
# Elements in one layer's unified state row, and per block including padding.
STATE_ROW = 8000
BLOCK_STEP = 8064
# ceil(STATE_ROW / BLOCK_SIZE) = 500, rounded up to the 16-byte (8 x bf16)
# alignment: 504 * 16 * 2 bytes fills the padded page exactly.
HEAD_SIZE = 504


def _unified_mamba_layer() -> tuple[KVCacheConfig, dict[str, torch.Tensor]]:
    """Build one align-mode Mamba layer registered in vLLM's unified layout.

    Returns:
        The KV cache config and the registered cache: a
        ``[num_blocks, 1, 1, row]`` tensor with strides
        ``(block_step, row, row, 1)``.
    """
    storage = torch.zeros(NUM_BLOCKS * BLOCK_STEP, dtype=torch.bfloat16)
    kv_cache = storage.as_strided(
        (NUM_BLOCKS, 1, 1, STATE_ROW), (BLOCK_STEP, STATE_ROW, STATE_ROW, 1)
    )
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((STATE_ROW,),),
        dtypes=(torch.bfloat16,),
        page_size_padded=BLOCK_STEP * storage.element_size(),
        mamba_cache_mode="align",
    )
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )
    return config, {"layer.0": kv_cache}


@pytest.mark.parametrize(
    ("kv_layout", "inner_shape"),
    [
        ("NHD", (BLOCK_SIZE, 1, HEAD_SIZE)),
        ("HND", (1, BLOCK_SIZE, HEAD_SIZE)),
        ("BLNHC", (BLOCK_SIZE, 1, HEAD_SIZE)),
        ("BLHNC", (1, BLOCK_SIZE, HEAD_SIZE)),
    ],
)
def test_mamba_unified_view_blocks_first_matches_layers_first(
    kv_layout: str, inner_shape: tuple[int, int, int]
) -> None:
    """BLNHC views like NHD and BLHNC like HND, over the same storage."""
    config, kv_caches = _unified_mamba_layer()

    layout_hints = cast(LayoutHints, {"kv_layout": kv_layout})
    edited = apply_kv_cache_group_edits(config, kv_caches, layout_hints)

    view = edited["layer.0"]
    assert isinstance(view, torch.Tensor)
    assert tuple(view.shape) == (NUM_BLOCKS, *inner_shape)
    assert view.stride(0) == BLOCK_STEP
    assert view.data_ptr() == kv_caches["layer.0"].data_ptr()

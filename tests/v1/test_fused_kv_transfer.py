# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process transfer of fused/packed (CS / TWO_HS) KV layouts.

Covers the ``MemObjKVLayout`` contract on ``multi_layer_kv_transfer``: fused
engine formats must declare the LMCache-side buffer layout explicitly, and the
parameter is rejected for every other format.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import (
    EngineKVFormat,
    MemObjKVLayout,
    TransferDirection,
)
from lmcache.v1.platform import torch_ops

_FUSED_FORMATS = [
    EngineKVFormat.NL_X_NB_BS_NH_CS,
    EngineKVFormat.NL_X_NB_NH_BS_CS,
    EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
    EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
]


def _dummy_transfer_args() -> dict:
    """Minimal arguments; contract validation fires before any tensor use."""
    return dict(
        key_value=torch.zeros(2, 1, 4, 8),
        key_value_ptrs=[torch.zeros(1, 4, 8)],
        slot_mapping=torch.zeros(4, dtype=torch.long),
        paged_memory_device=torch.device("cpu"),
        page_buffer_size=4,
        direction=TransferDirection.D2H,
        engine_kv_format=EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        block_size=4,
    )


@pytest.mark.parametrize("fmt", _FUSED_FORMATS)
def test_fused_format_requires_explicit_layout(fmt):
    args = _dummy_transfer_args()
    args["engine_kv_format"] = fmt
    with pytest.raises(ValueError, match="explicit mem_obj_kv_layout"):
        torch_ops.multi_layer_kv_transfer(**args)


@pytest.mark.parametrize(
    "layout", [MemObjKVLayout.SPLIT_KV_2LTD, MemObjKVLayout.FUSED_PACKED]
)
def test_non_fused_format_rejects_layout(layout):
    args = _dummy_transfer_args()
    args["mem_obj_kv_layout"] = layout
    with pytest.raises(ValueError, match="must pass UNSPECIFIED"):
        torch_ops.multi_layer_kv_transfer(**args)

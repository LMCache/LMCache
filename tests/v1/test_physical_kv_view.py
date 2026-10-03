# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.physical_kv_view import (
    apply_physical_kv_view,
)


def test_apply_physical_kv_view_updates_attention_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Third Party
    from vllm.model_executor.layers.attention import attention as attention_mod

    old_seq_lens = torch.tensor([8], dtype=torch.int32)
    old_block_table = torch.tensor([[1, 2]], dtype=torch.int32)
    old_slot_mapping = torch.tensor([31], dtype=torch.int64)
    metadata = SimpleNamespace(
        use_cascade=False,
        seq_lens=old_seq_lens,
        max_seq_len=8,
        block_table=old_block_table,
        slot_mapping=old_slot_mapping,
    )
    layer = SimpleNamespace(kv_cache=torch.empty(0))
    context = SimpleNamespace(
        no_compile_layers={"layer": layer},
        attn_metadata={"layer": metadata},
        slot_mapping={"layer": old_slot_mapping},
    )

    physical_seq_lens = torch.tensor([4], dtype=torch.int32)
    physical_block_table = torch.tensor([[7]], dtype=torch.int32)
    physical_slot_mapping = {"layer": torch.tensor([112], dtype=torch.int64)}

    apply_physical_kv_view(
        context,
        seq_lens=physical_seq_lens,
        max_seq_len=4,
        block_table=physical_block_table,
        slot_mapping=physical_slot_mapping,
    )

    # FlashAttention consumes these fields from per-layer attention metadata.
    assert metadata.seq_lens is physical_seq_lens
    assert metadata.max_seq_len == 4
    assert metadata.block_table is physical_block_table
    assert metadata.slot_mapping is physical_slot_mapping["layer"]

    # vLLM's separate KV-update path reads slot mapping from ForwardContext.
    monkeypatch.setattr(attention_mod, "get_forward_context", lambda: context)
    returned_metadata, returned_layer, _, returned_slot_mapping = (
        attention_mod.get_attention_context("layer")
    )
    assert returned_metadata is metadata
    assert returned_layer is layer
    assert returned_slot_mapping is physical_slot_mapping["layer"]


def test_apply_physical_kv_view_rejects_speculative_metadata() -> None:
    context = SimpleNamespace(attn_metadata=[{}], slot_mapping={})

    with pytest.raises(ValueError, match="non-speculative"):
        apply_physical_kv_view(
            context,
            seq_lens=torch.tensor([1]),
            max_seq_len=1,
            block_table=torch.tensor([[1]]),
            slot_mapping={},
        )


def test_apply_physical_kv_view_rejects_cascade_attention() -> None:
    metadata = SimpleNamespace(use_cascade=True)
    context = SimpleNamespace(
        attn_metadata={"layer": metadata},
        slot_mapping={"layer": torch.tensor([0])},
    )

    with pytest.raises(ValueError, match="cascade"):
        apply_physical_kv_view(
            context,
            seq_lens=torch.tensor([1]),
            max_seq_len=1,
            block_table=torch.tensor([[1]]),
            slot_mapping={"layer": torch.tensor([0])},
        )

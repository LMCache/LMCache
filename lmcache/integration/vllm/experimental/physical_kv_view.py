# SPDX-License-Identifier: Apache-2.0
"""Build a physical KV view for an eager vLLM forward."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    import torch
    from vllm.forward_context import ForwardContext


def apply_physical_kv_view(
    forward_context: "ForwardContext",
    *,
    seq_lens: "torch.Tensor",
    max_seq_len: int,
    block_table: "torch.Tensor",
    slot_mapping: dict[str, "torch.Tensor"],
) -> None:
    """Replace the KV-facing view without changing model token positions."""
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        raise ValueError("Physical KV view requires non-speculative attention metadata")

    if set(slot_mapping) != set(attn_metadata):
        raise ValueError("Physical KV slot mapping must cover every attention layer")

    for layer_name, metadata in attn_metadata.items():
        if getattr(metadata, "use_cascade", False):
            raise ValueError("Physical KV view does not support cascade attention")

        metadata.seq_lens = seq_lens
        metadata.max_seq_len = max_seq_len
        metadata.block_table = block_table
        metadata.slot_mapping = slot_mapping[layer_name]

    # vLLM's separate KV-update path reads slot mapping from ForwardContext,
    # while attention reads the fields above from its per-layer metadata.
    forward_context.slot_mapping = slot_mapping

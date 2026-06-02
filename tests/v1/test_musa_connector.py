# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated tests for the non-layerwise vLLM MUSA connector."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import VLLMPagedMemMUSAConnectorV2
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
)


def _skip_if_no_musa() -> None:
    """Skip the current test unless torch-musa is available."""
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("torch.musa is not available")


def _make_unique_slot_mapping(
    *, total_slots: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    """Create unique slot ids for paged KV cache tests.

    Args:
        total_slots: Total slots available in the paged cache.
        num_tokens: Number of token slots to select.
        device: Device where the slot mapping should live.

    Returns:
        A tensor of unique slot ids.
    """
    return torch.randperm(total_slots, device=device, dtype=torch.int64)[:num_tokens]


def _pack_slot_mapping(
    slot_mapping: torch.Tensor, starts: list[int], ends: list[int]
) -> torch.Tensor:
    """Pack multiple slot mapping ranges into one tensor.

    Args:
        slot_mapping: Full slot mapping.
        starts: Start offsets.
        ends: End offsets.

    Returns:
        Concatenated slot mapping ranges.
    """
    return torch.cat(
        [slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)],
        dim=0,
    )


def _make_metadata(
    *,
    model_name: str,
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
) -> LMCacheMetadata:
    """Create metadata for a synthetic vLLM MUSA KV cache.

    Args:
        model_name: Metadata model name.
        num_layers: Number of KV cache layers.
        num_tokens: Number of tokens in the transfer.
        num_heads: Number of KV heads.
        head_size: Per-head dimension.

    Returns:
        Metadata for the connector under test.
    """
    return LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, num_tokens, num_heads, head_size),
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise(use_gpu: bool) -> None:
    """Round-trip from_gpu -> to_gpu on the non-layerwise MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    head_size = 64
    num_tokens = 32

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=num_tokens,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test",
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        ),
        use_gpu=use_gpu,
        device=device,
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_dst,
        )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


def test_musa_connector_to_gpu_accepts_cpu_slot_mapping() -> None:
    """Round-trip with CPU ``slot_mapping`` and MUSA KV cache tensors."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    head_size = 64
    num_tokens = 32

    kvcaches_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )
    _, _, num_heads_actual, head_size_actual = kvcaches_src[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping_cpu = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=num_tokens,
        device=torch.device("cpu"),
    )
    slot_mapping_musa = slot_mapping_cpu.to(device)

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_cpu_slot_mapping",
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        ),
        use_gpu=False,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping_cpu,
            kvcaches=kvcaches_src,
        )

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping_cpu,
            kvcaches=kvcaches_dst,
        )

        check_paged_kv_cache_equal(
            kvcaches_src,
            kvcaches_dst,
            slot_mapping_musa,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise_multi_chunk(
    use_gpu: bool,
) -> None:
    """Non-layerwise multi-chunk round-trip on the MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 6
    block_size = 8
    head_size = 64
    total_tokens = 32

    starts = [0, 7, 19]
    ends = [4, 13, 25]

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )
    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_non_layerwise_multi_chunk",
            num_layers=num_layers,
            num_tokens=total_tokens,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        ),
        use_gpu=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobjs = []
    try:
        for start, end in zip(starts, ends, strict=False):
            num_chunk_tokens = end - start
            memobj = pin_alloc.allocate(
                torch.Size([2, num_layers, num_chunk_tokens, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_2LTD,
            )
            conn.from_gpu(
                memobj,
                start=start,
                end=end,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches,
            )
            memobjs.append((start, end, memobj))

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        for start, end, memobj in memobjs:
            conn.to_gpu(
                memobj,
                start=start,
                end=end,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches_dst,
            )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for _, _, memobj in memobjs:
            memobj.ref_count_down()
        pin_alloc.close()

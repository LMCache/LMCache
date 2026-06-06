# SPDX-License-Identifier: Apache-2.0
"""Hardware-gated tests for the non-layerwise vLLM MUSA connector."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import VLLMPagedMemMUSAConnectorV2
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
)
import lmcache.c_ops as lmc_ops


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


def _patch_musa_connector_attrs(
    monkeypatch: pytest.MonkeyPatch,
    conn: VLLMPagedMemMUSAConnectorV2,
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    gpu_kv_format: lmc_ops.GPUKVFormat = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
) -> None:
    """Patch connector layout discovery so transfer logic can run on CPU."""

    def _initialize_attributes(_kv_caches: list[torch.Tensor]) -> None:
        conn.device = torch.device("cpu")
        conn.gpu_kv_format = gpu_kv_format
        conn.num_layers = num_layers
        conn.num_blocks = num_blocks
        conn.block_size = block_size
        conn.page_buffer_size = num_blocks * block_size
        conn.hidden_dim_size = num_heads * head_size
        conn.head_size = head_size
        conn.use_mla = False
        conn.dtype = torch.bfloat16
        conn.num_heads = num_heads
        conn._attributes_initialized = True

    monkeypatch.setattr(conn, "_initialize_attributes", _initialize_attributes)


def test_musa_connector_to_gpu_skips_vllm_cached_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vllm_cached_tokens`` skips prefix slots before torch ``index_copy_``."""
    num_layers = 1
    num_blocks = 3
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size
    start = 4
    end = 8
    vllm_cached_tokens = 6
    skipped = vllm_cached_tokens - start

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_vllm_cached_tokens",
            num_layers=num_layers,
            num_tokens=end - start,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    kvcaches_dst = [
        torch.zeros(
            2,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
    ]
    memory_tensor = torch.arange(
        2 * num_layers * (end - start) * hidden_dim,
        dtype=torch.float32,
    ).reshape(2, num_layers, end - start, hidden_dim)
    memory_tensor = memory_tensor.to(torch.bfloat16)
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=memory_tensor,
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    slot_mapping = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 6, 7],
        dtype=torch.long,
    )

    conn.to_gpu(
        memory_obj,
        start=start,
        end=end,
        slot_mapping=slot_mapping,
        kvcaches=kvcaches_dst,
        vllm_cached_tokens=vllm_cached_tokens,
    )

    flat_k = kvcaches_dst[0][0].reshape(num_blocks * block_size, hidden_dim)
    flat_v = kvcaches_dst[0][1].reshape(num_blocks * block_size, hidden_dim)
    target_slots = slot_mapping[vllm_cached_tokens:end]

    assert torch.equal(flat_k[target_slots], memory_tensor[0, 0, skipped:])
    assert torch.equal(flat_v[target_slots], memory_tensor[1, 0, skipped:])
    assert torch.count_nonzero(flat_k[-1]) == 0
    assert torch.count_nonzero(flat_v[-1]) == 0


def test_musa_connector_rejects_unsupported_kv_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported MUSA KV layouts fail with a connector-level message."""
    num_layers = 1
    num_blocks = 3
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_unsupported_layout",
            num_layers=num_layers,
            num_tokens=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        gpu_kv_format=lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    )

    kvcaches_dst = [
        torch.zeros(
            num_blocks,
            2,
            block_size,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
    ]
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=torch.zeros(
                2,
                num_layers,
                block_size,
                hidden_dim,
                dtype=torch.bfloat16,
            ),
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )

    with pytest.raises(ValueError, match="VLLMPagedMemMUSAConnectorV2 supports"):
        conn.to_gpu(
            memory_obj,
            start=0,
            end=block_size,
            slot_mapping=torch.arange(block_size, dtype=torch.long),
            kvcaches=kvcaches_dst,
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

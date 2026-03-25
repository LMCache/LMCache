# SPDX-License-Identifier: Apache-2.0
# tests/v1/test_xpu_connector.py

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.xpu_connectors import (
    VLLMPagedMemLayerwiseXPUConnector,
    VLLMPagedMemXPUConnectorV2,
)
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
)


def _skip_if_no_xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("torch.xpu is not available")


def _make_unique_slot_mapping(
    *, total_slots: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    # Unique indices avoids overwriting the same slot multiple times.
    return torch.randperm(total_slots, device=device, dtype=torch.int64)[:num_tokens]


def _pack_slot_mapping(
    slot_mapping: torch.Tensor, starts: list[int], ends: list[int]
) -> torch.Tensor:
    return torch.cat(
        [slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)],
        dim=0,
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_xpu_connector_roundtrip_non_layerwise(use_gpu: bool):
    _skip_if_no_xpu()
    device = torch.device("xpu:0")

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

    # Derive actual dims from generated KV (avoid helper defaults mismatch)
    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    meta = LMCacheMetadata(
        model_name="xpu_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, num_tokens, num_heads_actual, head_size_actual),
    )
    conn = VLLMPagedMemXPUConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    try:
        # XPU -> CPU (KV_2LTD in memobj)
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )

        # CPU -> XPU into fresh caches
        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

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


@pytest.mark.parametrize("use_gpu", [False, True])
def test_xpu_connector_roundtrip_layerwise(use_gpu: bool):
    _skip_if_no_xpu()
    device = torch.device("xpu:0")

    num_layers = 4
    num_blocks = 8
    block_size = 16
    head_size = 64
    num_tokens = 64

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    # Derive actual dims from generated KV
    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    meta = LMCacheMetadata(
        model_name="xpu_test_layerwise",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, num_tokens, num_heads_actual, head_size_actual),
    )

    conn = VLLMPagedMemLayerwiseXPUConnector.from_metadata(
        meta,
        use_xpu=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 256)

    # Per-layer list-of-chunks. We use 1 chunk: [0, num_tokens)
    memobjs_by_layer = [
        [
            pin_alloc.allocate(
                torch.Size([num_tokens, 2, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
        ]
        for _ in range(num_layers)
    ]

    try:
        # XPU -> CPU (layerwise generator): yields num_layers + 1 times
        gen = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )

        # Drive generator: one yield per layer + final yield
        for _ in range(num_layers + 1):
            next(gen)

        # CPU -> XPU into fresh caches (layerwise generator):
        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

        gen2 = conn.batched_to_gpu(
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )

        next(gen2)  # layer 0 expects send()
        for layer_id in range(num_layers):
            gen2.send(memobjs_by_layer[layer_id])

        # After the last send, generator is at "yield  # after last layer"
        next(gen2)  # advances to "yield  # final"

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for m in layer:
                m.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_xpu_connector_roundtrip_non_layerwise_multi_chunk(
    use_gpu: bool,
) -> None:
    _skip_if_no_xpu()
    device = torch.device("xpu:0")

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

    meta = LMCacheMetadata(
        model_name="xpu_test_non_layerwise_multi_chunk",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, total_tokens, num_heads_actual, head_size_actual),
    )
    conn = VLLMPagedMemXPUConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobjs = []
    try:
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            memobj = pin_alloc.allocate(
                torch.Size([2, num_layers, n, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_2LTD,
            )
            conn.from_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches,
            )
            memobjs.append((s, e, memobj))

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        for s, e, memobj in memobjs:
            conn.to_gpu(
                memobj,
                start=s,
                end=e,
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


@pytest.mark.parametrize("use_xpu", [False, True])
def test_xpu_connector_roundtrip_layerwise_multi_chunk(use_xpu: bool) -> None:
    _skip_if_no_xpu()
    device = torch.device("xpu:0")

    num_layers = 4
    num_blocks = 8
    block_size = 8
    head_size = 64
    total_tokens = 40

    starts = [0, 9, 21]
    ends = [5, 15, 30]

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

    meta = LMCacheMetadata(
        model_name="xpu_test_layerwise_multi_chunk_gpu",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, total_tokens, num_heads_actual, head_size_actual),
    )
    conn = VLLMPagedMemLayerwiseXPUConnector.from_metadata(
        meta,
        use_xpu=use_xpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    torch.Size([n, 2, hidden_dim_actual]),
                    torch.bfloat16,
                    MemoryFormat.KV_T2D,
                )
            )
        memobjs_by_layer.append(per_layer)

    try:
        producer = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )
        for _ in range(num_layers + 1):
            next(producer)

        if use_xpu:
            assert conn.gpu_buffer_allocator is not None
        else:
            assert conn.gpu_buffer_allocator is None

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        consumer = conn.batched_to_gpu(
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )
        next(consumer)
        for layer_id in range(num_layers):
            consumer.send(memobjs_by_layer[layer_id])
        next(consumer)

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()

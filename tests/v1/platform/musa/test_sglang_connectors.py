# SPDX-License-Identifier: Apache-2.0
"""Tests for in-process SGLang KV-cache transfer on MUSA."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import (
    SGLangLayerwiseMUSAConnector,
    SGLangMUSAConnector,
)
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from tests.v1.utils import (
    check_paged_kv_cache_equal_with_mla,
    check_sglang_paged_kv_cache_equal,
    generate_sglang_kv_cache_paged_list_tensors,
)


def _has_musa() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()  # type: ignore[attr-defined]


def _current_musa_device() -> torch.device:
    return torch.device("musa", torch.musa.current_device())  # type: ignore[attr-defined]


def _zero_kvcaches(kvcaches: object, use_mla: bool) -> None:
    if use_mla:
        for tensor in cast(list[torch.Tensor], kvcaches):
            tensor.zero_()
        return
    for layer_list in cast(list[list[torch.Tensor]], kvcaches):
        for tensor in layer_list:
            tensor.zero_()


def _as_non_layerwise_connector_input(
    kvcaches: object,
    use_mla: bool,
) -> object:
    """Match the legacy adapter's flat MHA input and MLA layer list."""
    if use_mla:
        return kvcaches
    key_layers, value_layers = cast(list[list[torch.Tensor]], kvcaches)
    return [*key_layers, *value_layers]


def _assert_kvcaches_equal(
    source: object,
    destination: object,
    slot_mapping: torch.Tensor,
    *,
    num_heads: int,
    head_size: int,
    use_mla: bool,
) -> None:
    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            source,
            destination,
            slot_mapping,
            head_size,
        )
        return
    check_sglang_paged_kv_cache_equal(
        source,
        destination,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )


@pytest.mark.parametrize(
    ("use_mla", "expected"),
    [
        (False, (2, 3, 8, 64)),
        (True, (3, 8, 64)),
    ],
)
def test_sglang_musa_connector_reports_memory_shape(
    use_mla: bool,
    expected: tuple[int, ...],
) -> None:
    """The public shape contract distinguishes MHA and MLA memory layouts."""
    connector = SGLangMUSAConnector(
        hidden_dim_size=64,
        num_layers=3,
        device=torch.device("cpu"),
        use_mla=use_mla,
    )

    assert tuple(connector.get_shape(8)) == expected


def test_sglang_layerwise_musa_rejects_mla() -> None:
    """Layerwise MLA fails clearly instead of using the MHA tensor layout."""
    with pytest.raises(NotImplementedError, match="set use_layerwise=False"):
        SGLangLayerwiseMUSAConnector(
            hidden_dim_size=64,
            num_layers=3,
            device=torch.device("cpu"),
            use_mla=True,
        )


def test_sglang_layerwise_musa_satisfies_layerwise_contract() -> None:
    """Cache-engine layerwise validation accepts the MUSA implementation."""
    connector = SGLangLayerwiseMUSAConnector(
        hidden_dim_size=64,
        num_layers=3,
        device=torch.device("cpu"),
    )

    assert_layerwise_gpu_connector(connector)


def test_sglang_musa_connector_rejects_non_musa_kv_cache() -> None:
    """A public transfer fails before applying MUSA operations to CPU caches."""
    connector = SGLangMUSAConnector(
        hidden_dim_size=4,
        num_layers=1,
        device=torch.device("cpu"),
    )
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=torch.zeros(2, 1, 1, 4),
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    # The legacy in-process adapter flattens MHA as [K0..Kn, V0..Vn].
    # A single local KV head must not be mistaken for MLA.
    kvcaches = [torch.zeros(4, 1, 4), torch.zeros(4, 1, 4)]

    with pytest.raises(ValueError, match="require MUSA KV-cache tensors"):
        connector.to_gpu(
            memory_obj,
            0,
            1,
            kvcaches=kvcaches,
            slot_mapping=torch.tensor([0]),
        )


@pytest.mark.skipif(not _has_musa(), reason="MUSA hardware is not available")
@pytest.mark.parametrize("use_mla", [False, True])
def test_sglang_musa_connector_roundtrip(use_mla: bool) -> None:
    """Non-layerwise MHA and MLA chunks round-trip through pinned CPU memory."""
    device = _current_musa_device()
    num_layers = 2
    num_blocks = 4
    block_size = 8
    num_heads = 1 if use_mla else 2
    head_size = 8
    num_tokens = 16
    hidden_dim_size = num_heads * head_size
    source = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )
    destination = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )
    _zero_kvcaches(destination, use_mla)
    source_input = _as_non_layerwise_connector_input(source, use_mla)
    destination_input = _as_non_layerwise_connector_input(destination, use_mla)
    slot_mapping = torch.randperm(
        num_blocks * block_size,
        device=device,
        dtype=torch.long,
    )[:num_tokens]
    connector = SGLangMUSAConnector(
        hidden_dim_size=hidden_dim_size,
        num_layers=num_layers,
        device=device,
        use_mla=use_mla,
    )
    allocator = PinMemoryAllocator(size=4 * 1024 * 1024)
    memory_obj = allocator.allocate(
        connector.get_shape(num_tokens),
        torch.bfloat16,
        MemoryFormat.KV_T2D,
    )
    assert memory_obj is not None

    try:
        connector.from_gpu(
            memory_obj,
            0,
            num_tokens,
            kvcaches=source_input,
            slot_mapping=slot_mapping,
        )
        expected_format = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD
        assert memory_obj.metadata.fmt == expected_format
        connector.to_gpu(
            memory_obj,
            0,
            num_tokens,
            kvcaches=destination_input,
            slot_mapping=slot_mapping,
        )
        torch.musa.synchronize()  # type: ignore[attr-defined]
        _assert_kvcaches_equal(
            source,
            destination,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
        )
    finally:
        memory_obj.ref_count_down()
        allocator.close()


@pytest.mark.skipif(not _has_musa(), reason="MUSA hardware is not available")
def test_sglang_musa_connector_respects_partial_slot_offset() -> None:
    """Non-layerwise MHA applies a nonzero prefix offset to a partial slot map."""
    device = _current_musa_device()
    num_layers = 2
    num_blocks = 4
    block_size = 8
    num_heads = 2
    head_size = 8
    num_tokens = 16
    offset = 5
    hidden_dim_size = num_heads * head_size
    source = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )
    destination = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )
    _zero_kvcaches(destination, use_mla=False)
    source_input = _as_non_layerwise_connector_input(source, use_mla=False)
    destination_input = _as_non_layerwise_connector_input(destination, use_mla=False)
    slot_mapping = torch.randperm(
        num_blocks * block_size,
        device=device,
        dtype=torch.long,
    )[:num_tokens]
    connector = SGLangMUSAConnector(
        hidden_dim_size=hidden_dim_size,
        num_layers=num_layers,
        device=device,
        use_mla=False,
    )
    allocator = PinMemoryAllocator(size=4 * 1024 * 1024)
    memory_obj = allocator.allocate(
        connector.get_shape(num_tokens),
        torch.bfloat16,
        MemoryFormat.KV_T2D,
    )
    assert memory_obj is not None

    try:
        connector.from_gpu(
            memory_obj,
            offset,
            offset + num_tokens,
            kvcaches=source_input,
            slot_mapping=slot_mapping,
            offset=offset,
        )
        connector.to_gpu(
            memory_obj,
            offset,
            offset + num_tokens,
            kvcaches=destination_input,
            slot_mapping=slot_mapping,
            offset=offset,
        )
        torch.musa.synchronize()  # type: ignore[attr-defined]
        _assert_kvcaches_equal(
            source,
            destination,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
        )
    finally:
        memory_obj.ref_count_down()
        allocator.close()


@pytest.mark.skipif(not _has_musa(), reason="MUSA hardware is not available")
def test_sglang_layerwise_musa_connector_roundtrip() -> None:
    """Layerwise MHA handles a partial slot map with a nonzero prefix."""
    device = _current_musa_device()
    num_layers = 2
    num_blocks = 4
    block_size = 8
    num_heads = 2
    head_size = 8
    num_tokens = 16
    offset = 5
    hidden_dim_size = num_heads * head_size
    source = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )
    destination = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )
    _zero_kvcaches(destination, use_mla=False)
    slot_mapping = torch.randperm(
        num_blocks * block_size,
        device=device,
        dtype=torch.long,
    )[:num_tokens]
    connector = SGLangLayerwiseMUSAConnector(
        hidden_dim_size=hidden_dim_size,
        num_layers=num_layers,
        device=device,
    )
    allocator = PinMemoryAllocator(size=4 * 1024 * 1024)
    memory_objs: list[list[MemoryObj]] = []
    for _ in range(num_layers):
        memory_obj = allocator.allocate(
            connector.get_shape(num_tokens),
            torch.bfloat16,
            MemoryFormat.KV_T2D,
        )
        assert memory_obj is not None
        memory_objs.append([memory_obj])

    try:
        gather = connector.batched_from_gpu(
            memory_objs,
            [offset],
            [offset + num_tokens],
            kvcaches=source,
            slot_mapping=slot_mapping,
            offset=offset,
        )
        next(gather)
        for _ in range(num_layers):
            next(gather)
        with pytest.raises(StopIteration):
            next(gather)

        scatter = connector.batched_to_gpu(
            [offset],
            [offset + num_tokens],
            kvcaches=destination,
            slot_mapping=slot_mapping,
            offset=offset,
        )
        next(scatter)
        for layer_memory_objs in memory_objs:
            scatter.send(layer_memory_objs)
        with pytest.raises(StopIteration):
            next(scatter)

        torch.musa.synchronize()  # type: ignore[attr-defined]
        _assert_kvcaches_equal(
            source,
            destination,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
        )
    finally:
        for layer_memory_objs in memory_objs:
            layer_memory_objs[0].ref_count_down()
        allocator.close()

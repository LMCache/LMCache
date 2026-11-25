# SPDX-License-Identifier: Apache-2.0
"""Layout coverage for the Mamba unified-state view in kv_cache_group_edits."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
import math

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="kv_cache_group_edits imports vLLM at module top")

# Third Party
from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402
from vllm.v1 import kv_cache_interface  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    MambaSpec,
)

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import (  # noqa: E402
    apply_kv_cache_group_edits,
)
from lmcache.integration.vllm.kv_cache_groups import (  # noqa: E402
    create_engine_group_infos_from_vllm,
)
from lmcache.v1.client.transfer_context.base import (  # noqa: E402
    gather_paged_kv_to_cpu,
)
from lmcache.v1.common.custom_types import EngineType  # noqa: E402
from lmcache.v1.common.kv_layer_groups import KVLayerGroupsManager  # noqa: E402
from lmcache.v1.common.utils import (  # noqa: E402
    LayoutHints,
    normalize_and_discover_per_layer_formats,
)

NUM_BLOCKS = 3
BLOCK_SIZE = 16
# Elements in one layer's unified state row, and per block including padding.
STATE_ROW = 8000
BLOCK_STEP = 8064
# ceil(STATE_ROW / BLOCK_SIZE) = 500, rounded up to the 16-byte (8 x bf16)
# alignment: 504 * 16 * 2 bytes fills the padded page exactly.
HEAD_SIZE = 504


def _mamba_config(
    spec: MambaSpec, layer_names: tuple[str, ...] = ("layer.0",)
) -> KVCacheConfig:
    """Build an align-mode Mamba group using either vLLM cache-group API."""
    if hasattr(kv_cache_interface, "KVCacheGroup"):
        # Third Party
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        # vllm-project/vllm#63830 replaced mamba_cache_mode with
        # enable_prefix_caching.
        if "mamba_cache_mode" in MambaManager.Params.__dataclass_fields__:
            params = MambaManager.Params(mamba_cache_mode="align")
        else:
            params = MambaManager.Params(enable_prefix_caching=True)
        group = kv_cache_interface.KVCacheGroup(
            kv_cache_specs=dict.fromkeys(layer_names, spec),
            manager_config=kv_cache_interface.ManagerConfig(MambaManager, params),
            block_size=spec.block_size,
        )
    else:
        group = kv_cache_interface.KVCacheGroupSpec(
            list(layer_names), replace(spec, mamba_cache_mode="align")
        )
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=[group]
    )


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
    )
    config = _mamba_config(spec)
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


def _unpadded_byte_state_layer(
    row: int, block_size: int
) -> tuple[KVCacheConfig, dict[str, torch.Tensor]]:
    """Build one align-mode Mamba layer whose page holds exactly ``row`` bytes.

    Mirrors Kimi K3.1's KDA state: a byte-typed row with no page padding,
    whose bytes per token (``row / block_size``) are odd or fractional.
    """
    storage = torch.zeros(NUM_BLOCKS * row, dtype=torch.uint8)
    kv_cache = storage.as_strided((NUM_BLOCKS, 1, 1, row), (row, row, row, 1))
    spec = MambaSpec(
        block_size=block_size,
        shapes=((row,),),
        dtypes=(torch.uint8,),
        page_size_padded=row,
    )
    config = _mamba_config(spec)
    return config, {"layer.0": kv_cache}


@pytest.mark.parametrize(
    ("row", "block_size", "num_slots"),
    [
        # 795 bytes per token: divides, but odd, so no aligned width fits.
        (64 * 795, 64, 4),
        # 397.5 bytes per token: does not divide.
        (32 * 795, 64, 2),
        # Kimi K3.1 KDA state at TP1 (2^13 x 795 bytes) under an 8192 block.
        (8192 * 795, 8192, 512),
    ],
)
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_mamba_unified_view_compresses_slots_when_tokens_do_not_tile(
    row: int, block_size: int, num_slots: int, kv_layout: str
) -> None:
    """Fewer, 16-byte-wide slots exactly cover a row that tokens cannot tile."""
    config, kv_caches = _unpadded_byte_state_layer(row, block_size)

    layout_hints = cast(LayoutHints, {"kv_layout": kv_layout})
    view = apply_kv_cache_group_edits(config, kv_caches, layout_hints)["layer.0"]

    assert isinstance(view, torch.Tensor)
    head_size = row // num_slots
    assert head_size % 16 == 0
    slot_dims = (num_slots, 1) if kv_layout == "NHD" else (1, num_slots)
    assert tuple(view.shape) == (NUM_BLOCKS, *slot_dims, head_size)
    assert view.stride(0) == row
    assert view.data_ptr() == kv_caches["layer.0"].data_ptr()
    # The view covers each block's row exactly, so no byte of the next
    # block (or a sibling layer) is reachable.
    view[1].fill_(1)
    flat = kv_caches["layer.0"].reshape(-1)
    assert int(flat[row : 2 * row].sum()) == row
    assert int(flat[:row].sum()) == 0 and int(flat[2 * row :].sum()) == 0


def test_mamba_unified_view_rejects_row_without_aligned_tiling() -> None:
    """An odd byte row with an odd block step has no aligned width at all."""
    config, kv_caches = _unpadded_byte_state_layer(795, 64)

    layout_hints = cast(LayoutHints, {"kv_layout": "NHD"})
    with pytest.raises(ValueError, match="cannot tile"):
        apply_kv_cache_group_edits(config, kv_caches, layout_hints)


@pytest.mark.parametrize("state_alignment", [1, 16])
def test_published_state_components_address_each_layer_in_the_chunk(
    state_alignment: int,
) -> None:
    """Slicing a gathered chunk with the published content yields exactly the
    states vLLM bound to each layer's page in each block, including when an
    unaligned first state pushes the next one to an aligned offset."""
    # A 600-byte bf16 conv state is not 16-byte aligned; the ssm state follows.
    shapes = ((3, 100), (4, 16, 16))
    dtypes = (torch.bfloat16, torch.float32)
    row = sum(
        -(-math.prod(shape) * dtype.itemsize // state_alignment) * state_alignment
        for shape, dtype in zip(shapes, dtypes, strict=False)
    )
    page = -(-(row + 512) // 64) * 64
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=shapes,
        dtypes=dtypes,
        state_alignment=state_alignment,
        page_size_padded=page,
    )
    names = ("layer.0", "layer.1")
    # vLLM registers each layer as an int8 [num_blocks, 1, 1, row] page view.
    storage = torch.zeros(len(names) * NUM_BLOCKS * page, dtype=torch.int8)
    kv_caches = {
        name: storage.as_strided(
            (NUM_BLOCKS, 1, 1, row),
            (page, row, row, 1),
            storage_offset=i * NUM_BLOCKS * page,
        )
        for i, name in enumerate(names)
    }
    states = {}
    for name in names:
        layer = SimpleNamespace(
            get_state_shape=lambda: shapes,
            get_state_dtype=lambda: dtypes,
            state_alignment=state_alignment,
        )
        MambaBase.bind_kv_cache(layer, kv_caches[name])
        for state in layer.kv_cache:
            state.copy_(torch.randn(state.shape).to(state.dtype))
        states[name] = layer.kv_cache
    config = _mamba_config(spec, names)
    layout_hints = cast(LayoutHints, {"kv_layout": "NHD"})
    edited = apply_kv_cache_group_edits(config, kv_caches, layout_hints)

    infos = create_engine_group_infos_from_vllm(config, edited, layout_hints)
    normalized, formats = normalize_and_discover_per_layer_formats(
        list(edited.values()),
        [info.layer_indices for info in infos],
        EngineType.VLLM,
        layout_hints,
    )
    (kernel_group,) = KVLayerGroupsManager(
        normalized, formats, infos, lmcache_tokens_per_chunk=2 * BLOCK_SIZE
    ).kernel_groups
    content = kernel_group.content_desc()
    # Two blocks in one chunk, as a window wider than one block stores them.
    blocks = [2, 0]
    (chunk,) = gather_paged_kv_to_cpu(edited, blocks, 2, layout_hints=layout_hints)

    assert len(content.state_components) == len(shapes)
    for layer_idx, name in enumerate(names):
        layer_bytes = chunk[layer_idx].reshape(-1)
        for slot, block in enumerate(blocks):
            page_start = slot * content.page_bytes
            for component, state in zip(
                content.state_components, states[name], strict=True
            ):
                start = page_start + component.byte_offset
                data = layer_bytes[start : start + component.byte_length]
                dtype = getattr(torch, component.dtype_str)
                got = data.view(dtype).view(component.shape)
                assert torch.equal(got, state[block])

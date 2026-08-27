# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)
import lmcache.lmcache_native as lmcache_native


def resolve_vllm_kv_layout(
    layout_hints: LayoutHints, cpu_attention_backend: bool
) -> str:
    """Validate the ``kv_layout`` hint and apply the CPU-backend override."""
    kv_layout = layout_hints.get("kv_layout", "NHD")
    if kv_layout not in ("NHD", "HND", "BLHNC", "BLNHC"):
        raise ValueError(
            f"kv_layout hint {kv_layout!r} is not a layout LMCache supports; "
            "expected one of NHD, HND, BLHNC, BLNHC."
        )
    # The CPU attention backend allocates HND but hints NHD.
    if cpu_attention_backend and kv_layout in ("NHD", "HND"):
        return "HND"
    return kv_layout


def _reconstruct_blocks_first(
    kv_caches: DiscoverableKVCache, kv_layout: str
) -> "tuple[lmcache_native.EngineKVFormat, torch.Tensor]":
    """Stack the per-layer views into one [num_blocks, num_layers, ...] tensor.

    kv_caches holds one rank-4 view per layer into a single buffer: all
    views share shape, strides, and storage, and layer i starts at
    base offset + i * layer_step. Verify that, then as_strided layer 0
    into a rank-5 tensor with the layer dim at position 1 (stride
    layer_step). Raise ValueError if the views don't match.
    """
    fmt = (
        lmcache_native.EngineKVFormat.NB_NL_NH_BS_CS
        if kv_layout == "BLHNC"
        else lmcache_native.EngineKVFormat.NB_NL_BS_NH_CS
    )
    if not (
        isinstance(kv_caches, list)
        and kv_caches
        and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in kv_caches)
    ):
        raise ValueError(
            f"{kv_layout} declared but kv_caches is not a list of rank-4 "
            "per-layer views."
        )
    layers = sorted(kv_caches, key=lambda t: t.storage_offset())
    base = layers[0]
    num_layers = len(layers)
    inner = tuple(base.shape[1:])
    chunk = 1
    for dim in inner:
        chunk *= int(dim)
    tight_inner = []
    stride = 1
    for dim in reversed(inner):
        tight_inner.append(stride)
        stride *= int(dim)
    tight_inner.reverse()

    def _drift(reason: str) -> ValueError:
        return ValueError(
            f"{kv_layout} declared but the registered views disagree: "
            f"{reason}. shape={tuple(base.shape)}, stride={base.stride()}, "
            f"num_layers={num_layers}."
        )

    if tuple(base.stride()[1:]) != tuple(tight_inner):
        raise _drift("per-(layer, block) content is not contiguous")
    # >= not ==: HMA pools put other groups' bytes between this group's
    # layers within each block.
    layer_step = (
        layers[1].storage_offset() - layers[0].storage_offset()
        if num_layers > 1
        else chunk
    )
    if layer_step < chunk:
        raise _drift(f"layer step {layer_step} < per-layer block chunk {chunk}")
    if base.stride(0) < (num_layers - 1) * layer_step + chunk:
        raise _drift(
            f"block step {base.stride(0)} cannot hold {num_layers} layers "
            f"{layer_step} apart"
        )
    storage_ptr = base.untyped_storage().data_ptr()
    for i, t in enumerate(layers):
        if t.untyped_storage().data_ptr() != storage_ptr:
            raise _drift("layer views do not share one storage")
        if tuple(t.shape) != tuple(base.shape) or t.stride() != base.stride():
            raise _drift(f"layer {i} shape/stride mismatch")
        if t.storage_offset() != base.storage_offset() + i * layer_step:
            raise _drift(f"layer {i} is not at offset base + {i} * layer step")

    full = base.as_strided(
        (base.shape[0], num_layers, *inner),
        (base.stride(0), layer_step, *tight_inner),
        storage_offset=base.storage_offset(),
    )
    return fmt, full


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> "tuple[Optional[lmcache_native.EngineKVFormat], DiscoverableKVCache]":
        kv_layout = resolve_vllm_kv_layout(
            layout_hints, cpu_attention_backend=torch_device_type == "cpu"
        )
        if kv_layout in ("BLHNC", "BLNHC"):
            return _reconstruct_blocks_first(kv_caches, kv_layout)
        is_hnd = kv_layout == "HND"

        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (a 5-D split would collide with
        # flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD) -- indistinguishable from the shape alone, so the
        # resolved kv_layout decides. The tensor is kept raw: the trailing axis
        # is the per-head content size (2 * head_size, K/V packed).
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            if is_hnd:
                return lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS, kv_caches

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        if list_depth == 0:
            return lmcache_native.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        # vLLM-RBLN: HND with a singleton between heads and block tokens that
        # its attention backend requires. Always HND, so the hint is not read.
        if (
            list_depth == 1
            and tensor_ndim == 6
            and first_tensor.shape[0] == 2
            and first_tensor.shape[3] == 1
        ):
            return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA (or DSA indexer cache)
            if first_tensor.dtype == torch.uint8 and int(first_tensor.shape[-1]) == 132:
                return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        if list_depth == 2 and tensor_ndim == 4 and len(kv_caches[0]) == 2:
            return lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS, kv_caches
        return None, kv_caches

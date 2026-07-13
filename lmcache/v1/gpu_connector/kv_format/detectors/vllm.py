# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr,list-item,assignment,return-value,arg-type"
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
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


def _as_split_kv_layers(
    kv_caches: DiscoverableKVCache,
) -> "Optional[tuple[list[torch.Tensor], list[torch.Tensor]]]":
    """Recognize a per-layer split-KV registration and regroup it.

    Returns ``(key_layers, value_layers)`` when *kv_caches* is a per-layer list
    where every entry is a ``(key, value)`` pair (2-tuple or 2-list) of 4-D
    tensors, else ``None``. The two tensors are kept as-is (no copy, no fuse);
    they are just regrouped from per-layer pairs into a key-list and a
    value-list so the existing ``TWO_X_NL_X_*`` split machinery can consume them.

    A ``(key, value)`` pair is required to have matching shape/dtype/device and
    rank 4 (``[num_blocks, block_size, num_heads, head_dim]``); anything else
    returns ``None`` so fused and other layouts fall through unchanged.
    """
    if not isinstance(kv_caches, list) or not kv_caches:
        return None
    key_layers: list[torch.Tensor] = []
    value_layers: list[torch.Tensor] = []
    for entry in kv_caches:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            return None
        key, value = entry
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return None
        if key.dim() != 4 or value.dim() != 4:
            return None
        if key.shape != value.shape:
            raise ValueError(
                f"split KV key shape {tuple(key.shape)} != value shape "
                f"{tuple(value.shape)}"
            )
        if key.dtype != value.dtype:
            raise ValueError(
                f"split KV key dtype {key.dtype} != value dtype {value.dtype}"
            )
        key_layers.append(key)
        value_layers.append(value)
    return key_layers, value_layers


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # Split K/V: some backends expose the key and value caches as two
        # independent tensors per layer instead of one fused tensor -- e.g. a
        # host-memory backend that hands LMCache zero-copy views of separate
        # native key/value arrays (avoiding a fused staging copy). Such a
        # backend registers each layer as a ``(key, value)`` pair, which arrives
        # here as a per-layer list of 2-tuples/2-lists of 4-D [NB, BS, NH, HS]
        # tensors. Regroup into the ``[K_layers, V_layers]`` structure LMCache's
        # existing split (``TWO_X_NL_X_*``) format already handles -- no fused
        # tensor is ever allocated; K and V stay separate end to end.
        split = _as_split_kv_layers(kv_caches)
        if split is not None:
            key_layers, value_layers = split
            # Split K/V is physically NHD ([NB, BS, NH, HS]); an explicit HND
            # hint is rejected below only for the fused paths. Honor the hint
            # for symmetry but there is a single supported split layout today.
            return (
                lmc_ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
                [key_layers, value_layers],
            )

        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (the post-split 5-D shape would collide
        # with flash-infer when num_heads == 2). Split [NB, NH, BS, 2*HS] into
        # [NB, NH, BS, 2, HS].
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            fused_dim = kv_caches[0].shape[3]
            if fused_dim % 2 != 0:
                raise ValueError(
                    f"blocks-first fused trailing dim {fused_dim} is not 2 * head_size"
                )
            split = [t.reshape(*t.shape[:3], 2, fused_dim // 2) for t in kv_caches]
            return lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS, split

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        # vLLM's x86 CPU attention backend stores KV in HND but misreports it,
        # so CPU defaults to HND. However, other host-memory backends that also
        # present as the "cpu" torch device -- notably vLLM's Apple Metal
        # (vllm-metal) plugin, whose MLX unified-memory KV cache is NHD -- must
        # be able to override that. An explicit ``layout_hints["kv_layout"]``
        # therefore always wins; only when no hint is supplied does CPU fall
        # back to HND (and every other device to NHD).
        kv_layout = layout_hints.get("kv_layout")
        if kv_layout is None:
            kv_layout = "HND" if torch_device_type == "cpu" else "NHD"
        is_hnd = kv_layout == "HND"

        if list_depth == 0:
            return lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches

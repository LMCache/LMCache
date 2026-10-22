# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    KV_LAYOUT_NAMES,
    DiscoverableKVCache,
    LayoutHints,
    is_blocked_scale_row,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def resolve_vllm_kv_layout(
    layout_hints: LayoutHints, cpu_attention_backend: bool
) -> str:
    """Validate an explicit ``kv_layout`` hint or choose a legacy default."""
    kv_layout = layout_hints.get("kv_layout")
    if kv_layout is None:
        # Registrations predating layout hints relied on the CPU backend's HND
        # allocation and the NHD default everywhere else.
        return "HND" if cpu_attention_backend else "NHD"
    if kv_layout not in KV_LAYOUT_NAMES:
        raise ValueError(
            f"kv_layout hint {kv_layout!r} is not a layout LMCache supports; "
            f"expected one of {', '.join(KV_LAYOUT_NAMES)}."
        )
    return kv_layout


def _is_blocked_scale_plane(t: torch.Tensor) -> bool:
    """Whether a per-layer uint8 plane is a blocked-scale page (DSA indexer K,
    DeepSeek-V4 packed main KV; see ``BLOCKED_SCALE_ROWS``)."""
    return t.dtype == torch.uint8 and is_blocked_scale_row(int(t.shape[-1]))


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
        is_hnd = kv_layout in ("HND", "BLHNC")

        # Fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (a 5-D split would collide with
        # flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD) -- indistinguishable from the shape alone, so the
        # resolved kv_layout decides. The tensor is kept raw: the trailing axis
        # is the per-head content size (2 * head_size, K/V packed). Blocks-first
        # views (BLHNC / BLNHC) have the same per-layer shape and differ only
        # in stride(0), which resolve_block_stride_and_log_layout reads.
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            # MLA planes (DSA indexer K, DeepSeek-V4 packed KV) are [B, H=1, N, C]
            # under either layout hint: squeeze that axis so the page gets its format.
            shape = kv_caches[0].shape
            head_axis = 1 if int(shape[1]) == 1 else 2
            if int(shape[head_axis]) == 1 and _is_blocked_scale_plane(kv_caches[0]):
                return (
                    lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS,
                    [t.squeeze(head_axis) for t in kv_caches],
                )
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
            if first_tensor.shape[1] == 2:
                logger.warning(
                    "RBLN KV cache shape %s fits both the K/V-first and the "
                    "blocks-first layout (num_blocks == 2); assuming K/V-first.",
                    tuple(first_tensor.shape),
                )
            return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
        # vLLM-RBLN's default attention kernels allocate the same layout
        # blocks-first. NB == 2 matches the branch above too, which keeps the
        # K/V-first reading and warns.
        if (
            list_depth == 1
            and tensor_ndim == 6
            and first_tensor.shape[1] == 2
            and first_tensor.shape[3] == 1
        ):
            return lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS, kv_caches
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
            if _is_blocked_scale_plane(first_tensor):
                return lmcache_native.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
            return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        if list_depth == 2 and tensor_ndim == 4:
            layer0_planes = kv_caches[0]
            single_head = all(int(t.shape[2]) == 1 for t in layer0_planes)
            widths = {int(t.shape[-1]) for t in layer0_planes}
            if single_head and (len(layer0_planes) in (1, 3) or len(widths) > 1):
                return (
                    lmcache_native.EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,
                    kv_caches,
                )
            if len(layer0_planes) == 2:
                return (
                    lmcache_native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
                    kv_caches,
                )
        return None, kv_caches

# SPDX-License-Identifier: Apache-2.0
"""Optional native MUSA KV-transfer adapter for LMCache.

The adapter is deliberately fail-closed. It never makes ``musa_aiter`` a
required dependency and returns ``False`` whenever native dispatch is not
available, so callers can keep using the torch implementation as the fallback.
"""

# Standard
from importlib import import_module
from typing import Any
import os

# Third Party
import torch

ENV_MUSA_NATIVE_KV_TRANSFER = "LMCACHE_MUSA_NATIVE_KV_TRANSFER"
NATIVE_LMCACHE_KV_TRANSFER_ABI_VERSION = 1

_REQUIRED_NATIVE_SYMBOLS = (
    "native_lmcache_kv_transfer_abi_version",
    "lmcache_kv_paged_to_buffer",
    "lmcache_kv_buffer_to_paged",
    "lmcache_mla_paged_to_buffer",
    "lmcache_mla_buffer_to_paged",
)


def is_native_musa_kv_transfer_enabled() -> bool:
    """Return whether LMCache should try optional native MUSA KV transfer."""
    return os.environ.get(ENV_MUSA_NATIVE_KV_TRANSFER, "").lower() in {
        "1",
        "true",
        "yes",
    }


def load_native_musa_module() -> Any | None:
    """Import optional ``musa_aiter``, returning ``None`` when unavailable."""
    try:
        return import_module("musa_aiter")
    except Exception:
        return None


def check_native_abi(module: Any) -> bool:
    """Return whether ``module`` exposes the Stage2 LMCache KV-transfer ABI."""
    for name in _REQUIRED_NATIVE_SYMBOLS:
        if not callable(getattr(module, name, None)):
            return False
    try:
        version = int(module.native_lmcache_kv_transfer_abi_version())
    except Exception:
        return False
    return version == NATIVE_LMCACHE_KV_TRANSFER_ABI_VERSION


def _is_musa_contiguous_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor can be passed directly to MUSA native kernels."""
    return tensor.device.type == "musa" and tensor.is_contiguous()


def _native_tensors_ready(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> bool:
    """Return whether native MUSA dispatch can consume these tensors directly."""
    return (
        _is_musa_contiguous_tensor(memory_tensor)
        and _is_musa_contiguous_tensor(slot_mapping)
        and all(_is_musa_contiguous_tensor(kvcache) for kvcache in kvcaches)
    )


def try_native_to_gpu(
    *,
    use_mla: bool,
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Try native contiguous-buffer-to-paged-KV scatter.

    Args:
        use_mla: Whether the active layout is MLA.
        memory_tensor: LMCache contiguous memory object tensor.
        kvcaches: vLLM paged KV tensors.
        slot_mapping: Full vLLM slot mapping tensor.
        start: Inclusive token start for this transfer.
        end: Exclusive token end for this transfer.
        skip_prefix_n_tokens: Prefix tokens already cached by vLLM.
        block_size: vLLM paged KV block size.
        num_heads: Number of KV heads for non-MLA layouts.
        head_size: KV head size or MLA hidden size.

    Returns:
        ``True`` when native dispatch completed and the caller should skip the
        torch fallback. ``False`` when native dispatch is disabled, unavailable,
        ABI-incompatible, when tensors are not contiguous MUSA tensors, or when
        the native module rejects the transfer.
    """
    module = _native_module_if_ready()
    if module is None:
        return False

    transfer_start = start + skip_prefix_n_tokens
    if transfer_start >= end:
        return True
    if not _native_tensors_ready(memory_tensor, kvcaches, slot_mapping):
        return False

    slot_slice = slot_mapping[transfer_start:end]
    try:
        if use_mla:
            return bool(
                module.lmcache_mla_buffer_to_paged(
                    memory_tensor,
                    kvcaches,
                    slot_slice,
                    skip_prefix_n_tokens,
                    block_size,
                    head_size,
                )
            )
        return bool(
            module.lmcache_kv_buffer_to_paged(
                memory_tensor,
                kvcaches,
                slot_slice,
                skip_prefix_n_tokens,
                block_size,
                num_heads,
                head_size,
            )
        )
    except Exception:
        return False


def try_native_from_gpu(
    *,
    use_mla: bool,
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Try native paged-KV-to-contiguous-buffer gather.

    Args:
        use_mla: Whether the active layout is MLA.
        memory_tensor: LMCache contiguous memory object tensor to populate.
        kvcaches: vLLM paged KV tensors.
        slot_mapping: Full vLLM slot mapping tensor.
        start: Inclusive token start for this transfer.
        end: Exclusive token end for this transfer.
        block_size: vLLM paged KV block size.
        num_heads: Number of KV heads for non-MLA layouts.
        head_size: KV head size or MLA hidden size.

    Returns:
        ``True`` when native dispatch completed and the caller should skip the
        torch fallback. ``False`` when native dispatch is unavailable, when
        tensors are not contiguous MUSA tensors, or when native dispatch fails.
    """
    module = _native_module_if_ready()
    if module is None:
        return False
    if start >= end:
        return True
    if not _native_tensors_ready(memory_tensor, kvcaches, slot_mapping):
        return False

    slot_slice = slot_mapping[start:end]
    try:
        if use_mla:
            return bool(
                module.lmcache_mla_paged_to_buffer(
                    kvcaches,
                    memory_tensor,
                    slot_slice,
                    block_size,
                    head_size,
                )
            )
        return bool(
            module.lmcache_kv_paged_to_buffer(
                kvcaches,
                memory_tensor,
                slot_slice,
                block_size,
                num_heads,
                head_size,
            )
        )
    except Exception:
        return False


def _native_module_if_ready() -> Any | None:
    """Return a usable native module when opt-in and ABI-compatible."""
    if not is_native_musa_kv_transfer_enabled():
        return None
    module = load_native_musa_module()
    if module is None or not check_native_abi(module):
        return None
    return module

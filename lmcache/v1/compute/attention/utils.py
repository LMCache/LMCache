# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Local
from .flash_attn import LMCFlashAttnBackend


def _is_rocm() -> bool:
    """Check if we're running on ROCm (AMD GPU)."""
    return torch.version.hip is not None


def _flashinfer_available() -> bool:
    """Check if flashinfer is importable."""
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError:
        return False


def infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
    attn_name = type(vllm_attn.impl).__name__

    if enable_sparse:
        # On ROCm or when flashinfer is unavailable, use Triton backend
        if _is_rocm() or not _flashinfer_available():
            from .triton_sparse import LMCTritonSparseBackend

            logger.info(
                "Using LMCTritonSparseBackend for CacheBlend "
                f"(ROCm={_is_rocm()}, flashinfer={_flashinfer_available()})"
            )
            return LMCTritonSparseBackend(vllm_attn)

        # On CUDA with flashinfer available, use flashinfer backend
        from .flash_infer_sparse import LMCFlashInferSparseBackend

        if attn_name == "FlashInferImpl":
            return LMCFlashInferSparseBackend(vllm_attn)
        else:
            # Fallback to Triton even on CUDA if flashinfer impl doesn't match
            from .triton_sparse import LMCTritonSparseBackend

            logger.info(
                f"Attention impl {attn_name} is not FlashInferImpl; "
                "falling back to LMCTritonSparseBackend"
            )
            return LMCTritonSparseBackend(vllm_attn)

    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        return LMCFlashAttnBackend(vllm_attn)
    else:
        raise ValueError(
            f"Attention backend {attn_name} is not supported in LMCache. "
            f"enable_sparse={enable_sparse}"
        )

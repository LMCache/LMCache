# SPDX-License-Identifier: Apache-2.0
# Local
from .flash_attn import LMCFlashAttnBackend

# Removed static import of LMCFlashInferSparseBackend to prevent
# engine initialization failure when the environment is inconsistent.


def infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
    attn_name = type(vllm_attn.impl).__name__
    if attn_name == "FlashInferImpl" and enable_sparse:
        # Use lazy import to prevent engine initialization failure caused by
        # environment or version mismatches in unused dependencies.
        # This ensures the engine remains stable unless this specific
        # backend is explicitly required at runtime.
        # Local
        from .flash_infer_sparse import LMCFlashInferSparseBackend

        return LMCFlashInferSparseBackend(vllm_attn)
    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        return LMCFlashAttnBackend(vllm_attn)
    else:
        raise ValueError(f"Attention backend {attn_name} is not supported in LMCache.")

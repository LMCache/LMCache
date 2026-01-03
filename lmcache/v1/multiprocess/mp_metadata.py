# SPDX-License-Identifier: Apache-2.0
"""
MP Server Metadata Utilities

Provides utilities for creating LMCacheEngineMetadata for the MP server.
The MP server runs as a single process managing storage for all TP workers,
so it uses world_size=1 regardless of the vLLM tensor parallel configuration.
"""

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata


def create_mp_server_metadata(
    model_name: str,
    num_layers: int,
    hidden_dim: int,
    kv_dtype: torch.dtype,
    chunk_size: int = 256,
    use_mla: bool = False,
) -> LMCacheEngineMetadata:
    """
    Create LMCacheEngineMetadata for the MP server.

    The MP server is a single process that manages storage for all TP workers.
    Therefore, it uses world_size=1 to allow storage backends like
    RustRawBlockBackend that don't support TP > 1.

    Args:
        model_name: Name of the model being served
        num_layers: Number of layers in the model
        hidden_dim: Hidden dimension size (num_heads * head_size)
        kv_dtype: Data type for KV cache tensors
        chunk_size: LMCache chunk size in tokens
        use_mla: Whether the model uses Multi-head Latent Attention

    Returns:
        LMCacheEngineMetadata configured for MP server
    """
    kv_size = 1 if use_mla else 2
    fmt = "KV_MLA_FMT" if use_mla else "KV_2LTD"

    return LMCacheEngineMetadata(
        model_name=model_name,
        world_size=1,  # Single process manages storage
        worker_id=0,
        fmt=fmt,
        kv_dtype=kv_dtype,
        kv_shape=(num_layers, kv_size, chunk_size, 1, hidden_dim),
        use_mla=use_mla,
        role="mp_server",
        chunk_size=chunk_size,
    )


def create_mp_server_metadata_from_gpu_context(
    model_name: str,
    num_layers: int,
    hidden_dim: int,
    kv_dtype: torch.dtype,
    chunk_size: int,
    is_mla: bool,
) -> LMCacheEngineMetadata:
    """
    Create LMCacheEngineMetadata from GPU context information.

    This is called when the first vLLM instance registers its KV cache
    with the MP server.

    Args:
        model_name: Name of the model
        num_layers: Number of layers from GPUCacheContext
        hidden_dim: Hidden dimension from GPUCacheContext
        kv_dtype: KV cache dtype from GPUCacheContext
        chunk_size: LMCache chunk size
        is_mla: Whether model uses MLA from GPUCacheContext

    Returns:
        LMCacheEngineMetadata configured for MP server
    """
    return create_mp_server_metadata(
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        kv_dtype=kv_dtype,
        chunk_size=chunk_size,
        use_mla=is_mla,
    )

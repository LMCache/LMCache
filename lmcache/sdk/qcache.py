# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving Q tensors.
"""

# Future
from __future__ import annotations

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.sdk.context import LMCacheSDKCacheKind, LMCacheSDKContext

logger = init_logger(__name__)


def connect(
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
    device: str | torch.device = torch_device_type,
    pool_chunks: int | None = None,
    sw_size_tokens: int | None = None,
) -> LMCacheSDKContext:
    """Connect to the LMCache server and return a context for Q cache.
    The technique to get the query tensors are the same as KV cache,
    however, specific model name prefix needs to be used.

    Args:
        url: Multiprocess request endpoint. Its scheme selects ZMQ or gRPC.
        http_url: The HTTP URL of the LMCache server.
        model_name: The original model name.
        timeout: The timeout for the connection.
        device: Accelerator device for the transfer pool. Defaults to the
            current device.
        pool_chunks: LMCache chunks the transfer pool holds; the default
            uses the server's ``sdk.pool_chunks``, else 4.
        sw_size_tokens: Sliding window of the query tensors, a multiple of
            the block size; -1 for the whole decoded segment. The default
            uses the server's ``sdk.q_sw_size_tokens``, else -1. Must match
            the vLLM Q ring's ``lmcache.mp.q.sw_size_tokens``.

    Returns:
        An LMCacheSDKContext instance for Q cache.
    """
    ctx = LMCacheSDKContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        kind=LMCacheSDKCacheKind.QUERY,
        timeout=timeout,
        device=device,
        pool_chunks=pool_chunks,
        sw_size_tokens=sw_size_tokens,
    )
    ctx.register_caches()
    return ctx

# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving and storing KV cache tensors.
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
) -> "LMCacheSDKContext":
    """Create and initialize the LMCache SDK context.

    Args:
        url: Multiprocess request endpoint. Its scheme selects ZMQ or gRPC.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
        device: Accelerator device for the transfer pool. Defaults to the
            current device.
        pool_chunks: LMCache chunks the transfer pool holds; the default
            uses the server's ``sdk.pool_chunks``, else 4.

    Returns:
        An initialized LMCacheSDKContext instance.
        Ready to be passed to close(), retrieve(), and store() functions.
    """
    ctx = LMCacheSDKContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        kind=LMCacheSDKCacheKind.KV,
        timeout=timeout,
        device=device,
        pool_chunks=pool_chunks,
    )
    ctx.register_caches()
    return ctx

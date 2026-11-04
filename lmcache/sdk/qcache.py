# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving Q tensors.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.logging import init_logger
from lmcache.sdk.context import LMCacheSDKCacheKind, LMCacheSDKContext

logger = init_logger(__name__)


def connect(
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
) -> LMCacheSDKContext:
    """Connect to the LMCache server and return a context for Q cache.

    Query tensors are retrieved through free blocks of vLLM's Q ring, with the
    window the ring is registered with (``lmcache.mp.q.sw_size_tokens``). vLLM
    must take no ring blocks (schedule no requests) while the SDK retrieves.

    Args:
        url: Multiprocess request endpoint. Its scheme selects ZMQ or gRPC.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

    Returns:
        An LMCacheSDKContext instance for Q cache.
    """
    ctx = LMCacheSDKContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        kind=LMCacheSDKCacheKind.QUERY,
        timeout=timeout,
    )
    ctx.register_caches()
    return ctx

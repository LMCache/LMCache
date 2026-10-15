# SPDX-License-Identifier: Apache-2.0
"""Public LMCache SDK helpers."""

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.sdk import batch, context, kvcache, qcache, request
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.context import FULL_WINDOW

__all__ = [
    "batch",
    "context",
    "kvcache",
    "qcache",
    "request",
    "LMCacheSDKCacheKind",
    "FULL_WINDOW",
]


def connect(
    kind: LMCacheSDKCacheKind,
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
    device: str | torch.device = torch_device_type,
    pool_chunks: int | None = None,
    sw_size_tokens: int | None = None,
) -> context.LMCacheSDKContext:
    """Connect to the LMCache server and return a context for the given kind.
    Calls ``kvcache.connect`` or ``qcache.connect`` based on ``kind``.

    Args:
        kind: Which cache to connect to (``LMCacheSDKCacheKind.KV`` or
            ``LMCacheSDKCacheKind.QUERY``).
        url: Multiprocess request endpoint. Its scheme selects ZMQ or gRPC.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
        device: GPU device to use for temporary staging buffer.
        pool_chunks: number of LMCache chunks for staging buffer.
        sw_size_tokens: Sliding window size for QUERY kind.

    Returns:
        An initialized LMCacheSDKContext for the requested kind.

    Raises:
        ValueError: If ``kind`` is not a supported cache kind, or a window
            is requested for the KV kind.
    """
    if kind == LMCacheSDKCacheKind.KV and sw_size_tokens is not None:
        raise ValueError("sw_size_tokens only applies to the QUERY kind")
    if kind == LMCacheSDKCacheKind.KV:
        return kvcache.connect(
            url=url,
            http_url=http_url,
            model_name=model_name,
            timeout=timeout,
            device=device,
            pool_chunks=pool_chunks,
        )
    if kind == LMCacheSDKCacheKind.QUERY:
        return qcache.connect(
            url=url,
            http_url=http_url,
            model_name=model_name,
            timeout=timeout,
            device=device,
            pool_chunks=pool_chunks,
            sw_size_tokens=sw_size_tokens,
        )
    raise ValueError(f"unsupported cache kind: {kind}")

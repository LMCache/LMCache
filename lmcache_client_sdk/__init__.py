# SPDX-License-Identifier: Apache-2.0
"""A lightweight, standalone LMCache client SDK for inference engines.

This package exposes the engine-facing client contract of the LMCache
multiprocess server: connect, store, retrieve, and close. It is intentionally
minimal — no offline token-dropping orchestration and no cache-dropping (block
release) operations — so the API surface stays frozen and upward compatible.
"""

# First Party
from lmcache_client_sdk._version import __version__  # noqa: F401
from lmcache_client_sdk.context import LMCacheClientContext

__all__ = ["LMCacheClientContext", "connect"]


def connect(
    url: str,
    http_url: str,
    model_name: str,
    timeout: float = 60.0,
) -> LMCacheClientContext:
    """Connect to an LMCache server and return a client context.

    Args:
        url: Multiprocess request endpoint. Its scheme selects the transport
            (``tcp://`` for ZMQ).
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.

    Returns:
        An initialized :class:`LMCacheClientContext` ready for ``store()`` and
        ``retrieve()`` calls.

    Raises:
        LMCacheClientError: If the server is unreachable, the model layout is
            not registered, or engine-driven registration fails.
    """
    return LMCacheClientContext(
        url=url,
        http_url=http_url,
        model_name=model_name,
        timeout=timeout,
    )

# SPDX-License-Identifier: Apache-2.0
"""Safe construction of coordinator-to-server HTTP URLs."""

# Third Party
import httpx

# First Party
from lmcache.v1.mp_coordinator.views.instance_registry import MPInstance


def instance_url(target: MPInstance, path: str) -> httpx.URL:
    """Build an HTTP URL for one registered MP server.

    The instance host and port are supplied as structured URL components so
    neither can change the request path or inject other authority syntax.

    Args:
        target: Registered MP server that should receive the request.
        path: Absolute resource path on that server.

    Returns:
        An HTTPX URL with the target as its authority and ``path`` as its path.
    """
    return httpx.URL(
        scheme="http",
        host=target.ip,
        port=target.http_port,
        path=path,
    )

# SPDX-License-Identifier: Apache-2.0
"""gRPC endpoint addressing and call metadata shared by clients and servers."""

# Standard
from urllib.parse import urlparse

# Each client sends a random ID under this key, and servers use it to run one
# client's affinity calls on one worker, so both sides must use the same key.
CLIENT_ID_METADATA_KEY = "lmcache-client-id-bin"


def parse_grpc_target(server_url: str) -> str:
    """Convert an LMCache multiprocess endpoint into a gRPC target.

    Args:
        server_url: gRPC endpoint.

    Returns:
        A target accepted by ``grpc.insecure_channel``.

    Raises:
        ValueError: If the URL scheme or target is invalid.
    """
    if "://" not in server_url:
        if not server_url:
            raise ValueError("gRPC server target must not be empty")
        return server_url

    parsed = urlparse(server_url)
    if parsed.scheme == "grpc":
        if not parsed.netloc:
            raise ValueError(f"Missing host in gRPC URL: {server_url!r}")
        return parsed.netloc
    if parsed.scheme == "grpc+unix":
        path = f"/{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        if not path:
            raise ValueError(f"Missing socket path in gRPC URL: {server_url!r}")
        return f"unix://{path}"
    raise ValueError(f"Unsupported gRPC URL scheme: {parsed.scheme!r}")

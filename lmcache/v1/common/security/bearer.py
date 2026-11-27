# SPDX-License-Identifier: Apache-2.0
"""HTTP Bearer authorization helpers shared by HTTP and gRPC adapters."""


def bearer_token(authorization: str | None) -> str | None:
    """Extract a token from an HTTP Bearer authorization value.

    Args:
        authorization: Complete ``Authorization`` header value.

    Returns:
        The Bearer token, or ``None`` for a missing or malformed value.
    """
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return None
    return token


def bearer_authorization_header(api_key: str) -> str:
    """Build an HTTP Bearer authorization value from one API key.

    Args:
        api_key: Plaintext API key to send to an HTTP service.

    Returns:
        A value suitable for the HTTP ``Authorization`` header.
    """
    return f"Bearer {api_key}"

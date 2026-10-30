# SPDX-License-Identifier: Apache-2.0
"""Authentication configuration shared by multiprocess request transports."""

# Standard
from collections.abc import Sequence
from pathlib import Path

# First Party
from lmcache.security.api_key import (
    APIKeyVerifier,
    load_api_keys_from_files,
    load_service_api_keys,
)
from lmcache.security.http import bearer_authorization_header


def load_request_authentication(
    key_files: Sequence[str | Path],
) -> APIKeyVerifier:
    """Load accepted request-transport API keys.

    Args:
        key_files: Server-side key files supplied through the command line.

    Returns:
        A verifier containing all CLI and environment keys.
    """
    return APIKeyVerifier(load_service_api_keys(key_files))


def load_request_client_api_key(
    api_key: str | None = None,
    key_files: Sequence[str | Path] = (),
) -> str | None:
    """Load the one API key sent by a request client.

    An explicit value takes precedence. Otherwise the first configured CLI key
    file is used, followed by the shared environment sources. Additional keys
    remain valid server-side rotation keys, while clients send one credential.

    Args:
        api_key: Explicit client credential, if supplied programmatically.
        key_files: API key files configured for the service. The first key is
            used for outbound requests; all keys remain accepted inbound.

    Returns:
        The configured key, or ``None`` when authentication is not configured.

    Raises:
        OSError: If an environment key file cannot be read.
        ValueError: If both environment sources are configured.
    """
    if api_key is not None:
        if not api_key:
            raise ValueError("Request client API key must not be empty")
        return api_key
    if key_files:
        api_keys = load_api_keys_from_files(key_files)
        return api_keys[0] if api_keys else None
    api_keys = load_service_api_keys(())
    if len(api_keys) > 1:
        raise ValueError("Configure exactly one request client API key source")
    return api_keys[0] if api_keys else None


def request_client_auth_headers(
    key_files: Sequence[str | Path] = (),
    api_key: str | None = None,
) -> dict[str, str]:
    """Build outbound HTTP headers carrying the single configured client key.

    Used by the MP server's calls to an authenticated coordinator (instance
    registration, heartbeats, event publishing, chunk-size negotiation) and by
    the coordinator's calls to authenticated MP servers.

    Returns:
        An ``Authorization: Bearer`` header when one client key source is
        configured, or an empty mapping when none is. The empty mapping keeps
        the no-key default: a fleet without a configured key exchanges no
        credentials.
    """
    api_key = load_request_client_api_key(api_key, key_files)
    if api_key is None:
        return {}
    return {"Authorization": bearer_authorization_header(api_key)}

# SPDX-License-Identifier: Apache-2.0
"""HTTP authentication configuration for the mp coordinator.

The coordinator is the fleet-level control plane: its HTTP endpoints register
instances, mutate quotas and pins, inject cache events, and expose the key
directory. When an operator configures an API key, every non-public route must
present it. The shared primitives in :mod:`lmcache.security` are reused here so
the coordinator, the MP HTTP frontend, and the request transports accept one
credential set (``--api-key-file`` / ``LMCACHE_API_KEY_FILE`` /
``LMCACHE_API_KEY``) instead of each service growing its own.
"""

# Standard
from collections.abc import Sequence
from pathlib import Path

# Third Party
from starlette.types import ASGIApp
from typing_extensions import Self

# First Party
from lmcache.security.api_key import (
    APIKeyVerifier,
    load_api_keys_from_files,
    load_service_api_keys,
)
from lmcache.security.http import (
    HTTPBearerAuthenticationMiddleware,
    bearer_token,
)

# Paths that stay reachable without credentials even when authentication is on.
# ``/healthz`` is the orchestrator liveness probe: it returns only
# ``{"status": "healthy"}`` and must not force probe configuration through the
# fleet key. Everything else -- fleet membership, quotas, events, the key
# directory, and metrics -- requires a valid API key.
PUBLIC_COORDINATOR_PATHS = frozenset({"/healthz"})


class CoordinatorAuthenticationConfig(APIKeyVerifier):
    """API keys accepted by the mp coordinator fleet control plane."""

    def __init__(self, api_keys: Sequence[str] = ()) -> None:
        """Initialize accepted keys and select the first for outbound calls.

        Args:
            api_keys: Keys accepted by the coordinator. The first configured
                key is also used when the coordinator calls MP servers.
        """
        super().__init__(api_keys)
        self._client_api_key = api_keys[0] if api_keys else None

    def client_auth_headers(self) -> dict[str, str]:
        """Build credentials for the coordinator's outbound MP requests.

        Returns:
            Bearer headers using the first configured CLI/environment key, or
            the environment fallback when this config has no keys.
        """
        # First Party
        from lmcache.v1.multiprocess.transport.auth import (
            request_client_auth_headers,
        )

        if self._client_api_key is not None:
            return request_client_auth_headers(api_key=self._client_api_key)
        return request_client_auth_headers()

    @classmethod
    def from_key_files(
        cls, key_files: Sequence[str | Path]
    ) -> "CoordinatorAuthenticationConfig":
        """Load one API key from each command-line key file.

        Args:
            key_files: Paths containing one plaintext API key each.

        Returns:
            Authentication configuration containing the loaded keys.

        Raises:
            OSError: If a key file cannot be read.
            ValueError: If a file is empty or contains more than one line.
        """
        return cls(load_api_keys_from_files(key_files))

    @classmethod
    def from_sources(cls, key_files: Sequence[str | Path]) -> Self:
        """Load API keys from coordinator CLI files and environment variables.

        Args:
            key_files: Key files supplied through the command line.

        Returns:
            Authentication configuration containing every loaded key. An empty
            set leaves authentication disabled, preserving the no-key default.

        Raises:
            OSError: If a key file cannot be read.
            ValueError: If a key file is empty or contains more than one line.
        """
        return cls(load_service_api_keys(key_files))

    def accepts_authorization_header(self, authorization: str | None) -> bool:
        """Validate one HTTP Bearer authorization header.

        Args:
            authorization: Complete ``Authorization`` header value.

        Returns:
            ``True`` when the Bearer token matches a configured key.
        """
        return self.verify_token(bearer_token(authorization))


class CoordinatorAuthenticationMiddleware(HTTPBearerAuthenticationMiddleware):
    """Protect coordinator routes except its public liveness endpoint."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the coordinator authentication adapter.

        Args:
            app: Downstream ASGI application.
        """
        super().__init__(app, public_paths=PUBLIC_COORDINATOR_PATHS)

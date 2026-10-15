# SPDX-License-Identifier: Apache-2.0
"""HTTP authentication configuration for the multiprocess server."""

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

PUBLIC_HTTP_PATHS = frozenset({"/", "/healthcheck"})
HTTP_API_KEY_ENV_VAR = "LMCACHE_HTTP_API_KEY"
HTTP_API_KEY_FILE_ENV_VAR = "LMCACHE_HTTP_API_KEY_FILE"


class HTTPAuthenticationConfig(APIKeyVerifier):
    """API keys accepted by the multiprocess HTTP frontend."""

    @classmethod
    def from_key_files(
        cls, key_files: Sequence[str | Path]
    ) -> "HTTPAuthenticationConfig":
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
        """Load API keys from MP HTTP CLI files and environment variables.

        Args:
            key_files: Key files supplied through command-line options.

        Returns:
            Authentication configuration containing every loaded key.

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


class HTTPAuthenticationMiddleware(HTTPBearerAuthenticationMiddleware):
    """Protect MP HTTP routes except its public liveness endpoints."""

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the MP HTTP authentication adapter.

        Args:
            app: Downstream ASGI application.
        """
        super().__init__(app, public_paths=PUBLIC_HTTP_PATHS)

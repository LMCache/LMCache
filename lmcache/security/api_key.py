# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral API-key loading and verification."""

# Standard
from collections.abc import Sequence
from pathlib import Path
import argparse
import os
import secrets

# Third Party
from typing_extensions import Self

API_KEY_ENV_VAR = "LMCACHE_API_KEY"
API_KEY_FILE_ENV_VAR = "LMCACHE_API_KEY_FILE"
LEGACY_HTTP_API_KEY_ENV_VAR = "LMCACHE_HTTP_API_KEY"
LEGACY_HTTP_API_KEY_FILE_ENV_VAR = "LMCACHE_HTTP_API_KEY_FILE"


def add_api_key_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared authentication option on a service parser.

    Args:
        parser: Service command-line parser receiving the repeatable file option.
    """
    group = parser.add_argument_group("Authentication")
    group.add_argument(
        "--api-key-file",
        "--http-api-key-file",
        dest="api_key_file",
        action="append",
        default=[],
        metavar="PATH",
        help="Read an accepted service API key from PATH. Repeat during rotation. "
        "Environment alternatives are LMCACHE_API_KEY_FILE and LMCACHE_API_KEY. "
        "--http-api-key-file is a compatibility alias.",
    )


def load_service_api_keys(key_files: Sequence[str | Path]) -> tuple[str, ...]:
    """Load the shared service keys, with legacy HTTP environment fallbacks.

    Args:
        key_files: CLI files accepted by every authentication-enabled entry point.

    Returns:
        CLI keys plus the environment file and value. Each new environment
        variable takes precedence over its legacy HTTP equivalent.

    Raises:
        OSError: If a configured file cannot be read.
        ValueError: If a configured file is empty or contains multiple lines.
    """
    return load_api_keys(
        key_files,
        api_key_env_var=(
            API_KEY_ENV_VAR
            if os.environ.get(API_KEY_ENV_VAR)
            else LEGACY_HTTP_API_KEY_ENV_VAR
        ),
        api_key_file_env_var=(
            API_KEY_FILE_ENV_VAR
            if os.environ.get(API_KEY_FILE_ENV_VAR)
            else LEGACY_HTTP_API_KEY_FILE_ENV_VAR
        ),
    )


def load_api_keys_from_files(
    key_files: Sequence[str | Path],
) -> tuple[str, ...]:
    """Load one API key from each file.

    Args:
        key_files: Paths containing one plaintext API key each.

    Returns:
        API keys in the same order as ``key_files``.

    Raises:
        OSError: If a configured key file cannot be read.
        ValueError: If a key file is empty or contains more than one line.
    """
    api_keys: list[str] = []
    for key_file in key_files:
        path = Path(key_file)
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError(f"API key file is empty: {path}")
        if len(value.splitlines()) != 1:
            raise ValueError(f"API key file must contain exactly one line: {path}")
        api_keys.append(value)
    return tuple(api_keys)


def load_api_keys(
    key_files: Sequence[str | Path],
    *,
    api_key_env_var: str,
    api_key_file_env_var: str,
) -> tuple[str, ...]:
    """Load API keys from files and service-specific environment variables.

    File and environment sources are cumulative so a server can accept an old
    and a new key during rotation. Empty environment values are treated as
    unset.

    Args:
        key_files: Paths supplied by the service's command-line interface.
        api_key_env_var: Environment variable containing one API key value.
        api_key_file_env_var: Environment variable containing one key-file path.

    Returns:
        API keys in command-line file, environment file, environment value order.

    Raises:
        OSError: If a configured key file cannot be read.
        ValueError: If a key file is empty or contains more than one line.
    """
    combined_key_files = list(key_files)
    environment_key_file = os.environ.get(api_key_file_env_var)
    if environment_key_file:
        combined_key_files.append(environment_key_file)

    api_keys = list(load_api_keys_from_files(combined_key_files))
    environment_key = os.environ.get(api_key_env_var)
    if environment_key:
        api_keys.append(environment_key)
    return tuple(api_keys)


class APIKeyVerifier:
    """Verify caller-supplied tokens against accepted API keys.

    Keys remain private and the object representation exposes only whether
    verification is enabled and how many keys are accepted.

    Args:
        api_keys: Plaintext high-entropy API keys accepted by a service.

    Raises:
        ValueError: If a configured API key is empty.
    """

    def __init__(self, api_keys: Sequence[str] = ()) -> None:
        if any(not api_key for api_key in api_keys):
            raise ValueError("API keys must not be empty")
        self._api_keys = tuple(api_key.encode("utf-8") for api_key in api_keys)

    @classmethod
    def from_named_sources(
        cls,
        key_files: Sequence[str | Path],
        *,
        api_key_env_var: str,
        api_key_file_env_var: str,
    ) -> Self:
        """Build a verifier from files and environment variables.

        Args:
            key_files: Paths supplied by the service's command-line interface.
            api_key_env_var: Environment variable containing one API key value.
            api_key_file_env_var: Environment variable containing one key-file path.

        Returns:
            A verifier accepting every loaded API key.

        Raises:
            OSError: If a configured key file cannot be read.
            ValueError: If a key file is empty or contains more than one line.
        """
        return cls(
            load_api_keys(
                key_files,
                api_key_env_var=api_key_env_var,
                api_key_file_env_var=api_key_file_env_var,
            )
        )

    @property
    def enabled(self) -> bool:
        """Return whether at least one API key is configured."""
        return bool(self._api_keys)

    def verify_token(self, token: str | None) -> bool:
        """Verify one caller-supplied token in constant time per key.

        Args:
            token: Plaintext API key supplied by the caller, or ``None``.

        Returns:
            ``True`` when the token matches one configured key.
        """
        if token is None:
            return False
        token_bytes = token.encode("utf-8")
        matched = False
        for expected_api_key in self._api_keys:
            matched |= secrets.compare_digest(token_bytes, expected_api_key)
        return matched

    def __repr__(self) -> str:
        """Return a representation that never includes API keys."""
        return (
            f"{type(self).__name__}(enabled={self.enabled}, "
            f"key_count={len(self._api_keys)})"
        )

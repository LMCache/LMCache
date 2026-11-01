# SPDX-License-Identifier: Apache-2.0
"""Tests for transport-neutral API-key verification."""

# Third Party
import pytest

# First Party
from lmcache.security.api_key import APIKeyVerifier


@pytest.mark.parametrize("token", [None, "wrong", "wrong-\u00e9"])
def test_invalid_tokens_are_rejected(token: str | None) -> None:
    """Non-ASCII credentials must be rejected without a comparison error."""
    assert not APIKeyVerifier(["secret"]).verify_token(token)


def test_sources_are_cumulative_for_rotation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI files and environment sources all contribute accepted keys."""
    cli_file = tmp_path / "cli-key"
    cli_file.write_text("cli-secret\n", encoding="utf-8")
    environment_file = tmp_path / "environment-key"
    environment_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("TEST_API_KEY_FILE", str(environment_file))
    monkeypatch.setenv("TEST_API_KEY", "environment-secret")

    verifier = APIKeyVerifier.from_named_sources(
        [cli_file],
        api_key_env_var="TEST_API_KEY",
        api_key_file_env_var="TEST_API_KEY_FILE",
    )

    assert verifier.verify_token("cli-secret")
    assert verifier.verify_token("file-secret")
    assert verifier.verify_token("environment-secret")
    assert not verifier.verify_token("wrong")


def test_verifier_representation_hides_keys() -> None:
    """Diagnostic representations never reveal secret material."""
    verifier = APIKeyVerifier(["top-secret"])

    assert "top-secret" not in repr(verifier)
    assert "key_count=1" in repr(verifier)

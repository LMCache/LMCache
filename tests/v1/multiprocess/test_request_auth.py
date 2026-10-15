# SPDX-License-Identifier: Apache-2.0
"""Tests for request-transport authentication configuration."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.http_auth import HTTPAuthenticationConfig
from lmcache.v1.multiprocess.transport.auth import (
    load_request_authentication,
    load_request_client_api_key,
)


def test_http_and_request_transports_share_key_sources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both server entry points accept the same CLI, file, and inline keys."""
    key_file = tmp_path / "cli-key"
    key_file.write_text("cli-secret\n", encoding="utf-8")
    env_file = tmp_path / "env-key"
    env_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("LMCACHE_API_KEY", "inline-secret")
    monkeypatch.setenv("LMCACHE_API_KEY_FILE", str(env_file))
    for verifier in (
        HTTPAuthenticationConfig.from_sources([key_file]),
        load_request_authentication([key_file]),
    ):
        for token in ("cli-secret", "file-secret", "inline-secret"):
            assert verifier.verify_token(token)
        assert not verifier.verify_token("wrong")


def test_unified_key_takes_precedence_over_legacy_http_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy HTTP variables remain fallback aliases for the shared key source."""
    monkeypatch.setenv("LMCACHE_HTTP_API_KEY", "old-secret")
    monkeypatch.delenv("LMCACHE_API_KEY", raising=False)
    monkeypatch.delenv("LMCACHE_API_KEY_FILE", raising=False)
    monkeypatch.delenv("LMCACHE_HTTP_API_KEY_FILE", raising=False)
    assert load_request_client_api_key() == "old-secret"
    monkeypatch.setenv("LMCACHE_API_KEY", "new-secret")
    assert load_request_client_api_key() == "new-secret"
    verifier = HTTPAuthenticationConfig.from_sources([])
    assert verifier.verify_token("new-secret")
    assert not verifier.verify_token("old-secret")


def test_request_client_key_can_come_from_environment_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Request clients support secret-file injection without API changes."""
    key_file = tmp_path / "request-key"
    key_file.write_text("secret\n", encoding="utf-8")
    monkeypatch.setenv("LMCACHE_API_KEY_FILE", str(key_file))

    assert load_request_client_api_key() == "secret"


def test_request_client_rejects_multiple_environment_sources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One client request carries exactly one credential."""
    key_file = tmp_path / "request-key"
    key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("LMCACHE_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("LMCACHE_API_KEY", "inline-secret")

    with pytest.raises(ValueError, match="exactly one"):
        load_request_client_api_key()

# SPDX-License-Identifier: Apache-2.0
"""Tests for request-transport authentication configuration."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport.auth import load_request_client_api_key


def test_request_client_key_can_come_from_environment_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Request clients support secret-file injection without API changes."""
    key_file = tmp_path / "request-key"
    key_file.write_text("secret\n", encoding="utf-8")
    monkeypatch.setenv("LMCACHE_REQUEST_API_KEY_FILE", str(key_file))

    assert load_request_client_api_key() == "secret"


def test_request_client_rejects_multiple_environment_sources(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One client request carries exactly one credential."""
    key_file = tmp_path / "request-key"
    key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("LMCACHE_REQUEST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("LMCACHE_REQUEST_API_KEY", "inline-secret")

    with pytest.raises(ValueError, match="exactly one"):
        load_request_client_api_key()

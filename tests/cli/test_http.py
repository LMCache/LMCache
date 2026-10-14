# SPDX-License-Identifier: Apache-2.0
"""Tests for the CLI's LMCache HTTP server authentication."""

# Standard
from unittest.mock import MagicMock, patch
import urllib.request

# Third Party
import pytest

# First Party
from lmcache.cli.commands.quota._helpers import http_request
from lmcache.cli.http import fetch_json, kvcache_auth_headers


def _capture_urlopen() -> tuple[MagicMock, list[urllib.request.Request]]:
    """Build a fake ``urlopen`` that records each request and returns ``{}``."""
    requests: list[urllib.request.Request] = []
    response = MagicMock()
    response.read.return_value = b"{}"
    response.__enter__.return_value = response

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> MagicMock:
        requests.append(request)
        return response

    return MagicMock(side_effect=fake_urlopen), requests


def test_auth_headers_carry_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LMCACHE_HTTP_API_KEY", "s3cret")
    assert kvcache_auth_headers() == {"Authorization": "Bearer s3cret"}


def test_auth_headers_empty_without_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LMCACHE_HTTP_API_KEY", raising=False)
    assert kvcache_auth_headers() == {}


def test_fetch_json_sends_given_headers():
    fake_urlopen, requests = _capture_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        fetch_json(
            "http://localhost:8080/status",
            headers={"Authorization": "Bearer s3cret"},
        )
    assert requests[0].get_header("Authorization") == "Bearer s3cret"


def test_quota_request_sends_key(monkeypatch: pytest.MonkeyPatch):
    """``lmcache quota`` subcommands authenticate when the key is exported."""
    monkeypatch.setenv("LMCACHE_HTTP_API_KEY", "s3cret")
    fake_urlopen, requests = _capture_urlopen()
    with patch("urllib.request.urlopen", fake_urlopen):
        http_request("PUT", "http://localhost:8080/quota/alice", data={"x": 1})
    assert requests[0].get_header("Authorization") == "Bearer s3cret"
    assert requests[0].get_header("Content-type") == "application/json"

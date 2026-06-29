# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared CLI HTTP helpers (issue #3868)."""

# Standard
from io import BytesIO
import urllib.error
import urllib.request

# Third Party
import pytest

# First Party
from lmcache.cli.commands.describe import normalize_url as describe_normalize_url
from lmcache.cli.commands.ping import DEFAULT_URLS as ping_default_urls
from lmcache.cli.commands.quota._helpers import normalize_url as quota_normalize_url
from lmcache.cli.http import (
    DEFAULT_URLS,
    CliHttpError,
    normalize_url,
    request_json,
)


class _FakeResponse:
    """Minimal context-manager response with a ``read()`` body."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class TestNormalizeUrl:
    def test_adds_scheme(self) -> None:
        assert normalize_url("localhost:8080") == "http://localhost:8080"

    def test_keeps_scheme_and_strips_trailing_slashes(self) -> None:
        assert normalize_url("https://host:443///") == "https://host:443"


class TestDefaultUrls:
    def test_targets(self) -> None:
        assert DEFAULT_URLS["kvcache"] == "http://localhost:8080"
        assert DEFAULT_URLS["engine"] == "http://localhost:8000"


class TestRequestJson:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=None: _FakeResponse(b'{"ok": true}'),
        )
        assert request_json("GET", "http://x/status") == {"ok": True}

    def test_sets_json_body_and_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def _fake(req: urllib.request.Request, timeout: object = None) -> _FakeResponse:
            captured["data"] = req.data
            captured["ctype"] = req.get_header("Content-type")
            captured["method"] = req.get_method()
            return _FakeResponse(b"{}")

        monkeypatch.setattr(urllib.request, "urlopen", _fake)
        request_json("POST", "http://x", data={"a": 1})
        assert captured["data"] == b'{"a": 1}'
        assert captured["ctype"] == "application/json"
        assert captured["method"] == "POST"

    def test_http_error_maps_to_clihttperror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(req: object, timeout: object = None) -> _FakeResponse:
            raise urllib.error.HTTPError(
                "http://x",
                503,
                "Service Unavailable",
                {},  # type: ignore[arg-type]
                BytesIO(b'{"error": "down"}'),
            )

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        with pytest.raises(CliHttpError) as ei:
            request_json("GET", "http://x")
        exc = ei.value
        assert exc.status == 503
        assert exc.reason == "Service Unavailable"
        assert exc.body == '{"error": "down"}'

    def test_connection_error_maps_to_clihttperror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(req: object, timeout: object = None) -> _FakeResponse:
            raise urllib.error.URLError("refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        with pytest.raises(CliHttpError) as ei:
            request_json("GET", "http://x")
        exc = ei.value
        assert exc.status is None
        assert "refused" in exc.reason


class TestBackwardCompatReexports:
    """The shared helpers must still be importable from their old homes."""

    def test_describe_reexports_normalize_url(self) -> None:
        assert describe_normalize_url is normalize_url

    def test_quota_helpers_reexports_normalize_url(self) -> None:
        assert quota_normalize_url is normalize_url

    def test_ping_uses_shared_default_urls(self) -> None:
        assert ping_default_urls is DEFAULT_URLS

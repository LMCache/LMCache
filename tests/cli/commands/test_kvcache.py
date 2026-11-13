# SPDX-License-Identifier: Apache-2.0
"""Exercise KV cache commands through the public CLI and real loopback HTTP."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from urllib.parse import parse_qs, urlsplit
import json

# Third Party
import pytest

# First Party
from lmcache.cli.main import main

Reply = tuple[int, dict[str, object] | str]
Request = tuple[str, str, bytes]

NEXT_PAGE_TOKEN = "opaque+/=?& %游标"
PAGE: dict[str, object] = {
    "adapter": "fs",
    "shared": True,
    "entries": [
        {
            "key": {
                "chunk_hash_hex": "deadbeef",
                "model_name": "org/model",
                "kv_rank": 2,
                "object_group_id": 3,
                "cache_salt": "tenant-a",
                "tags": {"version": "v2"},
            },
            "size_bytes": 4096,
        }
    ],
    "next_page_token": NEXT_PAGE_TOKEN,
}
EMPTY_PAGE: dict[str, object] = {
    "adapter": "fs",
    "shared": False,
    "entries": [],
    "next_page_token": None,
}


@dataclass
class CacheServer:
    """HTTP boundary with queued replies and captured method, path, and body."""

    url: str
    replies: Queue[Reply]
    requests: Queue[Request]


@pytest.fixture
def cache_server() -> Iterator[CacheServer]:
    """Serve scripted API responses on an ephemeral loopback port.

    Yields:
        The server URL, reply queue, and captured requests.
    """
    replies: Queue[Reply] = Queue()
    requests: Queue[Request] = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Capture one request and send the next reply without interpreting it."""
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.put((self.command, self.path, body))
            try:
                status, payload = replies.get_nowait()
            except Empty:
                status, payload = 500, {"detail": "unexpected extra request"}
            encoded = (
                payload if isinstance(payload, str) else json.dumps(payload)
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_POST = do_GET

        def log_message(self, format: str, *args: object) -> None:
            """Suppress HTTP access logs in successful tests."""

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield CacheServer(
                f"http://127.0.0.1:{server.server_port}", replies, requests
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_list_one_page_and_resume(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """One invocation preserves the wire page and never follows its cursor."""
    cache_server.replies.put((200, PAGE))
    tokens = ["kvcache", "--format", "json", "list", "--url", cache_server.url + "/"]
    main(tokens)
    output = json.loads(capsys.readouterr().out)
    assert output == {"title": "L2 Cache Objects", "metrics": PAGE}
    method, target, body = cache_server.requests.get(timeout=5)
    assert (method, urlsplit(target).path, body) == ("GET", "/cache/objects", b"")
    assert parse_qs(urlsplit(target).query) == {"tier": ["l2"], "page_size": ["500"]}
    assert cache_server.requests.empty()

    cache_server.replies.put((200, EMPTY_PAGE))
    main([*tokens, "--page-token", output["metrics"]["next_page_token"]])
    assert json.loads(capsys.readouterr().out)["metrics"] == EMPTY_PAGE
    method, target, body = cache_server.requests.get(timeout=5)
    assert (method, urlsplit(target).path, body) == ("GET", "/cache/objects", b"")
    assert parse_qs(urlsplit(target).query) == {
        "tier": ["l2"],
        "page_size": ["500"],
        "page_token": [NEXT_PAGE_TOKEN],
    }
    assert cache_server.requests.empty()


@pytest.mark.parametrize("page_size", [1, 5000])
def test_list_filters_and_page_size(cache_server: CacheServer, page_size: int) -> None:
    """Filters survive URL encoding and both page-size boundaries reach the server."""
    cache_server.replies.put((200, EMPTY_PAGE))
    main(
        [
            "kvcache",
            "--quiet",
            "list",
            "--url",
            cache_server.url.removeprefix("http://"),
            "--adapter",
            "fs&secondary",
            "--model-name",
            "org/模型 + v1?#tag",
            "--page-size",
            str(page_size),
        ]
    )
    method, target, body = cache_server.requests.get(timeout=5)
    assert (method, urlsplit(target).path, body) == ("GET", "/cache/objects", b"")
    assert parse_qs(urlsplit(target).query) == {
        "tier": ["l2"],
        "page_size": [str(page_size)],
        "adapter": ["fs&secondary"],
        "model_name": ["org/模型 + v1?#tag"],
    }


def test_list_quiet_still_writes_output(
    cache_server: CacheServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Quiet suppresses stdout while retaining the complete JSON file report."""
    cache_server.replies.put((200, PAGE))
    output = tmp_path / "page.json"
    main(
        [
            "kvcache",
            "--format",
            "json",
            "--quiet",
            "--output",
            str(output),
            "list",
            "--url",
            cache_server.url,
        ]
    )
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text()) == {
        "title": "L2 Cache Objects",
        "metrics": PAGE,
    }


@pytest.mark.parametrize("page", [PAGE, EMPTY_PAGE], ids=["objects", "empty"])
def test_list_terminal_output(
    cache_server: CacheServer,
    page: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default report identifies the adapter, cursor, and objects or empty page."""
    cache_server.replies.put((200, page))
    main(["kvcache", "list", "--url", cache_server.url])
    text = capsys.readouterr().out
    assert "L2 Cache Objects" in text
    assert "Adapter:" in text and "fs" in text
    assert "Next page token:" in text
    if page["entries"]:
        assert "deadbeef" in text and "4096" in text
        assert NEXT_PAGE_TOKEN in text
    else:
        assert "(none)" in text


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((400, {"detail": "malformed page token"}), "HTTP 400"),
        ((404, {"detail": "no L2 adapter"}), "HTTP 404"),
        ((503, {"detail": "listing unsupported"}), "listing unsupported"),
        ((200, "not json"), "Invalid JSON response"),
    ],
    ids=["bad-cursor", "unknown-adapter", "unsupported-listing", "invalid-json"],
)
def test_list_request_errors(
    cache_server: CacheServer,
    reply: Reply,
    error: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP/JSON failures exit 1 with stderr diagnostics, even in quiet mode."""
    cache_server.replies.put(reply)
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "kvcache",
                "--quiet",
                "--output",
                str(output),
                "list",
                "--url",
                cache_server.url,
            ]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert error in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("page_size", ["0", "-1", "5001", "abc"])
def test_list_invalid_page_size(cache_server: CacheServer, page_size: str) -> None:
    """Invalid sizes fail parsing before contacting the cache service."""
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--url", cache_server.url, "--page-size", page_size])
    assert exc.value.code == 2
    assert cache_server.requests.empty()


def test_clear_preserves_l1_post(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adding list keeps the existing clear operation and report unchanged."""
    cache_server.replies.put((200, {"ok": True}))
    main(["kvcache", "--format", "json", "clear", "--url", cache_server.url])
    method, target, body = cache_server.requests.get(timeout=5)
    assert (method, target, json.loads(body)) == (
        "POST",
        "/cache/clear",
        {"tier": "l1"},
    )
    assert json.loads(capsys.readouterr().out) == {
        "title": "KV Cache Clear",
        "metrics": {"status": "OK"},
    }

# SPDX-License-Identifier: Apache-2.0
"""Exercise KV cache commands through the public CLI and real loopback HTTP."""

# Standard
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from urllib.parse import parse_qs, urlsplit
import json
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.cli.main import main
from tests.import_utils import lmcache_subprocess_env

ReplyPayload = Mapping[str, object] | str | bytes
Reply = tuple[int, ReplyPayload]
ReplyWithLength = tuple[int, ReplyPayload, int]
Request = tuple[str, str, bytes, str]

NEXT_PAGE_TOKEN = "opaque+/=?& %游标"
CHUNK_HASH = "0123456789abcdef"
UNSUPPORTED_LISTING = "L2 adapter 'disk' does not support listing: unavailable"
PAGE: dict[str, object] = {
    "adapter": "s3",
    "shared": True,
    "entries": [
        {
            "key": {
                "chunk_hash_hex": CHUNK_HASH,
                "model_name": "org/model",
                "kv_rank": 2,
                "object_group_id": 3,
                "cache_salt": "tenant-a",
                "tags": {"version": "v2", "env": "prod"},
            },
            "size_bytes": 4096,
        }
    ],
    "next_page_token": NEXT_PAGE_TOKEN,
}
FLAT_PAGE: dict[str, object] = {
    "adapter": "s3",
    "shared": True,
    "next_page_token": NEXT_PAGE_TOKEN,
    "entries": [
        {
            "chunk_hash_hex": CHUNK_HASH,
            "model_name": "org/model",
            "kv_rank": 2,
            "object_group_id": 3,
            "cache_salt": "tenant-a",
            "tags": {"version": "v2", "env": "prod"},
            "size_bytes": 4096,
        }
    ],
}
EMPTY_PAGE: dict[str, object] = {
    "adapter": "s3",
    "shared": False,
    "entries": [],
    "next_page_token": None,
}


@dataclass
class CacheServer:
    """HTTP boundary with queued replies and captured requests."""

    url: str
    replies: Queue[Reply | ReplyWithLength]
    requests: Queue[Request]


@pytest.fixture
def cache_server() -> Iterator[CacheServer]:
    """Serve scripted API responses on an ephemeral loopback port.

    Yields:
        The server URL, reply queue, and captured requests.
    """
    replies: Queue[Reply | ReplyWithLength] = Queue()
    requests: Queue[Request] = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Capture one request and send the next reply without interpreting it."""
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.put(
                (self.command, self.path, body, self.headers.get("Content-Type", ""))
            )
            try:
                reply = replies.get_nowait()
            except Empty:
                reply = (500, {"detail": "unexpected extra request"})
            if len(reply) == 3:
                status, payload, declared = reply
            else:
                status, payload = reply
                declared = None
            if isinstance(payload, bytes):
                encoded = payload
            else:
                encoded = (
                    payload if isinstance(payload, str) else json.dumps(dict(payload))
                ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if declared is not None and declared > len(encoded):
                self.close_connection = True
                self.send_header("Connection", "close")
            self.send_header(
                "Content-Length",
                str(declared if declared is not None else len(encoded)),
            )
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
    """One invocation flattens the wire page and never follows its cursor."""
    cache_server.replies.put((200, PAGE))
    tokens = ["kvcache", "list", "--format", "json", "--url", cache_server.url + "/"]
    main(tokens)
    output = json.loads(capsys.readouterr().out)
    assert output == {"title": "L2 Cache Objects", "metrics": FLAT_PAGE}
    method, target, body, _content_type = cache_server.requests.get(timeout=5)
    assert (method, urlsplit(target).path, body) == ("GET", "/cache/objects", b"")
    assert parse_qs(urlsplit(target).query) == {"tier": ["l2"], "page_size": ["500"]}
    assert cache_server.requests.empty()

    cache_server.replies.put((200, EMPTY_PAGE))
    main([*tokens, "--page-token", output["metrics"]["next_page_token"]])
    assert json.loads(capsys.readouterr().out)["metrics"] == EMPTY_PAGE
    method, target, body, _content_type = cache_server.requests.get(timeout=5)
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
            "list",
            "--quiet",
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
    method, target, body, _content_type = cache_server.requests.get(timeout=5)
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
    """Quiet saves the full hash and the original tag order in the JSON report."""
    cache_server.replies.put((200, PAGE))
    output = tmp_path / "page.json"
    main(
        [
            "kvcache",
            "list",
            "--format",
            "json",
            "--quiet",
            "--output",
            str(output),
            "--url",
            cache_server.url,
        ]
    )
    assert capsys.readouterr().out == ""
    expected = {"title": "L2 Cache Objects", "metrics": FLAT_PAGE}
    assert output.read_bytes() == (json.dumps(expected, indent=2) + "\n").encode()


@pytest.mark.parametrize("page", [PAGE, EMPTY_PAGE], ids=["objects", "empty"])
def test_list_terminal_output(
    cache_server: CacheServer,
    page: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default terminal report uses compact rows on stdout and in a file."""
    cache_server.replies.put((200, page))
    output = tmp_path / "page.txt"
    main(["kvcache", "list", "--url", cache_server.url, "--output", str(output)])
    text = capsys.readouterr().out
    assert output.read_text() == text
    assert "L2 Cache Objects" in text
    assert "Adapter:" in text and "s3" in text
    assert "Next page token:" in text
    if page["entries"]:
        for header in ("Chunk hash", "Model", "Rank", "Group", "Salt", "Tags", "Bytes"):
            assert header in text
        row = next(line for line in text.splitlines() if "org/model" in line)
        assert row.split() == [
            "0123456789ab…",
            "org/model",
            "2",
            "3",
            "tenant-a",
            "env=prod,version=v2",
            "4096",
        ]
        assert NEXT_PAGE_TOKEN in text
    else:
        assert "(none)" in text


@pytest.mark.parametrize("chunk_hash", ["deadbeef", "0123456789ab"])
def test_list_terminal_short_hash_and_empty_tags(
    cache_server: CacheServer, chunk_hash: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hashes up to 12 characters are printed in full and empty tags use a dash."""
    key = {
        "chunk_hash_hex": chunk_hash,
        "model_name": "org/model",
        "kv_rank": 0,
        "tags": {},
    }
    page = {**EMPTY_PAGE, "entries": [{"key": key, "size_bytes": 4096}]}
    cache_server.replies.put((200, page))
    main(["kvcache", "list", "--format", "terminal", "--url", cache_server.url])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [chunk_hash, "org/model", "0", "0", "-", "4096"] in rows


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((400, {"detail": "malformed page token"}), "malformed page token"),
        ((404, {"detail": "no L2 adapter"}), "no L2 adapter"),
        ((503, {"detail": UNSUPPORTED_LISTING}), UNSUPPORTED_LISTING),
        ((503, {"error": "engine not initialized"}), "engine not initialized"),
        ((400, {"message": "disk full"}), "disk full"),
        ((200, "not json"), "Invalid JSON response"),
    ],
    ids=[
        "bad-cursor",
        "unknown-adapter",
        "unsupported-listing",
        "error-key",
        "message-key",
        "invalid-json",
    ],
)
def test_list_request_errors(
    cache_server: CacheServer,
    reply: Reply,
    error: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP errors report the status, URL, and reason, including unsupported listing."""
    cache_server.replies.put(reply)
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "kvcache",
                "list",
                "--quiet",
                "--output",
                str(output),
                "--url",
                cache_server.url,
            ]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    if reply[0] != 200:
        _, target, _, _ = cache_server.requests.get_nowait()
        error = f"HTTP {reply[0]} from {cache_server.url}{target}: {error}"
    assert error in captured.err
    assert "Server unhealthy" not in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("page_size", ["0", "-1", "5001", "abc"])
def test_list_invalid_page_size(cache_server: CacheServer, page_size: str) -> None:
    """Invalid sizes fail parsing before contacting the cache service."""
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--url", cache_server.url, "--page-size", page_size])
    assert exc.value.code == 2
    assert cache_server.requests.empty()


def test_list_requires_output_flags_after_the_subcommand(
    cache_server: CacheServer,
) -> None:
    """Only clear has a legacy group-level syntax."""
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "--format", "json", "list", "--url", cache_server.url])
    assert exc.value.code == 2
    assert cache_server.requests.empty()


@pytest.mark.parametrize(
    ("group_options", "leaf_options"),
    [
        (["-q", "--format", "json", "--output", "clear"], []),
        (["--format=terminal", "--output", "clear"], ["--quiet", "--format", "json"]),
    ],
)
def test_legacy_clear_output_options(
    cache_server: CacheServer,
    tmp_path: Path,
    group_options: list[str],
    leaf_options: list[str],
) -> None:
    """Older scripts reach the same POST and preserve their output settings."""
    cache_server.replies.put((200, {"ok": True}))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "kvcache",
            *group_options,
            "clear",
            *leaf_options,
            "--url",
            cache_server.url,
        ],
        cwd=tmp_path,
        env=lmcache_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert not proc.stdout
    assert json.loads((tmp_path / "clear").read_text())["metrics"] == {"status": "OK"}
    method, target, body, _ = cache_server.requests.get_nowait()
    assert (method, target, json.loads(body)) == (
        "POST",
        "/cache/clear",
        {"tier": "l1"},
    )


LEGACY_PAGE: dict[str, object] = {
    "adapter": "s3",
    "shared": False,
    "entries": [
        {
            "key": {
                "chunk_hash_hex": "c0ffee",
                "model_name": "org/model",
                "kv_rank": 1,
            },
            "size_bytes": 2048,
        }
    ],
    "next_page_token": None,
}


def test_list_legacy_keys_get_protocol_defaults(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fields omitted by older servers flatten to typed defaults, not null."""
    cache_server.replies.put((200, LEGACY_PAGE))
    main(["kvcache", "list", "--format", "json", "--url", cache_server.url])
    output = json.loads(capsys.readouterr().out)
    assert output["metrics"]["entries"] == [
        {
            "chunk_hash_hex": "c0ffee",
            "model_name": "org/model",
            "kv_rank": 1,
            "object_group_id": 0,
            "cache_salt": "",
            "tags": {},
            "size_bytes": 2048,
        }
    ]


def test_clear_preserves_l1_post(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adding list keeps the existing clear operation and report unchanged."""
    cache_server.replies.put((200, {"ok": True}))
    main(["kvcache", "clear", "--format", "json", "--url", cache_server.url])
    method, target, body, content_type = cache_server.requests.get(timeout=5)
    assert content_type == "application/json"
    assert (method, target, json.loads(body)) == (
        "POST",
        "/cache/clear",
        {"tier": "l1"},
    )
    assert json.loads(capsys.readouterr().out) == {
        "title": "KV Cache Clear",
        "metrics": {"status": "OK"},
    }


@pytest.mark.parametrize(
    "command",
    [["kvcache", "clear"], ["describe", "kvcache"]],
    ids=["clear", "describe"],
)
def test_clear_and_describe_report_server_health_on_503(
    cache_server: CacheServer, command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Both commands include the health diagnostic and the server's reason."""
    cache_server.replies.put((503, {"detail": "server not initialized"}))
    with pytest.raises(SystemExit) as exc:
        main([*command, "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    assert "Server unhealthy: server not initialized" in capsys.readouterr().err


def test_list_json_detail_beyond_whitespace_bound(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parseable detail survives legal padding past the 4 KiB bound."""
    body = b'{"detail": "full-detail"}'.ljust(4097, b" ")
    cache_server.replies.put((503, body))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert ": full-detail" in captured.err
    assert "[truncated]" not in captured.err


def test_list_short_body_reports_remaining_bytes(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A body cut short of its Content-Length reports the remaining count."""
    cache_server.replies.put((400, b'{"ok":', 100))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Incomplete HTTP response" in captured.err
    assert "94 more expected" in captured.err


@pytest.mark.parametrize("status", [400, 503])
def test_list_chunked_body_cut_reports_incomplete(
    status: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chunked error body cut mid-transfer is an incomplete response."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b'5\r\n{"ok"\r\n')
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            """Suppress HTTP access logs in successful tests."""

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            with pytest.raises(SystemExit) as exc:
                main(
                    [
                        "kvcache",
                        "list",
                        "--quiet",
                        "--url",
                        f"http://127.0.0.1:{server.server_port}",
                    ]
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Incomplete HTTP response" in captured.err
    assert "Traceback" not in captured.err


def test_list_not_modified_with_content_length_is_not_incomplete(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 304 describes its representation's size; it carries no body."""
    cache_server.replies.put((304, b"", 80))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "304" in captured.err
    assert "Incomplete" not in captured.err


def test_list_multibyte_char_across_body_bound(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A multi-byte character split by the 4 KiB bound keeps the JSON detail."""
    body = b'{"detail": "kaboom"}'.ljust(4095, b" ") + "雪".encode()
    cache_server.replies.put((400, body))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "kaboom" in captured.err
    assert "Traceback" not in captured.err


def test_list_oversized_multibyte_error_body_decodes_prefix(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A large Chinese error body yields a readable truncated prefix."""
    body = json.dumps({"detail": "诊断" * 800}, ensure_ascii=False).encode()
    cache_server.replies.put((503, body))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert f"HTTP 503 from {cache_server.url}/cache/objects?" in captured.err
    assert "诊断" in captured.err
    assert "[truncated]" in captured.err
    assert "Traceback" not in captured.err


def test_list_invalid_utf8_in_error_body(
    cache_server: CacheServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """Undecodable bytes in an error body stay a read error, not a detail."""
    cache_server.replies.put((400, b'{"detail": "a\xffb"}'))
    with pytest.raises(SystemExit) as exc:
        main(["kvcache", "list", "--quiet", "--url", cache_server.url])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Cannot read HTTP 400 response" in captured.err
    assert "Traceback" not in captured.err

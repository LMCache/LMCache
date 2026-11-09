# SPDX-License-Identifier: Apache-2.0
"""Exercise quota addressing through real HTTP requests without a GPU server."""

# Standard
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Thread
from urllib.parse import unquote, urlsplit
import argparse
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.query.coordinator_command import CoordinatorQueryCommand
from lmcache.cli.commands.quota import QuotaCommand


@pytest.fixture(scope="module")
def quota_server() -> Iterator[tuple[str, Queue[tuple[str, str]]]]:
    """Serve quota responses and capture method/path pairs at the HTTP boundary.

    Yields:
        The loopback server URL and a queue of received requests.
    """
    requests: Queue[tuple[str, str]] = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Echo the decoded salt using the quota API's anonymous sentinel."""
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.put((self.command, self.path))
            salt = unquote(urlsplit(self.path).path.removeprefix("/quota/"))
            body = json.dumps(
                {"cache_salt": "" if salt == "_default" else salt, "limit_gb": 1}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_PUT = do_GET
        do_DELETE = do_GET

        def log_message(self, format: str, *args: object) -> None:
            """Suppress access logs in successful tests."""

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("command", ["get", "set", "delete", "coordinator"])
@pytest.mark.parametrize(
    ("salt", "path_salt"),
    [
        ("tenant1", "tenant1"),
        ("", "_default"),
        ("_default", "_default"),
        ("tenant#blue", "tenant%23blue"),
        ("tenant?blue", "tenant%3Fblue"),
        ("tenant%20blue", "tenant%2520blue"),
        ("tenant blue", "tenant%20blue"),
        ("租户A", "%E7%A7%9F%E6%88%B7A"),
    ],
)
def test_quota_addresses_exact_salt(
    command: str,
    salt: str,
    path_salt: str,
    quota_server: tuple[str, Queue[tuple[str, str]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Commands preserve tenant identity on the wire and in displayed responses."""
    url, requests = quota_server
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    if command == "coordinator":
        CoordinatorQueryCommand().register(subparsers)
        tokens = ["coordinator", "--api", "quota", "--cache-salt", salt]
    else:
        QuotaCommand().register(subparsers)
        tokens = ["quota", command, salt]
        if command == "set":
            tokens += ["--limit-gb", "1"]
    args = parser.parse_args([*tokens, "--url", url, "--format", "json"])
    args.func(args)

    method = {"set": "PUT", "delete": "DELETE"}.get(command, "GET")
    assert requests.get(timeout=5) == (method, f"/quota/{path_salt}")
    output = capsys.readouterr().out
    assert json.loads(output)["metrics"]["cache_salt"] == (
        "" if salt == "_default" else salt
    )


def test_coordinator_without_salt_lists_quotas(
    quota_server: tuple[str, Queue[tuple[str, str]]],
) -> None:
    """Omitting the salt keeps the list endpoint distinct from anonymous traffic."""
    url, requests = quota_server
    parser = argparse.ArgumentParser()
    CoordinatorQueryCommand().register(parser.add_subparsers())
    args = parser.parse_args(["coordinator", "--api", "quota", "--url", url])
    args.func(args)
    assert requests.get(timeout=5) == ("GET", "/quota")

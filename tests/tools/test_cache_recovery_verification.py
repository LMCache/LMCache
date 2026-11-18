# SPDX-License-Identifier: Apache-2.0
"""Prevent successful inference alone from passing cache recovery checks."""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import subprocess
import threading

# Third Party
import pytest

# First Party
from tests.import_utils import repo_path

HELPER = repo_path(
    ".buildkite",
    "k3_tests",
    "multiprocess",
    "scripts",
    "workloads",
    "verification",
    "l1-round-trip-verification.sh",
)


@pytest.mark.parametrize(
    "behavior", ["retrieve", "no_read", "read_without_load", "empty_output"]
)
def test_cache_service_requires_actual_retrieve(tmp_path: Path, behavior: str) -> None:
    """Require completed writes, actual reads, and usable matching responses."""
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Serve the counters measured by the real shell assertions."""
            reads = int(requests > 1 and behavior in {"retrieve", "read_without_load"})
            loaded = int(requests > 1 and behavior == "retrieve")
            body = (
                f"lmcache_mp_l1_write_chunks_total {int(requests > 0)}\n"
                f"lmcache_mp_l1_read_chunks_total {reads}\n"
                f"lmcache_mp_num_chunks_loaded_total {loaded}\n"
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            """Return successful requests even when no KV was retrieved."""
            nonlocal requests
            self.rfile.read(int(self.headers["Content-Length"]))
            requests += 1
            text = "" if behavior == "empty_output" else "a generated answer"
            body = json.dumps(
                {"choices": [{"text": text}], "usage": {"completion_tokens": 4}}
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep HTTP fixture logging out of test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("An input prompt")
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; LMCACHE_HTTP_PORT="$2"; ENGINE_PORT="$2"; '
                'PROMPT_FILE="$3/prompt.txt"; OUT_A="$3/cold.txt"; '
                'OUT_B="$3/hit.txt"; MODEL=test; MAX_TOKENS=4; '
                "engine_clear_local_cache() { :; }; "
                "l1_round_trip_verify_cache_service recovery",
                "bash",
                str(HELPER),
                str(server.server_port),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert (result.returncode == 0) == (behavior == "retrieve"), (
        result.stdout + result.stderr
    )

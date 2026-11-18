# SPDX-License-Identifier: Apache-2.0
"""Tests for the vLLM nightly index age check used by the Docker builds.

A local HTTP server stands in for wheels.vllm.ai, so no external network is
needed. Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_check_vllm_nightly_age.py
"""

# Standard
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
import importlib.util
import re
import threading

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / ".github/scripts/check_vllm_nightly_age.py"

# Path prefix -> age in days of the index page; None sends no Last-Modified.
_INDEXES: dict[str, int | None] = {"fresh": 1, "stale": 30, "no-date": None}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_lmcache_vllm_age", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the age check from {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


age_check = _load_module()


class _IndexHandler(BaseHTTPRequestHandler):
    """Serve ``/<name>/vllm/`` for each entry of ``_INDEXES``; 404 otherwise."""

    def do_HEAD(self) -> None:  # noqa: N802 (http.server naming)
        name = self.path.strip("/").removesuffix("/vllm")
        if name not in _INDEXES or not self.path.endswith("/vllm/"):
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        age_days = _INDEXES[name]
        if age_days is not None:
            modified = datetime.now(timezone.utc) - timedelta(days=age_days)
            self.send_header("Last-Modified", format_datetime(modified, usegmt=True))
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def index_base() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_fresh_index_passes(index_base: str) -> None:
    assert age_check.main([f"{index_base}/fresh", "7"]) == 0


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("stale", "30 days old"),
        ("missing", "not published"),
        ("no-date", "no Last-Modified"),
    ],
)
def test_unusable_index_fails_with_guidance(
    index_base: str, name: str, reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert age_check.main([f"{index_base}/{name}", "7"]) == 1
    err = capsys.readouterr().err
    assert reason in err
    assert "VLLM_NIGHTLY_MAX_AGE_DAYS=0" in err


def test_zero_limit_skips_the_request() -> None:
    # Port 1 refuses connections: any request would raise instead of passing.
    assert age_check.main(["http://127.0.0.1:1/never", "0"]) == 0


@pytest.mark.parametrize("max_age", ["", "-1", "seven"])
def test_invalid_limit_is_a_usage_error(max_age: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        age_check.main(["http://127.0.0.1:1/never", max_age])
    assert excinfo.value.code == 2


def test_dockerfiles_pin_the_same_vllm() -> None:
    pins = {}
    for name in ("Dockerfile", "Dockerfile.standalone"):
        text = (ROOT / "docker" / name).read_text()
        pins[name] = re.findall(r"^ARG VLLM_VERSION=(\S+)$", text, re.MULTILINE)
    assert pins["Dockerfile"] == pins["Dockerfile.standalone"]
    assert len(pins["Dockerfile"]) == 1

# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache describe kvcache``."""

# Standard
from argparse import Namespace
from collections.abc import Mapping
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from unittest.mock import patch
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.describe import (
    DescribeCommand,
    DescribeError,
    fetch_health,
    fetch_json,
    fetch_metric_totals,
    fetch_running_requests,
    fmt_health,
    normalize_url,
    safe_get,
)

# ---------------------------------------------------------------------------
# Sample /status payload
# ---------------------------------------------------------------------------


def _l1_status(used_bytes: int, total_bytes: int, objects: int) -> dict:
    """One L1 entry as ``L1Manager.report_status`` reports it."""
    return {
        "is_healthy": True,
        "total_object_count": objects,
        "write_locked_count": 0,
        "read_locked_count": 0,
        "temporary_count": 0,
        "staging_object_count": 0,
        "staging_bytes": 0,
        "memory_used_bytes": used_bytes,
        "memory_total_bytes": total_bytes,
        "memory_configured_bytes": total_bytes,
        "backend": "dram",
        "memory_usage_ratio": used_bytes / total_bytes if total_bytes > 0 else 0.0,
        "write_ttl_seconds": 10,
        "read_ttl_seconds": 10,
    }


def _eviction_status(policy: str) -> dict:
    """One L1 eviction controller entry as the server reports it."""
    return {
        "is_healthy": True,
        "thread_alive": True,
        "eviction_policy": policy,
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.2,
    }


_SINGLE_L1 = _l1_status(45_415_895_859, 64_424_509_440, 1024)  # ~42.30 / 60 GB
SAMPLE_STATUS = {
    "is_healthy": True,
    "engine_type": "MPCacheServer",
    "chunk_size": 256,
    "hash_algorithm": "sha256",
    "registered_gpu_ids": [0],
    "cache_context_meta": {
        "0": {
            "model_name": "llama",
            "world_size": 1,
            "kv_cache_layout": {
                "num_layers": 32,
                "num_blocks": 2048,
                "cache_size_per_token": 163840,
                "kernel_groups": [
                    {
                        "kernel_group_idx": 0,
                        "engine_group_idx": 0,
                        "object_group_idx": 0,
                        "num_layers": 32,
                        "layer_indices": list(range(32)),
                        "tokens_per_block": 16,
                        "slots_per_block": 16,
                        "dtype": "torch.float16",
                        "engine_kv_concrete_shape": "32 x [2, 2048, 16, 8, 128]",
                        "is_mla": False,
                        "engine_kv_format": "NL_X_TWO_NB_BS_NH_HS",
                        "engine_kv_shape": "NL x [2, NB, BS, NH, HS]",
                        "attention_backend": "vLLM non-MLA flash attention",
                    },
                ],
            },
        },
    },
    "active_sessions": 3,
    # A single-L1 server sends the per-tag maps and also copies its only
    # L1 into the single-L1 keys.
    "storage_manager": {
        "is_healthy": True,
        "l1_managers": {"_default": _SINGLE_L1},
        "store_controllers": {"_default": {"is_healthy": True}},
        "l1_eviction_controllers": {"_default": _eviction_status("LRU")},
        "prefetch_controller": {"is_healthy": True},
        "l2_eviction_controller": {"is_healthy": True, "adapters": []},
        "l2_adapters": [],
        "num_l2_adapters": 0,
        "l1_usage": [45_415_895_859, 64_424_509_440],
        "l1_capacity_bytes_by_backend": {"dram": 64_424_509_440},
        "l1_manager": _SINGLE_L1,
        "store_controller": {"is_healthy": True},
        "l1_eviction_controller": _eviction_status("LRU"),
    },
}

# A multi-L1 server sends only the per-tag maps; each L1 may set its own policy.
# A shadow L1 ("peer") indexes a peer's bytes: it reports zero local memory and
# the default eviction policy, but its objects are readable here.
MULTI_L1_STATUS = {
    **SAMPLE_STATUS,
    "storage_manager": {
        "is_healthy": True,
        "l1_managers": {
            "hot": _l1_status(536_870_912, 1_073_741_824, 100),  # 0.5 / 1 GB
            "warm": _l1_status(0, 2_147_483_648, 0),  # 0 / 2 GB
            "peer": _l1_status(0, 0, 5),
        },
        "store_controllers": {
            "hot": {"is_healthy": True},
            "warm": {"is_healthy": True},
            "peer": {"is_healthy": True},
        },
        "l1_eviction_controllers": {
            "hot": _eviction_status("LRU"),
            "warm": _eviction_status("ARC"),
            "peer": _eviction_status("LRU"),
        },
        "prefetch_controller": {"is_healthy": True},
        "l2_eviction_controller": {"is_healthy": True, "adapters": []},
        "l2_adapters": [],
        "num_l2_adapters": 0,
        "l1_usage": [536_870_912, 3_221_225_472],
        "l1_capacity_bytes_by_backend": {"dram": 3_221_225_472},
    },
}


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_adds_http(self):
        assert normalize_url("localhost:8000") == "http://localhost:8000"

    def test_preserves_http(self):
        assert normalize_url("http://localhost:8000") == "http://localhost:8000"

    @pytest.mark.parametrize(
        "scheme", ["http", "https", "HTTP", "HTTPS", "HtTp", "hTtPs"]
    )
    def test_preserves_scheme_case(self, scheme: str) -> None:
        """Recognize HTTP(S) schemes without changing case-sensitive URL paths."""
        url = f"{scheme}://Example.com:443/CaseSensitive"
        assert normalize_url(url) == url

    def test_strips_trailing_slash(self):
        assert normalize_url("http://host:8000/") == "http://host:8000"

    def test_strips_multiple_trailing_slashes(self):
        assert normalize_url("http://host:8000///") == "http://host:8000"


class TestFmtHealth:
    def test_healthy(self):
        assert fmt_health(True) == "OK"

    def test_unhealthy(self):
        assert fmt_health(False) == "UNHEALTHY"

    def test_none(self):
        assert fmt_health(None) is None


class TestSafeGet:
    def test_nested(self):
        d = {"a": {"b": {"c": 42}}}
        assert safe_get(d, "a", "b", "c") == 42

    def test_missing_key(self):
        d = {"a": {"b": 1}}
        assert safe_get(d, "a", "x") is None

    def test_missing_key_with_default(self):
        d = {"a": 1}
        assert safe_get(d, "a", "b", default="N/A") == "N/A"

    def test_non_dict_intermediate(self):
        d = {"a": 5}
        assert safe_get(d, "a", "b") is None


# ---------------------------------------------------------------------------
# Field extraction integration test
# ---------------------------------------------------------------------------


class TestDescribeKvcacheFields:
    """Test that ``_describe_kvcache`` extracts fields correctly from a
    sample ``/status`` response."""

    def test_field_extraction(self):
        """Verify metrics are populated from the sample status dict."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        # Patch fetch_json to return our sample data
        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=SAMPLE_STATUS,
        ):
            # Capture the JSON output
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        m = output["metrics"]
        assert m["health"] == "OK"
        assert m["url"] == "http://localhost:8000"
        assert m["engine_type"] == "MPCacheServer"
        assert m["chunk_size"] == 256
        assert m["l1_capacity_gb"] == 60.0
        assert m["l1_used_gb"] == "42.30 (70.5%)"
        assert m["eviction_policy"] == "LRU"
        assert m["cached_objects"] == 1024
        assert m["active_sessions"] == 3

        # Per-model section (list)
        assert "models" in m
        model = m["models"][0]
        assert model["model"] == "llama"
        assert model["world_size"] == 1
        assert model["gpu_ids"] == "0"
        assert model["num_layers"] == 32
        assert model["num_blocks"] == 2048
        assert model["cache_size_per_token"] == 163840

        # Per-kernel-group section (list)
        assert "kernel_groups" in m
        kg = m["kernel_groups"][0]
        assert kg["model"] == "llama"
        assert kg["kernel_group_idx"] == 0
        assert kg["engine_group_idx"] == 0
        assert kg["object_group_idx"] == 0
        assert kg["num_layers"] == 32
        assert kg["slots_per_block"] == 16
        assert kg["dtype"] == "torch.float16"
        assert kg["is_mla"] is False
        assert kg["attention_backend"] == "vLLM non-MLA flash attention"
        assert kg["engine_kv_shape"] == "NL x [2, NB, BS, NH, HS]"
        assert kg["engine_kv_concrete_shape"] == "32 x [2, 2048, 16, 8, 128]"

        # A single L1 is fully described by the totals above.
        assert "l1_managers" not in m

    def test_multi_l1_totals_and_per_l1_sections(self):
        """A multi-L1 server reports totals and one section per L1.

        Such a server omits the single-L1 ``l1_manager`` keys, so every
        value must come from the per-tag maps. A zero-capacity shadow L1
        adds its objects to the total but nothing to capacity or usage.
        """
        # Standard
        import io

        cmd = DescribeCommand()
        args = Namespace(
            target="kvcache", url="http://localhost:8000", format="json", output=None
        )
        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=MULTI_L1_STATUS,
        ):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(args)

        m = json.loads(buf.getvalue())["metrics"]
        assert m["l1_capacity_gb"] == 3.0
        assert m["l1_used_gb"] == "0.50 (16.7%)"
        assert m["eviction_policy"] == "LRU, ARC"
        assert m["cached_objects"] == 105
        assert m["l1_managers"] == [
            {
                "tag": "hot",
                "backend": "dram",
                "capacity_gb": 1.0,
                "used_gb": "0.50 (50.0%)",
                "cached_objects": 100,
                "eviction_policy": "LRU",
            },
            {
                "tag": "warm",
                "backend": "dram",
                "capacity_gb": 2.0,
                "used_gb": "0.00 (0.0%)",
                "cached_objects": 0,
                "eviction_policy": "ARC",
            },
            {
                "tag": "peer",
                "backend": "dram",
                "capacity_gb": 0.0,
                "used_gb": "0.00 (0.0%)",
                "cached_objects": 5,
                "eviction_policy": "LRU",
            },
        ]

    def test_unhealthy(self):
        """Verify health shows UNHEALTHY when is_healthy is False."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        unhealthy_data = {**SAMPLE_STATUS, "is_healthy": False}
        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=unhealthy_data,
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        assert output["metrics"]["health"] == "UNHEALTHY"

    def test_missing_fields_show_na(self):
        """Verify missing nested fields render as None (N/A in terminal)."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        minimal_data = {"is_healthy": True, "engine_type": "MPCacheServer"}
        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            return_value=minimal_data,
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

            output = json.loads(buf.getvalue())

        m = output["metrics"]
        assert m["l1_capacity_gb"] is None
        assert m["l1_used_gb"] is None
        assert m["eviction_policy"] is None
        assert m["cached_objects"] is None
        assert m["active_sessions"] is None


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestDescribeErrors:
    def test_connection_refused(self):
        """Verify sys.exit(1) on connection error."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:19999"
            format = None
            output = None

        with pytest.raises(SystemExit) as exc_info:
            cmd.execute(FakeArgs())

        assert exc_info.value.code == 1

    def test_503_error(self):
        """Verify sys.exit(1) on HTTP 503."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "kvcache"
            url = "http://localhost:8000"
            format = None
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            side_effect=DescribeError("Server unhealthy: engine not initialized"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())

            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# fetch_json with a real HTTP server
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves a canned JSON response."""

    response_body: bytes = b"{}"
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format, *args):
        pass  # suppress stderr noise


class TestFetchJson:
    def test_success(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {
                "response_body": json.dumps({"ok": True}).encode(),
                "response_code": 200,
            },
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            result = fetch_json(f"http://127.0.0.1:{port}/status")
            assert result == {"ok": True}
        finally:
            server.server_close()

    def test_503(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {
                "response_body": json.dumps(
                    {"error": "engine not initialized"}
                ).encode(),
                "response_code": 503,
            },
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            with pytest.raises(DescribeError, match="Server unhealthy"):
                fetch_json(f"http://127.0.0.1:{port}/status")
        finally:
            server.server_close()

    @pytest.mark.parametrize("status", [200, 503])
    @pytest.mark.parametrize("target", [None, "kvcache", "engine"])
    def test_truncated_response(
        self, status: int, target: str | None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class Handler(_MockHandler):
            def do_GET(self) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b'{"ok":')

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                if target is None:
                    with pytest.raises(
                        DescribeError, match="Incomplete HTTP response"
                    ) as exc:
                        fetch_json(f"{url}/status")
                    assert isinstance(exc.value.__cause__.__cause__, IncompleteRead)
                else:
                    with pytest.raises(SystemExit) as exit_info:
                        DescribeCommand().execute(Namespace(target=target, url=url))
                    assert exit_info.value.code == 1
                    stderr = capsys.readouterr().err
                    assert "Incomplete HTTP response" in stderr
                    assert url in stderr
                    assert "Traceback" not in stderr
            finally:
                thread.join(timeout=5)

    def test_503_body_timeout(self) -> None:
        """An unhealthy server can stall after sending its response headers."""
        release = Event()

        class Handler(_MockHandler):
            def do_GET(self) -> None:
                self.send_response(503)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.flush()
                release.wait(timeout=5)

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/status"
                with pytest.raises(DescribeError) as exc:
                    fetch_json(url, timeout=1)
                assert url in str(exc.value)
                assert isinstance(exc.value.__cause__.__cause__, TimeoutError)
            finally:
                release.set()
                thread.join(timeout=5)

    @pytest.mark.parametrize("target", [None, "kvcache", "engine"])
    def test_503_invalid_utf8(
        self, target: str | None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An undecodable error body must reach the normal CLI error handler."""

        class Handler(_MockHandler):
            response_code = 503
            response_body = b"\xff"

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                if target is None:
                    with pytest.raises(DescribeError) as exc:
                        fetch_json(f"{url}/status")
                    assert url in str(exc.value)
                    assert isinstance(exc.value.__cause__.__cause__, UnicodeDecodeError)
                else:
                    with pytest.raises(SystemExit) as exc_info:
                        DescribeCommand().execute(Namespace(target=target, url=url))
                    assert exc_info.value.code == 1
                    stderr = capsys.readouterr().err
                    assert url in stderr
                    assert "Traceback" not in stderr
            finally:
                thread.join(timeout=5)

    def test_invalid_json(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {
                "response_body": b"",
                "response_code": 200,
            },
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            with pytest.raises(DescribeError, match="Invalid JSON response"):
                fetch_json(f"http://127.0.0.1:{port}/status")
        finally:
            server.server_close()


class TestFetchHealth:
    def test_ok(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": b"", "response_code": 200},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            assert fetch_health(f"http://127.0.0.1:{port}/health") is True
        finally:
            server.server_close()

    def test_non_200(self):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": b"", "response_code": 503},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            assert fetch_health(f"http://127.0.0.1:{port}/health") is False
        finally:
            server.server_close()

    def test_connection_refused(self):
        assert fetch_health("http://127.0.0.1:19999/health") is False


class TestFetchRunningRequests:
    def _serve(self, body: bytes, code: int = 200):
        handler = type(
            "_H",
            (_MockHandler,),
            {"response_body": body, "response_code": code},
        )
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        return server, port

    def test_single_series(self):
        body = (
            b"# HELP vllm:num_requests_running Number of running requests.\n"
            b"# TYPE vllm:num_requests_running gauge\n"
            b'vllm:num_requests_running{model_name="m"} 3.0\n'
        )
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") == 3
        finally:
            server.server_close()

    def test_sums_multiple_series(self):
        body = (
            b'vllm:num_requests_running{model_name="a"} 3.0\n'
            b'vllm:num_requests_running{model_name="b"} 1.0\n'
        )
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") == 4
        finally:
            server.server_close()

    def test_metric_absent(self):
        body = b"# only other metrics here\nvllm:num_requests_waiting 5.0\n"
        server, port = self._serve(body)
        try:
            assert fetch_running_requests(f"http://127.0.0.1:{port}/metrics") is None
        finally:
            server.server_close()

    def test_connection_refused(self):
        assert fetch_running_requests("http://127.0.0.1:19999/metrics") is None


# ---------------------------------------------------------------------------
# describe kvcache: cache hits from /metrics
# ---------------------------------------------------------------------------

# Two series per counter (two cache salts), grouped per family as the MP
# server's OpenTelemetry Prometheus exporter writes them. The trailing
# ``_created`` block is what native ``prometheus_client`` counters add.
_HIT_SERIES = {
    "requested": (600, 424),
    "hit": (500, 268),
    "hit_l1": (400, 112),
    "hit_l2": (100, 156),
    "prompt": (610, 420),
    "request_hit": (520, 280),
}


def _lookup_metrics_text(series: Mapping[str, tuple[float, float]]) -> bytes:
    """Render lookup counters in Prometheus text format, one series per salt."""
    lines = []
    for short, values in series.items():
        family = f"lmcache_mp_lookup_{short}_tokens"
        salts = [f'{{cache_salt="{salt}",model_name="m"}}' for salt in ("a", "b")]
        lines.append(f"# TYPE {family}_total counter")
        for labels, value in zip(salts, values, strict=True):
            lines.append(f"{family}_total{labels} {float(value)}")
        lines.append(f"# TYPE {family}_created gauge")
        lines.extend(f"{family}_created{labels} 1.7e9" for labels in salts)
    return ("\n".join(lines) + "\n").encode()


class _KVCacheServer:
    """Serve ``/status`` JSON and a ``/metrics`` response on a free port.

    A ``metrics_length`` larger than the body makes ``/metrics`` close before
    its declared length, as a server that dies mid-response does.
    """

    def __init__(
        self,
        metrics_body: bytes,
        metrics_code: int = 200,
        metrics_length: int | None = None,
    ) -> None:
        status_body = json.dumps(SAMPLE_STATUS).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                length = None
                if self.path == "/status":
                    code, body = 200, status_body
                elif self.path == "/metrics":
                    code, body = metrics_code, metrics_body
                    length = metrics_length
                else:
                    code, body = 404, b""
                self.send_response(code)
                self.send_header("Content-Length", str(length or len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_KVCacheServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _describe_kvcache_json(url: str) -> dict:
    """Run ``describe kvcache --format json`` against *url*; return metrics."""
    # Standard
    import io

    args = Namespace(target="kvcache", url=url, format="json", output=None)
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        DescribeCommand().execute(args)
    return json.loads(buf.getvalue())["metrics"]


_NA_HITS = {
    "lookup_requested_tokens": None,
    "lookup_hit_tokens": None,
    "lookup_hit_rate": None,
    "l1_hit_tokens": None,
    "l2_hit_tokens": None,
    "request_hit_rate": None,
}


class TestDescribeKvcacheCacheHits:
    """``describe kvcache`` reports lookup hits from the server's ``/metrics``."""

    def test_sums_series_and_derives_rates(self) -> None:
        with _KVCacheServer(_lookup_metrics_text(_HIT_SERIES)) as server:
            metrics = _describe_kvcache_json(server.url)

        assert metrics["cache_hits"] == {
            "lookup_requested_tokens": 1024,
            "lookup_hit_tokens": 768,
            "lookup_hit_rate": 75.0,
            "l1_hit_tokens": 512,
            "l2_hit_tokens": 256,
            "request_hit_rate": 77.7,  # 800 / 1030
        }
        # The /status sections are unaffected.
        assert metrics["health"] == "OK"
        assert metrics["cached_objects"] == 1024

    @pytest.mark.parametrize(
        ("body", "code", "length"),
        [
            pytest.param(b"", 404, None, id="metrics-endpoint-missing"),
            pytest.param(b"not { prometheus\n", 200, None, id="unparsable"),
            pytest.param(b"other_metric_total 1.0\n", 200, None, id="counters-absent"),
            pytest.param(
                _lookup_metrics_text(_HIT_SERIES)[:40], 200, 4096, id="truncated"
            ),
            # Valid exposition values that no token count can hold.
            pytest.param(
                _lookup_metrics_text({s: (float("nan"), 1) for s in _HIT_SERIES}),
                200,
                None,
                id="nan",
            ),
            pytest.param(
                _lookup_metrics_text({s: (float("inf"), 1) for s in _HIT_SERIES}),
                200,
                None,
                id="inf",
            ),
        ],
    )
    def test_unavailable_hits_show_na(
        self, body: bytes, code: int, length: int | None
    ) -> None:
        with _KVCacheServer(body, metrics_code=code, metrics_length=length) as server:
            metrics = _describe_kvcache_json(server.url)

        assert metrics["cache_hits"] == _NA_HITS
        assert metrics["health"] == "OK"

    def test_rates_na_before_first_lookup(self) -> None:
        zeros = {short: (0, 0) for short in _HIT_SERIES}
        with _KVCacheServer(_lookup_metrics_text(zeros)) as server:
            hits = _describe_kvcache_json(server.url)["cache_hits"]

        assert hits["lookup_requested_tokens"] == 0
        assert hits["lookup_hit_tokens"] == 0
        assert hits["lookup_hit_rate"] is None
        assert hits["request_hit_rate"] is None


class TestFetchMetricTotals:
    def test_reads_counter_families_and_omits_absent(self) -> None:
        with _KVCacheServer(_lookup_metrics_text(_HIT_SERIES)) as server:
            totals = fetch_metric_totals(
                f"{server.url}/metrics",
                ["lmcache_mp_lookup_hit_tokens", "lmcache_mp_not_exported"],
            )

        # ``_created`` timestamps are not added to the counter's sum.
        assert totals == {"lmcache_mp_lookup_hit_tokens": 768.0}

    def test_connection_refused(self) -> None:
        assert fetch_metric_totals("http://127.0.0.1:19999/metrics", ["x"]) == {}


# ---------------------------------------------------------------------------
# describe engine
# ---------------------------------------------------------------------------

SAMPLE_MODELS = {
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 131072,
        }
    ],
}


class TestDescribeEngineFields:
    """Test that ``_describe_engine`` extracts fields from a sample
    ``/v1/models`` response plus a ``/health`` result."""

    def _run(self, models, is_healthy):
        """Execute ``describe engine`` with mocked HTTP and return metrics dict."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = "http://localhost:8000"
            format = "json"
            output = None

        with (
            patch(
                "lmcache.cli.commands.describe.fetch_json",
                return_value=models,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_health",
                return_value=is_healthy,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_running_requests",
                return_value=3,
            ),
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())
            return json.loads(buf.getvalue())

    def test_field_extraction(self):
        """Verify model, context, status, and running requests are populated."""
        output = self._run(SAMPLE_MODELS, is_healthy=True)
        assert output["title"] == "Inference Engine"
        m = output["metrics"]
        assert m["model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert m["max_context"] == 131072
        assert m["status"] == "OK"
        assert m["running_requests"] == 3

    def test_unhealthy(self):
        """Verify status shows UNHEALTHY when /health is not OK."""
        output = self._run(SAMPLE_MODELS, is_healthy=False)
        assert output["metrics"]["status"] == "UNHEALTHY"

    def test_missing_model(self):
        """Verify empty model list renders model/context as None."""
        output = self._run({"object": "list", "data": []}, is_healthy=True)
        m = output["metrics"]
        assert m["model"] is None
        assert m["max_context"] is None


class TestDescribeEngineDefaultUrl:
    def test_engine_default_url(self):
        """Verify url=None resolves to the engine default (localhost:8000)."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = None
            format = "json"
            output = None

        captured: dict[str, str] = {}

        def fake_fetch(url, *args, **kwargs):
            captured["url"] = url
            return SAMPLE_MODELS

        with (
            patch(
                "lmcache.cli.commands.describe.fetch_json",
                side_effect=fake_fetch,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_health",
                return_value=True,
            ),
            patch(
                "lmcache.cli.commands.describe.fetch_running_requests",
                return_value=3,
            ),
        ):
            # Standard
            import io

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd.execute(FakeArgs())

        assert "8000" in captured["url"]


class TestDescribeEngineErrors:
    def test_error_exits_1(self):
        """Verify sys.exit(1) when the engine cannot be reached."""
        # First Party
        from lmcache.cli.commands.describe import DescribeCommand

        cmd = DescribeCommand()

        class FakeArgs:
            target = "engine"
            url = "http://localhost:8000"
            format = None
            output = None

        with patch(
            "lmcache.cli.commands.describe.fetch_json",
            side_effect=DescribeError("Cannot connect"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd.execute(FakeArgs())
            assert exc_info.value.code == 1

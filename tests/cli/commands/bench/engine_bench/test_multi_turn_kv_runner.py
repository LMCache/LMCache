# SPDX-License-Identifier: Apache-2.0
"""Run the example through its CLI, real SDK, workload and JSON report."""

# Standard
from pathlib import Path
from unittest.mock import patch
import importlib.util
import json
import types

# Third Party
from openai import AsyncOpenAI
import httpx
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench import request_sender as sender_mod
from lmcache.cli.commands.bench.engine_bench.workloads import multi_turn_kv as mt

# Local
from .fake_tokenizer import make_fake_tokenizer
from .test_token_request_sender import token_response
from .workloads.test_multi_turn_kv import result


def load_runner() -> types.ModuleType:
    """Load the standalone example without changing the CLI workload registry."""
    path = (
        Path(__file__).resolve().parents[5]
        / "examples/benchmarks/multi_turn_kv/benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("multi_turn_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMultiTurnKVRunner:
    @pytest.mark.parametrize(
        "require_l2,loaded,exit_code,failed",
        [
            (False, 0, None, False),
            (True, 4, None, False),
            (True, 0, 2, False),
            (False, 0, 1, True),
        ],
    )
    def test_cli_to_http_and_report(
        self,
        tmp_path: Path,
        require_l2: bool,
        loaded: int,
        exit_code: int | None,
        failed: bool,
    ) -> None:
        runner = load_runner()
        requests: list[list[int]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            prompt = json.loads(request.content)["prompt"]
            requests.append(prompt)
            return token_response(prompt, [91, 92], usage_count=0 if failed else None)

        client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        output = tmp_path / "report.json"
        argv = [
            "benchmark.py",
            "--engine-url",
            "http://test",
            "--model",
            "m",
            "--output",
            str(output),
            "--num-sessions",
            "2",
            "--turns",
            "2",
            "--short-context",
            "4",
            "--long-context",
            "8",
            "--turn-tokens",
            "2",
            "--output-tokens",
            "2",
            "--max-context-tokens",
            "16",
            "--vocab-size",
            "64",
            "--quiet",
            "--metrics-settle-seconds",
            "0",
        ]
        if require_l2:
            argv.extend(["--metrics-url", "http://metrics/metrics", "--require-l2"])
        after = {
            "lookup_hit_tokens": 8.0,
            "l2_hit_tokens": float(loaded),
            "l2_loaded_chunks": float(loaded),
        }
        with (
            patch("sys.argv", argv),
            patch.object(mt, "try_load_tokenizer", return_value=make_fake_tokenizer()),
            patch.object(sender_mod, "AsyncOpenAI", return_value=client),
            patch.object(runner, "read_counters", side_effect=[{}, after]),
        ):
            if exit_code is None:
                runner.main()
            else:
                with pytest.raises(SystemExit) as exc:
                    runner.main()
                assert exc.value.code == exit_code
        report = json.loads(output.read_text())
        assert report["summary"]["successful"] == (0 if failed else 4)
        assert report["summary"]["failed"] == (2 if failed else 0)
        assert report["unattempted_turns"] == (2 if failed else 0)
        assert report["phases"]["initial"]["attempted"] == 2
        assert report["phases"]["continuation"]["attempted"] == (0 if failed else 2)
        assert set(report["sessions"]) == {"0", "1"}
        assert len(report["turns"]) == (2 if failed else 4)
        if not failed:
            assert any(
                prompt[:6] + [91, 92] == later[:8]
                for prompt in requests
                for later in requests
                if len(prompt) == 6 and len(later) > 6
            )
        assert report["l2_activity"] == (
            "unavailable"
            if not require_l2
            else "observed"
            if loaded
            else "not_observed"
        )
        assert client.is_closed()

    def test_counter_labels_missing_series_resets_and_invalid_samples(self) -> None:
        runner = load_runner()
        body = b"""# TYPE lmcache_mp_lookup_hit_tokens_total counter
lmcache_mp_lookup_hit_tokens_total{model_name="model with spaces"} 12
lmcache_mp_lookup_hit_tokens_total{model_name="other"} 8
lmcache_mp_l2_prefetch_load_completed_chunks_total{l2_name="fs"} 3
lmcache_mp_l2_prefetch_failure_chunks_total nan
unrelated 999
"""
        with patch.object(runner.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = body
            counters = runner.read_counters("http://metrics/metrics")
        assert counters == {"lookup_hit_tokens": 20, "l2_loaded_chunks": 3}
        deltas = runner.counter_deltas({"lookup_hit_tokens": 25}, counters)
        assert deltas["lookup_hit_tokens"] is None
        assert deltas["l2_loaded_chunks"] == 3
        assert deltas["l2_hit_tokens"] is None
        assert all(
            value is None for value in runner.counter_deltas(None, counters).values()
        )

    def test_percentiles_and_slo_include_failed_attempts(self) -> None:
        runner = load_runner()
        assert runner.summarize([1.0, 2.0, 3.0]) == {
            "p50": 2.0,
            "p95": 2.9,
            "p99": 2.98,
        }
        assert runner.summarize([])["p50"] is None
        assert runner.summarize([4.0])["p99"] == 4.0
        successful = mt.ConversationTurn(0, "short", 0, "initial", 0.02, 0, result("a"))
        failed = mt.ConversationTurn(
            1, "long", 0, "initial", 0.04, 0, result("b", False)
        )
        summary = runner.turn_summary([successful, failed], 35.0)
        assert summary["attempted"] == 2
        assert summary["meeting_queue_plus_ttft_slo"] == 1
        assert summary["queue_ms"]["p50"] == 30.0

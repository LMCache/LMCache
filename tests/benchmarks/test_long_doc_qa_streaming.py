# SPDX-License-Identifier: Apache-2.0
"""Exercise long-document timing through the real OpenAI SDK's SSE decoder."""

# Standard
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
import asyncio
import csv
import importlib.util
import json
import subprocess
import sys
import threading

# Third Party
from openai import AsyncOpenAI
import httpx
import pytest


class StreamClock:
    """Advance elapsed time as SSE frames arrive; move wall time backwards."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self.wall = 1000.0

    def perf_counter(self) -> float:
        """Return elapsed seconds controlled by the stream."""
        return self.elapsed

    def time(self) -> float:
        """Return wall time independently of elapsed request timing."""
        self.wall -= 1
        return self.wall


class EventStream(httpx.AsyncByteStream):
    """Deliver valid SSE frames, including fragmented HTTP reads."""

    def __init__(
        self, clock: StreamClock, events: list[tuple[float, dict[str, object]]]
    ) -> None:
        self.clock = clock
        self.events = events

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for elapsed, event in self.events:
            self.clock.elapsed = elapsed
            frame = f"data: {json.dumps(event)}\n\n".encode()
            # The SDK must assemble events independently of HTTP read boundaries.
            yield frame[:7]
            yield frame[7:]
        self.clock.elapsed = 2.0
        yield b"data: [DONE]\n\n"


@pytest.fixture
def long_doc_qa_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the benchmark without invoking its CLI."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/long_doc_qa/long_doc_qa.py"
    spec = importlib.util.spec_from_file_location("long_doc_qa_stream_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write_resp", lambda text: None)
    return module


def stream_event(
    completions: bool, text: str, finish_reason: str | None = None
) -> dict[str, object]:
    """Build a completion or chat-completion SSE payload."""
    choice: dict[str, object] = {"index": 0, "finish_reason": finish_reason}
    if completions:
        choice["text"] = text
    else:
        choice["delta"] = {"content": text}
    return {
        "id": "test",
        "object": "text_completion" if completions else "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [choice],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("completions", [False, True])
@pytest.mark.parametrize("ending", ["complete", "partial", "empty", "single"])
async def test_stream_latency_and_completion(
    long_doc_qa_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    completions: bool,
    ending: str,
) -> None:
    """Content defines TTFT/TTOT; empty, usage and termination frames do not."""
    clock = StreamClock()
    monkeypatch.setattr(long_doc_qa_module, "time", clock)
    monkeypatch.setattr(long_doc_qa_module, "completions_mode", completions)
    events = [(0.1, stream_event(completions, ""))]
    if ending != "empty":
        events.append((0.25, stream_event(completions, "first")))
    if ending in ("complete", "partial"):
        events.append((0.75, stream_event(completions, " last")))
    if ending != "partial":
        events.append((1.0, stream_event(completions, "", "stop")))
    events.append(
        (1.5, {"id": "test", "choices": [], "usage": {"completion_tokens": 2}})
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=EventStream(clock, events),
        )
    )
    async with AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=transport),
    ) as client:
        stats = await long_doc_qa_module.process_single_prompt(
            client, "test-model", "prompt", 0, 1, 2, asyncio.Semaphore(1)
        )
    assert stats.successful == (ending in ("complete", "single"))
    assert stats.ttft == (-1 if ending == "empty" else 0.25)
    assert (
        stats.ttot
        == {"complete": 0.75, "partial": 0.75, "single": 0.25, "empty": -1}[ending]
    )
    assert stats.request_end < stats.request_start


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_cli_summary_and_failed_stream_artifacts(
    long_doc_qa_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    partial: bool,
) -> None:
    """CLI reports request TTOT and fails incomplete streams after saving CSVs."""
    monkeypatch.chdir(tmp_path)
    args = long_doc_qa_module.create_argument_parser().parse_args(
        [
            "--model",
            "test-model",
            "--num-documents",
            "1",
            "--repeat-count",
            "1",
            "--json-output",
        ]
    )

    async def requests(**kwargs: object) -> list[object]:
        return [
            long_doc_qa_module.RequestStats(0, 10.0, 0.2, 12.0, not partial, ttot=0.8)
        ]

    monkeypatch.setattr(long_doc_qa_module, "test_long_document_qa", requests)
    if partial:
        with pytest.raises(RuntimeError, match="unsuccessful streaming"):
            await long_doc_qa_module.main(args)
        assert not capsys.readouterr().out.rstrip().endswith("}")
    else:
        await long_doc_qa_module.main(args)
        summary = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert summary["query_ttot_per_prompt"] == 0.8
        assert summary["warmup_ttot_per_prompt"] == 0.8
        assert summary["query_ttft_per_prompt"] == 0.2
    for phase in ("warmup", "query"):
        frame = long_doc_qa_module.pd.read_csv(tmp_path / f"{phase}_round.csv")
        assert frame["ttot"].tolist() == [0.8]
        assert frame["successful"].tolist() == [not partial]


@pytest.mark.parametrize("include_ttot", [False, True])
def test_ci_diagnostics_ttot_and_legacy_csv(tmp_path: Path, include_ttot: bool) -> None:
    """CI summarizes successful TTOT samples and still reads archived CSVs."""
    script = (
        Path(__file__).resolve().parents[2]
        / ".buildkite/k3_tests/multiprocess/scripts/workloads/common"
        / "long_doc_qa_diagnostics.py"
    )
    for phase in ("warmup", "query"):
        fields = ["request_start", "request_end", "ttft", "successful"]
        rows = [[0, 1, 0.2, True], [0, 0.001, 0.001, False]]
        if include_ttot:
            fields.append("ttot")
            rows[0].append(0.8)
            rows[1].append(0.001)
        with (tmp_path / f"{phase}_round.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            writer.writerows(rows)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--phase",
            "lmcache",
            "--engine",
            "vllm",
            "--results-dir",
            str(tmp_path),
            "--command",
            "benchmark",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "successful=1 failed=1" in result.stdout
    if include_ttot:
        assert (
            "ttot: min=0.800000 median=0.800000 p95=0.800000 max=0.800000"
            in result.stdout
        )
    else:
        assert "ttot:" not in result.stdout


@pytest.mark.parametrize("completions", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_benchmark_cli_over_http(
    tmp_path: Path, completions: bool, partial: bool
) -> None:
    """The full CLI rejects partial HTTP/SSE streams and saves both CSVs."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            """Serve an SSE response with or without a terminal finish reason."""
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                stream_event(completions, "first"),
                stream_event(completions, " last"),
            ]
            if not partial:
                events.append(stream_event(completions, "", "stop"))
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test HTTP server quiet."""

    script = (
        Path(__file__).resolve().parents[2] / "benchmarks/long_doc_qa/long_doc_qa.py"
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            command = [
                sys.executable,
                str(script),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "test-model",
                "--num-documents",
                "1",
                "--document-length",
                "8",
                "--repeat-count",
                "1",
                "--output-len",
                "2",
                "--json-output",
            ]
            if completions:
                command.append("--completions")
            result = subprocess.run(
                command,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        finally:
            server.shutdown()
            worker.join()
    assert result.returncode == (1 if partial else 0), result.stdout + result.stderr
    for phase in ("warmup", "query"):
        with (tmp_path / f"{phase}_round.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 1
        row = rows[0]
        assert row["successful"] == str(not partial)
        assert 0 <= float(row["ttft"]) <= float(row["ttot"])
    if partial:
        assert "unsuccessful streaming requests" in result.stderr
    else:
        summary = json.loads(result.stdout.splitlines()[-1])
        assert summary["query_ttot_per_prompt"] == float(row["ttot"])

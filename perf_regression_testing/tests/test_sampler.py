# SPDX-License-Identifier: Apache-2.0
# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import threading
import time

# First Party
from perf_regression_testing.runner.sampler import Sampler

ENGINE = """# HELP vllm:num_preemptions_total x
vllm:num_preemptions_total{engine="0"} 2
vllm:num_preemptions_total{engine="1"} 3
vllm:prompt_tokens_by_source_total{source="local_compute"} 10
vllm:generation_tokens_total 7
"""


def _serve(text: str) -> tuple[HTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/metrics"


def test_counters_are_summed_over_labels_and_urls(tmp_path: Path) -> None:
    servers = [_serve(ENGINE), _serve(ENGINE)]
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, [u for _, u in servers], interval_s=0.05)
    s.start()
    time.sleep(0.1)
    s.stop()
    for srv, _ in servers:
        srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["vllm:num_preemptions_total"] == 10.0
    assert rec["vllm:prompt_tokens_by_source/local_compute"] == 20.0
    assert "vllm:generation_tokens_total" not in rec


def test_an_unreachable_url_is_recorded_not_raised(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    srv.shutdown()
    srv.server_close()
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, [url], interval_s=0.05)
    s.start()
    time.sleep(0.1)
    s.stop()
    rec = json.loads(out.read_text().splitlines()[0])
    assert f"err {url}" in rec

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
vllm:num_requests_running 4
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


def test_counters_are_summed_over_labels(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, url, interval_s=0.05)
    s.start()
    time.sleep(0.1)
    s.stop()
    srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["vllm:num_preemptions_total"] == 5.0
    assert rec["vllm:prompt_tokens_by_source/local_compute"] == 10.0
    assert rec["vllm:generation_tokens_total"] == 7.0
    assert "vllm:num_requests_running" not in rec


def test_an_unreachable_url_is_recorded_not_raised(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    srv.shutdown()
    srv.server_close()
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, url, interval_s=0.05)
    s.start()
    time.sleep(0.1)
    s.stop()
    rec = json.loads(out.read_text().splitlines()[0])
    assert "err" in rec


def test_cgroup_memory_is_sampled(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    (tmp_path / "memory.stat").write_text("anon 100\nfile 200\nfile_mapped 7\n")
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, url, interval_s=0.05, cgroup=tmp_path)
    s.start()
    time.sleep(0.1)
    s.stop()
    srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert (rec["mem_anon"], rec["mem_file"]) == (100, 200)
    assert "file_mapped" not in str(rec)


def test_a_gone_cgroup_is_recorded_not_raised(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, url, interval_s=0.05, cgroup=tmp_path / "gone")
    s.start()
    time.sleep(0.1)
    s.stop()
    srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert "mem_err" in rec and "mem_anon" not in rec


def test_block_io_is_sampled_and_a_gone_container_recorded(tmp_path: Path) -> None:
    srv, url = _serve(ENGINE)
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, url, interval_s=0.05, io=lambda: {"rbytes": 1, "wbytes": 2})
    s.start()
    time.sleep(0.1)
    s.stop()

    def gone() -> dict[str, int]:
        raise FileNotFoundError("io.stat")

    out2 = tmp_path / "gone.jsonl"
    s = Sampler(out2, url, interval_s=0.05, io=gone)
    s.start()
    time.sleep(0.1)
    s.stop()
    srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert (rec["io_rbytes"], rec["io_wbytes"]) == (1, 2)
    rec = json.loads(out2.read_text().splitlines()[0])
    assert "io_err" in rec and "io_wbytes" not in rec

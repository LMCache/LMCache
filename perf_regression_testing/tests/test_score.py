# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

# First Party
from perf_regression_testing.runner.score import score


def _request(start_s: float, end_s: float) -> dict[str, object]:
    return {
        "metadata": {
            "benchmark_phase": "profiling",
            "request_start_ns": int(start_s * 1e9),
            "request_end_ns": int(end_s * 1e9),
        },
        "metrics": {
            "time_to_first_token": {"value": 500.0},
            "request_latency": {"value": (end_s - start_s) * 1e3},
            "output_sequence_length": {"value": 11},
        },
    }


def _counters(t: float, prompt: float, generated: float) -> dict[str, float]:
    return {
        "t": t,
        "vllm:prompt_tokens_by_source/local_compute": prompt / 4,
        "vllm:prompt_tokens_by_source/local_cache_hit": prompt / 4,
        "vllm:prompt_tokens_by_source/external_kv_transfer": prompt / 2,
        "vllm:generation_tokens_total": generated,
        "vllm:num_preemptions_total": 0.0,
    }


def test_token_throughput_is_the_window_delta_of_the_server_counters(
    tmp_path: Path,
) -> None:
    (tmp_path / "aiperf").mkdir()
    reqs = [_request(100.0, 104.0), _request(101.0, 150.0)]
    (tmp_path / "aiperf" / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reqs)
    )
    # Counters before, inside and after the 100 s window [100, 200].
    samples = [
        _counters(50, 0, 0),
        _counters(150, 5000, 1000),
        _counters(250, 15000, 3000),
    ]
    (tmp_path / "samples.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    s = score(tmp_path, window_s=100.0)
    # Interpolated at 100 s and 200 s: prompt 2500 -> 10000, generated 500 -> 2000.
    assert s["input_tok_s"] == 75.0
    assert s["output_tok_s"] == 15.0
    assert s["requests"] == 2

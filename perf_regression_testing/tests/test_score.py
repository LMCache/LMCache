# SPDX-License-Identifier: Apache-2.0
# Standard
from array import array
from pathlib import Path
import json

# First Party
from perf_regression_testing.runner.score import PROMPT_DIR, score


def _request(
    start_s: float, end_s: float, phase: str = "profiling", rid: str = ""
) -> dict[str, object]:
    return {
        "metadata": {
            "x_request_id": rid,
            "benchmark_phase": phase,
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
    assert s["cache_hit_share"] == 0.75
    assert s["ideal_hit_share"] is None


def test_ideal_hit_share_is_the_token_prefix_shared_with_any_earlier_prompt(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / PROMPT_DIR
    prompts.mkdir(parents=True)
    a, b = list(range(128)), list(range(1000, 1064))
    # warm caches a; w1 extends it; w2 shares nothing; w3 never reached the
    # server; w4 leaves w1 30 tokens into a block; w5 is all cached, so it
    # hits all but its last token; late is after the window, which ends at 200.
    sent = {
        "warm": a,
        "w1": a + b + [7] * 10,
        "w2": list(range(5000, 5100)),
        "w4": a + b[:30] + [8],
        "w5": a[:50],
        "late": a + b,
    }
    for rid, ids in sent.items():
        (prompts / f"{rid}.i32").write_bytes(array("i", ids).tobytes())
    reqs = [
        _request(90.0, 95.0, "warmup", "warm"),
        _request(100.0, 104.0, rid="w1"),
        _request(102.0, 106.0, rid="w2"),
        _request(103.0, 107.0, rid="w3"),
        _request(104.0, 108.0, rid="w4"),
        _request(105.0, 109.0, rid="w5"),
        _request(250.0, 260.0, rid="late"),
    ]
    (tmp_path / "aiperf" / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reqs)
    )
    (tmp_path / "samples.jsonl").write_text(
        "".join(json.dumps(_counters(t, t, t)) + "\n" for t in (0, 300))
    )
    s = score(tmp_path, window_s=100.0)
    assert s["ideal_hit_share"] == (128 + 0 + 158 + 49) / (202 + 100 + 159 + 50)

# SPDX-License-Identifier: Apache-2.0
# Standard
from array import array
from pathlib import Path
import json

# First Party
from perf_regression_testing.runner.score import PROMPT_DIR, score


def _request(
    start_s: float,
    end_s: float,
    phase: str = "profiling",
    rid: str = "",
    ttft_ms: float = 500.0,
) -> dict[str, object]:
    return {
        "metadata": {
            "x_request_id": rid,
            "benchmark_phase": phase,
            "request_start_ns": int(start_s * 1e9),
            "request_end_ns": int(end_s * 1e9),
        },
        "metrics": {
            "time_to_first_token": {"value": ttft_ms},
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
    # hits all but its last token; the window ends at 200, and slow starts in
    # it but is still in prefill then, and late starts after it.
    sent = {
        "warm": a,
        "w1": a + b + [7] * 10,
        "w2": list(range(5000, 5100)),
        "w4": a + b[:30] + [8],
        "w5": a[:50],
        "slow": list(range(9000, 9300)),
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
        _request(190.0, 260.0, rid="slow", ttft_ms=30_000.0),
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


def test_spec_accept_len_counts_the_target_token_and_accepted_drafts(
    tmp_path: Path,
) -> None:
    (tmp_path / "aiperf").mkdir()
    (tmp_path / "aiperf" / "profile_export.jsonl").write_text(
        json.dumps(_request(100.0, 104.0)) + "\n"
    )
    samples = [_counters(t, 0, 0) for t in (50, 250)]
    plain = tmp_path / "samples.jsonl"
    plain.write_text("".join(json.dumps(s) + "\n" for s in samples))
    assert score(tmp_path, window_s=100.0)["spec_accept_len"] is None

    for s, drafts in zip(samples, (0, 400), strict=True):
        s["vllm:spec_decode_num_drafts_total"] = drafts
        s["vllm:spec_decode_num_accepted_tokens_total"] = drafts * 2.5
    plain.write_text("".join(json.dumps(s) + "\n" for s in samples))
    assert score(tmp_path, window_s=100.0)["spec_accept_len"] == 3.5


def test_disk_write_rates_split_warmup_from_the_window(tmp_path: Path) -> None:
    (tmp_path / "aiperf").mkdir()
    reqs = [_request(100.0, 104.0), _request(101.0, 150.0)]
    (tmp_path / "aiperf" / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reqs)
    )
    # Warmup 0-100 s writes 1 GB/s; the window [100, 200] 2 GB/s, then 4 in its
    # last quarter.
    io = {0: 0, 100: 100e9, 175: 250e9, 200: 350e9, 250: 350e9}
    samples = [_counters(t, 0, 0) | {"io_wbytes": w} for t, w in io.items()]
    (tmp_path / "samples.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    s = score(tmp_path, window_s=100.0)
    assert s["disk_write_gb_s_warmup"] == 1.0
    assert s["disk_write_gb_s_steady"] == 2.5
    assert s["disk_write_gb_s_steady_quarters"] == [2.0, 2.0, 2.0, 4.0]


def test_no_disk_rates_without_io_samples(tmp_path: Path) -> None:
    (tmp_path / "aiperf").mkdir()
    (tmp_path / "aiperf" / "profile_export.jsonl").write_text(
        json.dumps(_request(100.0, 104.0)) + "\n"
    )
    samples = [_counters(t, 0, 0) for t in (50, 250)]
    (tmp_path / "samples.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    assert "disk_write_gb_s_steady" not in score(tmp_path, window_s=100.0)

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a small vLLM MP smoke against Python or native MP servers.

The parent process owns the MP server. Each vLLM generation runs in a fresh
child process so CUDA initialization and vLLM's engine-core process use the
spawn start method cleanly. The second generation reuses the same prompt and
server so the smoke can observe reuse. Native mode also checks server counters
to prove the second process served a retrieve path.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

_RESULT_PREFIX = "LMCACHE_VLLM_SMOKE_RESULT "


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str) -> None:
    deadline = time.time() + 20
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"server did not become ready at {url}: {last_error}")


def _wait_for_ping(port: int) -> None:
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.protocol import RequestType

    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
    deadline = time.time() + 20
    last_error: Exception | None = None
    try:
        while time.time() < deadline:
            try:
                if client.submit_request(RequestType.PING, []).result(timeout=1):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.2)
    finally:
        client.close()
    raise RuntimeError(f"server did not respond to PING: {last_error}")


def _status(http_port: int) -> dict[str, object]:
    return json.loads(
        urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/status",
            timeout=10,
        ).read()
    )


def _summary(status: dict[str, object]) -> dict[str, object]:
    metrics = status["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("status metrics must be a dict")
    cache_hits = metrics["cache_hits"]
    cache_misses = metrics["cache_misses"]
    if not isinstance(cache_hits, int) or not isinstance(cache_misses, int):
        raise TypeError("cache hit/miss counters must be ints")
    cache_lookup_count = cache_hits + cache_misses
    return {
        "registered_context_count": status["registered_context_count"],
        "store_count": metrics["store_count"],
        "retrieve_count": metrics["retrieve_count"],
        "lookup_count": metrics["lookup_count"],
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / cache_lookup_count
        if cache_lookup_count
        else 0.0,
        "unsupported_count": metrics["unsupported_count"],
        "transfer_lock_count": metrics.get("transfer_lock_count", 0),
        "transfer_lock_failure_count": metrics.get(
            "transfer_lock_failure_count",
            0,
        ),
    }


def _int_field(values: dict[str, object], key: str) -> int:
    value = values[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _clock_ticks_per_second() -> int:
    value = os.sysconf("SC_CLK_TCK")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("SC_CLK_TCK must be a positive integer")
    return value


def _read_process_resources(pid: int) -> dict[str, int | float] | None:
    """Best-effort Linux process resource snapshot for benchmark reports."""
    proc_dir = Path("/proc") / str(pid)
    try:
        status_text = (proc_dir / "status").read_text(encoding="utf-8")
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
    except OSError:
        return None

    rss_bytes = 0
    rss_peak_bytes = 0
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            rss_bytes = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            rss_peak_bytes = int(line.split()[1]) * 1024

    stat_tail = stat_text.rsplit(")", maxsplit=1)[1].split()
    ticks = _clock_ticks_per_second()
    user_cpu_s = int(stat_tail[11]) / ticks
    system_cpu_s = int(stat_tail[12]) / ticks
    return {
        "rss_bytes": rss_bytes,
        "rss_peak_bytes": rss_peak_bytes,
        "user_cpu_s": user_cpu_s,
        "system_cpu_s": system_cpu_s,
        "total_cpu_s": user_cpu_s + system_cpu_s,
        "thread_count": int(stat_tail[17]),
    }


def _process_resource_delta(
    before: dict[str, int | float] | None,
    after: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if before is None or after is None:
        return None
    return {
        "rss_bytes_delta": int(after["rss_bytes"]) - int(before["rss_bytes"]),
        "rss_peak_bytes": after["rss_peak_bytes"],
        "user_cpu_s_delta": float(after["user_cpu_s"]) - float(before["user_cpu_s"]),
        "system_cpu_s_delta": float(after["system_cpu_s"])
        - float(before["system_cpu_s"]),
        "total_cpu_s_delta": float(after["total_cpu_s"]) - float(before["total_cpu_s"]),
        "thread_count_end": after["thread_count"],
    }


def _unexpected_native_stderr_lines(stderr: str) -> list[str]:
    expected_prefixes = ("LMCache native MP server listening on ",)
    return [
        line
        for line in stderr.splitlines()
        if line and not any(line.startswith(prefix) for prefix in expected_prefixes)
    ]


def _make_prompts(prompt: str, batch_size: int) -> list[str]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if batch_size == 1:
        return [prompt]
    return [f"{prompt}\nRequest variant {idx}" for idx in range(batch_size)]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def _summary_stats(values: list[float]) -> dict[str, float | None | list[float]]:
    return {
        "mean": _mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "values": values,
    }


def _latency_stats_ms(
    values: list[float],
) -> dict[str, float | int | None | list[float]]:
    stats = _summary_stats(values)
    return {
        "count": len(values),
        "mean_ms": stats["mean"],
        "p50_ms": stats["p50"],
        "p95_ms": stats["p95"],
        "p99_ms": stats["p99"],
        "values_ms": stats["values"],
    }


def _request_ttft_s(output: Any) -> float | None:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    first_token_latency = getattr(metrics, "first_token_latency", None)
    if isinstance(first_token_latency, int | float) and first_token_latency > 0:
        return float(first_token_latency)
    first_token_ts = getattr(metrics, "first_token_ts", None)
    scheduled_ts = getattr(metrics, "scheduled_ts", None)
    if (
        isinstance(first_token_ts, int | float)
        and isinstance(scheduled_ts, int | float)
        and first_token_ts >= scheduled_ts
        and scheduled_ts > 0
    ):
        return float(first_token_ts - scheduled_ts)
    return None


def _summarize_generation_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "output_text": output.outputs[0].text,
            "output_token_count": len(output.outputs[0].token_ids),
            "ttft_s": _request_ttft_s(output),
        }
        for output in outputs
    ]


def _run_generate_round(
    llm: Any,
    prompts: list[str],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    # Third Party
    from vllm import SamplingParams

    generate_start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0),
    )
    generate_elapsed_s = time.perf_counter() - generate_start
    generation_outputs = _summarize_generation_outputs(outputs)
    output_token_counts = [
        int(output["output_token_count"]) for output in generation_outputs
    ]
    total_output_tokens = sum(output_token_counts)
    ttfts = [
        float(output["ttft_s"])
        for output in generation_outputs
        if isinstance(output["ttft_s"], int | float)
    ]
    return {
        "generate_elapsed_s": generate_elapsed_s,
        "output_text": generation_outputs[0]["output_text"],
        "output_token_count": generation_outputs[0]["output_token_count"],
        "output_tokens_per_s": total_output_tokens / generate_elapsed_s,
        "outputs": generation_outputs,
        "prompt_count": len(prompts),
        "ttft_s_mean": _mean(ttfts),
        "ttft_s_values": ttfts,
        "total_output_tokens": total_output_tokens,
    }


def _summarize_rounds(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    generate_elapsed_values = [
        float(round_result["generate_elapsed_s"]) for round_result in rounds
    ]
    throughput_values = [
        float(round_result["output_tokens_per_s"]) for round_result in rounds
    ]
    ttft_values = [
        float(ttft)
        for round_result in rounds
        for ttft in round_result["ttft_s_values"]
        if isinstance(ttft, int | float)
    ]
    generate_elapsed_stats = _summary_stats(generate_elapsed_values)
    throughput_stats = _summary_stats(throughput_values)
    ttft_stats = _summary_stats(ttft_values)
    return {
        "generate_elapsed_s": generate_elapsed_stats,
        "generate_elapsed_s_mean": generate_elapsed_stats["mean"],
        "generate_elapsed_s_p50": generate_elapsed_stats["p50"],
        "generate_elapsed_s_p95": generate_elapsed_stats["p95"],
        "generate_elapsed_s_p99": generate_elapsed_stats["p99"],
        "generate_elapsed_s_values": generate_elapsed_stats["values"],
        "measured_rounds": len(rounds),
        "output_tokens_per_s": throughput_stats,
        "output_tokens_per_s_mean": throughput_stats["mean"],
        "output_tokens_per_s_p50": throughput_stats["p50"],
        "output_tokens_per_s_p95": throughput_stats["p95"],
        "output_tokens_per_s_p99": throughput_stats["p99"],
        "output_tokens_per_s_values": throughput_stats["values"],
        "ttft_s": ttft_stats,
        "ttft_s_mean": ttft_stats["mean"],
        "ttft_s_p50": ttft_stats["p50"],
        "ttft_s_p95": ttft_stats["p95"],
        "ttft_s_p99": ttft_stats["p99"],
        "ttft_s_values": ttft_stats["values"],
    }


def _current_vllm_kv_cache_layout() -> str | None:
    try:
        # Third Party
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        return get_kv_cache_layout()
    except Exception:  # noqa: BLE001
        return None


def _server_command(
    server: str,
    *,
    zmq_port: int,
    http_port: int,
    l1_size_gb: float,
    chunk_size: int,
    disk_path: Path,
    native_gpu_hot_cache: bool,
) -> list[str]:
    if server == "native":
        # First Party
        from lmcache.v1.multiprocess.native_launcher import ensure_native_binary

        argv = [
            str(ensure_native_binary(enable_cuda=True)),
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--l1-size-gb",
            str(l1_size_gb),
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            str(chunk_size),
            "--cxx-disk-path",
            str(disk_path),
        ]
        if native_gpu_hot_cache:
            argv.append("--cuda-gpu-hot-cache")
        return argv
    if server == "python":
        return [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(zmq_port),
            "--l1-size-gb",
            str(l1_size_gb),
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            str(chunk_size),
            "--disable-observability",
        ]
    raise ValueError(f"unsupported server: {server!r}")


def _wait_for_server(server: str, zmq_port: int, http_port: int) -> None:
    if server == "native":
        _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
    else:
        _wait_for_ping(zmq_port)


def _run_worker(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.kv_cache_layout:
        os.environ["VLLM_KV_CACHE_LAYOUT"] = args.kv_cache_layout
    if args.use_layerwise:
        os.environ["LMCACHE_USE_LAYERWISE"] = "1"
    if args.worker_trace_output:
        os.environ["LMCACHE_MP_TRACE_FILE"] = str(args.worker_trace_output)
        os.environ["LMCACHE_MP_TRACE_LABEL"] = args.worker_label

    # Third Party
    from vllm import LLM
    from vllm.config import KVTransferConfig

    extra_config: dict[str, object] = {
        "lmcache.mp.host": "127.0.0.1",
        "lmcache.mp.port": args.port,
    }
    if args.raw_cuda_ipc:
        extra_config["lmcache.mp.raw_cuda_ipc"] = True

    transfer_config = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_connector_module_path="lmcache.integration.vllm.lmcache_mp_connector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )
    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=False,
        enable_prefix_caching=False,
        kv_transfer_config=transfer_config,
    )
    init_elapsed_s = time.perf_counter() - init_start
    resolved_kv_cache_layout = _current_vllm_kv_cache_layout()
    prompts = _make_prompts(args.prompt, args.batch_size)
    warmup_rounds = []
    for _ in range(args.steady_state_warmup_rounds):
        warmup_rounds.append(
            _run_generate_round(llm, prompts, max_tokens=args.max_tokens)
        )
    measured_rounds = []
    for _ in range(args.steady_state_rounds):
        measured_rounds.append(
            _run_generate_round(llm, prompts, max_tokens=args.max_tokens)
        )
    kvcache_checksum = _worker_kvcache_checksum_probe(args)
    first_round = measured_rounds[0]
    print(
        _RESULT_PREFIX
        + json.dumps(
            {
                "elapsed_s": init_elapsed_s + float(first_round["generate_elapsed_s"]),
                "generate_elapsed_s": first_round["generate_elapsed_s"],
                "init_elapsed_s": init_elapsed_s,
                "kvcache_checksum": kvcache_checksum,
                "output_text": first_round["output_text"],
                "output_token_count": first_round["output_token_count"],
                "output_tokens_per_s": first_round["output_tokens_per_s"],
                "outputs": first_round["outputs"],
                "prompt_count": len(prompts),
                "requested_kv_cache_layout": args.kv_cache_layout or None,
                "steady_state": _summarize_rounds(measured_rounds),
                "steady_state_rounds": measured_rounds,
                "steady_state_warmup_rounds": len(warmup_rounds),
                "ttft_s_mean": first_round["ttft_s_mean"],
                "vllm_kv_cache_layout": resolved_kv_cache_layout,
                "warmup_rounds": warmup_rounds,
            },
            sort_keys=True,
        )
    )


def _run_generation(
    script: Path,
    *,
    model: str,
    port: int,
    http_port: int,
    prompt: str,
    max_model_len: int,
    max_tokens: int,
    batch_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    kv_cache_layout: str,
    raw_cuda_ipc: bool,
    use_layerwise: bool,
    steady_state_rounds: int,
    steady_state_warmup_rounds: int,
    worker_trace_output: Path | None,
    worker_label: str,
    kvcache_checksum: bool,
    timeout_s: int,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(script),
        "--worker",
        "--model",
        model,
        "--port",
        str(port),
        "--http-port",
        str(http_port),
        "--prompt",
        prompt,
        "--max-model-len",
        str(max_model_len),
        "--max-tokens",
        str(max_tokens),
        "--batch-size",
        str(batch_size),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--kv-cache-layout",
        kv_cache_layout,
        "--steady-state-rounds",
        str(steady_state_rounds),
        "--steady-state-warmup-rounds",
        str(steady_state_warmup_rounds),
        "--worker-label",
        worker_label,
    ]
    if worker_trace_output is not None:
        cmd.extend(["--worker-trace-output", str(worker_trace_output)])
    if raw_cuda_ipc:
        cmd.append("--raw-cuda-ipc")
    if use_layerwise:
        cmd.append("--use-layerwise")
    if kvcache_checksum:
        cmd.append("--kvcache-checksum")
    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if kv_cache_layout:
        env["VLLM_KV_CACHE_LAYOUT"] = kv_cache_layout
    if use_layerwise:
        env["LMCACHE_USE_LAYERWISE"] = "1"
    if worker_trace_output is not None:
        env["LMCACHE_MP_TRACE_FILE"] = str(worker_trace_output)
        env["LMCACHE_MP_TRACE_LABEL"] = worker_label
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "vLLM smoke worker failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX) :])
    raise RuntimeError(
        "vLLM smoke worker did not emit result line\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _read_trace_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    rows.sort(key=_trace_row_time_s)
    return rows


def _kvcache_checksum_trace_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("phase") == "submit"
        and row.get("request_type") in {"STORE", "RETRIEVE"}
    ]


def _trace_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    return value


def _trace_int_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"{field_name} must be a list of ints")
    return [int(item) for item in value]


def _trace_logical_block_size(rows: list[dict[str, object]]) -> int:
    for row in rows:
        if (
            row.get("phase") != "submit"
            or row.get("request_type") != "REGISTER_KV_CACHE"
        ):
            continue
        payloads = row.get("payloads")
        if not isinstance(payloads, list) or len(payloads) < 6:
            continue
        hints = payloads[5]
        if not isinstance(hints, dict):
            continue
        value = hints.get("inference_engine_logical_block_size")
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("trace did not include inference_engine_logical_block_size")


def _trace_chunk_token_size(rows: list[dict[str, object]]) -> int:
    for row in rows:
        if (
            row.get("phase") != "response"
            or row.get("request_type") != "GET_CHUNK_SIZE"
        ):
            continue
        response = row.get("response")
        if isinstance(response, int) and response > 0:
            return response
    raise RuntimeError("trace did not include GET_CHUNK_SIZE response")


def _trace_checksum_chunk_blocks(rows: list[dict[str, object]]) -> int:
    chunk_token_size = _trace_chunk_token_size(rows)
    logical_block_size = _trace_logical_block_size(rows)
    if chunk_token_size % logical_block_size != 0:
        raise RuntimeError(
            "chunk token size is not divisible by logical block size: "
            f"{chunk_token_size} vs {logical_block_size}"
        )
    return chunk_token_size // logical_block_size


def _query_native_kvcache_checksum(
    *,
    http_port: int,
    instance_id: int,
    block_ids: list[int],
    chunk_blocks: int,
) -> dict[str, object]:
    query = urllib.parse.urlencode(
        {
            "instance_id": instance_id,
            "block_ids": ",".join(str(block_id) for block_id in block_ids),
            "chunk_size": chunk_blocks,
        }
    )
    url = f"http://127.0.0.1:{http_port}/kvcache/check?{query}"
    try:
        return json.loads(urllib.request.urlopen(url, timeout=30).read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"native /kvcache/check failed with HTTP {exc.code}: {body}"
        ) from exc


def _worker_kvcache_checksum_probe(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if not args.kvcache_checksum:
        return None
    if args.http_port <= 0:
        raise RuntimeError("--http-port is required for --kvcache-checksum")
    if args.worker_trace_output is None:
        raise RuntimeError("--worker-trace-output is required for --kvcache-checksum")
    rows = _read_trace_rows([args.worker_trace_output])
    checksum_rows = _kvcache_checksum_trace_rows(rows)
    if not checksum_rows:
        return None
    row = checksum_rows[-1]
    payloads = row.get("payloads")
    if not isinstance(payloads, list) or len(payloads) < 3:
        raise RuntimeError("STORE/RETRIEVE trace row does not include payloads")
    instance_id = _trace_int(payloads[1], "instance_id")
    block_ids = _trace_int_list(payloads[2], "block_ids")
    chunk_blocks = _trace_checksum_chunk_blocks(rows)
    response = _query_native_kvcache_checksum(
        http_port=args.http_port,
        instance_id=instance_id,
        block_ids=block_ids,
        chunk_blocks=chunk_blocks,
    )
    return {
        "request_type": row["request_type"],
        "instance_id": instance_id,
        "block_ids": block_ids,
        "chunk_blocks": chunk_blocks,
        "response": response,
    }


def _dict_field(values: dict[str, object], key: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict")
    return value


def _assert_kvcache_checksum_match(
    writer_generation: dict[str, object],
    reader_generations: list[dict[str, object]],
) -> None:
    writer_probe = _dict_field(writer_generation, "kvcache_checksum")
    writer_response = _dict_field(writer_probe, "response")
    if writer_probe.get("request_type") != "STORE":
        raise RuntimeError(f"writer checksum probe was not STORE: {writer_probe}")
    for reader_index, reader_generation in enumerate(reader_generations):
        reader_probe = _dict_field(reader_generation, "kvcache_checksum")
        reader_response = _dict_field(reader_probe, "response")
        if reader_probe.get("request_type") != "RETRIEVE":
            raise RuntimeError(
                f"reader {reader_index} checksum probe was not RETRIEVE: {reader_probe}"
            )
        for key in ("chunk_size", "num_chunks", "block_id_ranges"):
            if writer_response.get(key) != reader_response.get(key):
                raise RuntimeError(
                    f"reader {reader_index} checksum metadata differs for {key}: "
                    f"writer={writer_response}, reader={reader_response}"
                )
        if writer_response.get("chunk_checksums") != reader_response.get(
            "chunk_checksums"
        ):
            raise RuntimeError(
                f"reader {reader_index} checksum mismatch: "
                f"writer={writer_response}, reader={reader_response}"
            )


def _kvcache_checksum_match_trace_row(
    args: argparse.Namespace,
    writer_generation: dict[str, object],
    reader_generations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "kind": "vllm_kvcache_checksum_match",
        "model": args.model,
        "raw_cuda_ipc": args.raw_cuda_ipc,
        "use_layerwise": args.use_layerwise,
        "requested_kv_cache_layout": args.kv_cache_layout or None,
        "writer": _dict_field(writer_generation, "kvcache_checksum"),
        "readers": [
            _dict_field(reader_generation, "kvcache_checksum")
            for reader_generation in reader_generations
        ],
    }


def _trace_row_time_s(row: dict[str, object]) -> float:
    value = row["time_s"]
    if not isinstance(value, int | float):
        raise TypeError("trace row time_s must be numeric")
    return float(value)


def _trace_request_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("phase") != "submit":
            continue
        request_type = str(row["request_type"])
        counts[request_type] = counts.get(request_type, 0) + 1
    return counts


def _trace_request_key(row: dict[str, object]) -> tuple[str, int, int]:
    worker_label = row.get("worker_label")
    pid = row.get("pid")
    request_uid = row.get("request_uid")
    if not isinstance(worker_label, str):
        raise TypeError("trace row worker_label must be a string")
    if not isinstance(pid, int):
        raise TypeError("trace row pid must be an int")
    if not isinstance(request_uid, int):
        raise TypeError("trace row request_uid must be an int")
    return (worker_label, pid, request_uid)


def _trace_request_latency_summary(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float | int | None | list[float]]]:
    submits: dict[tuple[str, int, int], dict[str, object]] = {}
    latencies_by_request: dict[str, list[float]] = {}
    for row in rows:
        phase = row.get("phase")
        if phase not in {"submit", "response"}:
            continue
        request_type = row.get("request_type")
        if not isinstance(request_type, str):
            raise TypeError("trace row request_type must be a string")
        request_time_s = _trace_row_time_s(row)
        key = _trace_request_key(row)
        if phase == "submit":
            submits[key] = row
            continue
        submit_row = submits.get(key)
        if submit_row is None:
            continue
        submit_request_type = submit_row.get("request_type")
        if submit_request_type != request_type:
            raise RuntimeError(
                "trace request type changed between submit and response: "
                f"{submit_request_type!r} vs {request_type!r}"
            )
        submit_time_s = _trace_row_time_s(submit_row)
        latencies_by_request.setdefault(request_type, []).append(
            (request_time_s - submit_time_s) * 1000.0
        )
    return {
        request_type: _latency_stats_ms(latencies)
        for request_type, latencies in sorted(latencies_by_request.items())
    }


def _trace_register_layout_hints(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    hints = []
    for row in rows:
        if (
            row.get("phase") != "submit"
            or row.get("request_type") != "REGISTER_KV_CACHE"
        ):
            continue
        payloads = row.get("payloads")
        if not isinstance(payloads, list) or len(payloads) < 6:
            continue
        layout_hints = payloads[5]
        if isinstance(layout_hints, dict):
            hints.append(layout_hints)
    return hints


def _summarize_mp_trace(rows: list[dict[str, object]]) -> dict[str, object]:
    submit_rows = [row for row in rows if row.get("phase") == "submit"]
    response_rows = [row for row in rows if row.get("phase") == "response"]
    return {
        "row_count": len(rows),
        "submit_count": len(submit_rows),
        "response_count": len(response_rows),
        "request_counts": _trace_request_counts(rows),
        "request_latency_ms": _trace_request_latency_summary(rows),
        "request_latency_scope": (
            "client-observed MP request round trip from MessageQueueClient "
            "submit to response inside the vLLM worker process"
        ),
        "submit_sequence": [row["request_type"] for row in submit_rows],
        "register_layout_hints": _trace_register_layout_hints(rows),
    }


def _assert_mp_trace_lifecycle(
    rows: list[dict[str, object]],
    *,
    use_layerwise: bool,
    kv_cache_layout: str,
) -> None:
    counts = _trace_request_counts(rows)
    required = (
        "REGISTER_KV_CACHE",
        "STORE",
        "LOOKUP",
        "QUERY_PREFETCH_STATUS",
        "RETRIEVE",
    )
    missing = [
        request_type for request_type in required if counts.get(request_type, 0) < 1
    ]
    if missing:
        raise RuntimeError(
            "vLLM MP trace is missing required request types: " + ", ".join(missing)
        )

    layout_hints = _trace_register_layout_hints(rows)
    if not layout_hints:
        raise RuntimeError("vLLM MP trace did not capture REGISTER_KV_CACHE hints")
    if use_layerwise and not any(
        hints.get("use_layerwise") is True for hints in layout_hints
    ):
        raise RuntimeError("vLLM MP trace did not record use_layerwise=True")
    if kv_cache_layout and not any(
        hints.get("kv_layout") == kv_cache_layout for hints in layout_hints
    ):
        raise RuntimeError(
            f"vLLM MP trace did not record kv_layout={kv_cache_layout!r}"
        )


def run_smoke(args: argparse.Namespace, server: str | None = None) -> dict[str, object]:
    server = server or args.server
    if args.reader_processes < 1:
        raise ValueError("--reader-processes must be at least 1")
    if args.require_kvcache_checksum_match and server != "native":
        raise ValueError("--require-kvcache-checksum-match requires native server")
    capture_worker_trace = (
        args.compare_python
        or args.mp_trace_output
        or args.require_mp_trace_lifecycle
        or args.require_kvcache_checksum_match
    )
    zmq_port = _free_port()
    http_port = _free_port()
    prompt = args.prompt or " ".join(
        ["LMCache native MP validation"] * args.prompt_repetitions
    )
    mp_trace_rows: list[dict[str, object]] = []
    checksum_trace_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as td:
        worker_trace_paths: list[Path] = []
        trace_dir = Path(td) / "mp_trace"
        proc = None
        server_stderr = None
        server_resources_start = None
        server_resources_end = None
        after_first_status = None
        after_reader_statuses = []
        proc = subprocess.Popen(
            _server_command(
                server,
                zmq_port=zmq_port,
                http_port=http_port,
                l1_size_gb=args.l1_size_gb,
                chunk_size=args.chunk_size,
                disk_path=Path(td) / "disk",
                native_gpu_hot_cache=args.native_gpu_hot_cache,
            ),
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_server(server, zmq_port, http_port)
            server_resources_start = _read_process_resources(proc.pid)
            script = Path(__file__).resolve()
            first = _run_generation(
                script,
                model=args.model,
                port=zmq_port,
                http_port=http_port,
                prompt=prompt,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_layout=args.kv_cache_layout,
                raw_cuda_ipc=args.raw_cuda_ipc,
                use_layerwise=args.use_layerwise,
                steady_state_rounds=args.steady_state_rounds,
                steady_state_warmup_rounds=args.steady_state_warmup_rounds,
                worker_trace_output=trace_dir / "writer-0.jsonl"
                if capture_worker_trace
                else None,
                worker_label="writer-0",
                kvcache_checksum=args.require_kvcache_checksum_match,
                timeout_s=args.worker_timeout_s,
            )
            if capture_worker_trace:
                worker_trace_paths.append(trace_dir / "writer-0.jsonl")
            if server == "native":
                after_first_status = _status(http_port)
                after_first = _summary(after_first_status)
            else:
                after_first = None

            def run_reader_generation(reader_index: int) -> dict[str, object]:
                trace_output = (
                    trace_dir / f"reader-{reader_index}.jsonl"
                    if capture_worker_trace
                    else None
                )
                return _run_generation(
                    script,
                    model=args.model,
                    port=zmq_port,
                    http_port=http_port,
                    prompt=prompt,
                    max_model_len=args.max_model_len,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                    dtype=args.dtype,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    kv_cache_layout=args.kv_cache_layout,
                    raw_cuda_ipc=args.raw_cuda_ipc,
                    use_layerwise=args.use_layerwise,
                    steady_state_rounds=args.steady_state_rounds,
                    steady_state_warmup_rounds=args.steady_state_warmup_rounds,
                    worker_trace_output=trace_output,
                    worker_label=f"reader-{reader_index}",
                    kvcache_checksum=args.require_kvcache_checksum_match,
                    timeout_s=args.worker_timeout_s,
                )

            reader_generations = []
            after_readers = []
            if args.concurrent_readers and args.reader_processes > 1:
                with ThreadPoolExecutor(max_workers=args.reader_processes) as executor:
                    futures = [
                        executor.submit(run_reader_generation, reader_index)
                        for reader_index in range(args.reader_processes)
                    ]
                    for future in as_completed(futures):
                        reader_generations.append(future.result())
                        if server == "native":
                            status = _status(http_port)
                            after_reader_statuses.append(status)
                            after_readers.append(_summary(status))
            else:
                for reader_index in range(args.reader_processes):
                    reader_generations.append(run_reader_generation(reader_index))
                    if server == "native":
                        status = _status(http_port)
                        after_reader_statuses.append(status)
                        after_readers.append(_summary(status))
            if capture_worker_trace:
                worker_trace_paths.extend(
                    trace_dir / f"reader-{reader_index}.jsonl"
                    for reader_index in range(args.reader_processes)
                )
            second = reader_generations[0]
            after_second = after_readers[0] if after_readers else None
            after_last_reader = after_readers[-1] if after_readers else None
            server_resources_end = _read_process_resources(proc.pid)
            mp_trace_rows = _read_trace_rows(worker_trace_paths)
            if args.require_mp_trace_lifecycle:
                _assert_mp_trace_lifecycle(
                    mp_trace_rows,
                    use_layerwise=args.use_layerwise,
                    kv_cache_layout=args.kv_cache_layout,
                )
            if args.require_kvcache_checksum_match:
                _assert_kvcache_checksum_match(first, reader_generations)
                checksum_trace_rows.append(
                    _kvcache_checksum_match_trace_row(
                        args,
                        first,
                        reader_generations,
                    )
                )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            if proc.stderr is not None:
                server_stderr = proc.stderr.read()

    mp_trace_summary = _summarize_mp_trace(mp_trace_rows) if mp_trace_rows else None
    if args.mp_trace_output is not None:
        args.mp_trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output_rows = [*mp_trace_rows, *checksum_trace_rows]
        args.mp_trace_output.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in trace_output_rows
            ),
            encoding="utf-8",
        )

    if (
        server == "native"
        and args.require_retrieve
        and isinstance(after_first, dict)
        and isinstance(after_last_reader, dict)
        and _int_field(after_last_reader, "retrieve_count")
        <= _int_field(after_first, "retrieve_count")
    ):
        raise RuntimeError(
            "reader vLLM generations did not increase native RETRIEVE count: "
            f"after_first={after_first}, after_last_reader={after_last_reader}"
        )
    unexpected_stderr = _unexpected_native_stderr_lines(server_stderr or "")
    if server == "native" and args.require_clean_native_stderr and unexpected_stderr:
        raise RuntimeError(
            "native MP server emitted unexpected stderr lines: "
            + json.dumps(unexpected_stderr)
        )
    return {
        "server": server,
        "model": args.model,
        "raw_cuda_ipc": args.raw_cuda_ipc,
        "native_gpu_hot_cache": (
            args.native_gpu_hot_cache if server == "native" else False
        ),
        "use_layerwise": args.use_layerwise,
        "prompt_token_probe": (
            "shared prompt batch used for one writer process followed by reader "
            "vLLM processes; prompt variants share the configured base prompt"
        ),
        "batch_size": args.batch_size,
        "prompt_repetitions": args.prompt_repetitions,
        "requested_kv_cache_layout": args.kv_cache_layout or None,
        "concurrent_readers": args.concurrent_readers,
        "reader_processes": args.reader_processes,
        "first_generation": first,
        "second_generation": second,
        "reader_generations": reader_generations,
        "after_first": after_first,
        "after_second": after_second,
        "after_readers": after_readers,
        "native_status_snapshots": {
            "after_first": after_first_status,
            "after_readers": after_reader_statuses,
        }
        if server == "native"
        else None,
        "server_resources_start": server_resources_start,
        "server_resources_end": server_resources_end,
        "server_resources_delta": _process_resource_delta(
            server_resources_start,
            server_resources_end,
        ),
        "mp_trace_summary": mp_trace_summary,
        "clean_native_stderr": not unexpected_stderr,
        "unexpected_native_stderr_lines": unexpected_stderr,
        "server_stderr_tail": (server_stderr or "")[-2000:],
    }


def run_compare(args: argparse.Namespace) -> dict[str, object]:
    python_report = run_smoke(args, "python")
    native_report = run_smoke(args, "native")
    python_second = python_report["second_generation"]
    native_second = native_report["second_generation"]
    if not isinstance(python_second, dict) or not isinstance(native_second, dict):
        raise TypeError("generation report must be a dict")
    python_elapsed = python_second["elapsed_s"]
    native_elapsed = native_second["elapsed_s"]
    python_generate_elapsed = python_second["generate_elapsed_s"]
    native_generate_elapsed = native_second["generate_elapsed_s"]
    python_steady_state = python_second["steady_state"]
    native_steady_state = native_second["steady_state"]
    if not isinstance(python_elapsed, int | float) or not isinstance(
        native_elapsed,
        int | float,
    ):
        raise TypeError("elapsed_s must be numeric")
    if not isinstance(python_generate_elapsed, int | float) or not isinstance(
        native_generate_elapsed,
        int | float,
    ):
        raise TypeError("generate_elapsed_s must be numeric")
    if not isinstance(python_steady_state, dict) or not isinstance(
        native_steady_state,
        dict,
    ):
        raise TypeError("steady_state report must be a dict")
    python_throughput = python_steady_state["output_tokens_per_s_mean"]
    native_throughput = native_steady_state["output_tokens_per_s_mean"]
    if not isinstance(python_throughput, int | float) or not isinstance(
        native_throughput,
        int | float,
    ):
        raise TypeError("steady-state output_tokens_per_s_mean must be numeric")
    python_resources = python_report.get("server_resources_delta")
    native_resources = native_report.get("server_resources_delta")
    python_trace = python_report.get("mp_trace_summary")
    native_trace = native_report.get("mp_trace_summary")
    python_request_latency = (
        python_trace.get("request_latency_ms")
        if isinstance(python_trace, dict)
        else None
    )
    native_request_latency = (
        native_trace.get("request_latency_ms")
        if isinstance(native_trace, dict)
        else None
    )
    return {
        "scope": (
            "vLLM reuse benchmark; elapsed_s includes vLLM process startup and "
            "model load, generate_elapsed_s excludes model init, and steady_state "
            "reports warmup-controlled measured generate rounds inside one loaded "
            "LLM process. server_resources_delta samples the MP server process "
            "from /proc before the writer run and after the final reader run"
        ),
        "python": python_report,
        "native": native_report,
        "second_generation_generate_elapsed_s": {
            "python": float(python_generate_elapsed),
            "native": float(native_generate_elapsed),
            "native_over_python": float(native_generate_elapsed)
            / float(python_generate_elapsed),
        },
        "second_generation_elapsed_s": {
            "python": float(python_elapsed),
            "native": float(native_elapsed),
            "native_over_python": float(native_elapsed) / float(python_elapsed),
        },
        "steady_state_output_tokens_per_s": {
            "python": float(python_throughput),
            "native": float(native_throughput),
            "native_over_python": float(native_throughput) / float(python_throughput),
        },
        "steady_state_ttft_s_mean": {
            "python": python_steady_state["ttft_s_mean"],
            "native": native_steady_state["ttft_s_mean"],
        },
        "server_resources_delta": {
            "python": python_resources,
            "native": native_resources,
        },
        "mp_request_latency_ms": {
            "python": python_request_latency,
            "native": native_request_latency,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--server", choices=["native", "python"], default="native")
    parser.add_argument(
        "--compare-python",
        action="store_true",
        help="Run the same smoke against Python MP and native MP.",
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--http-port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-trace-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-label", default="worker", help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-repetitions", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--reader-processes",
        type=int,
        default=1,
        help="Number of reuse-generation vLLM processes to run after the writer.",
    )
    parser.add_argument(
        "--concurrent-readers",
        action="store_true",
        help="Run reuse-generation reader processes concurrently.",
    )
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--steady-state-rounds", type=int, default=1)
    parser.add_argument("--steady-state-warmup-rounds", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--kv-cache-layout",
        choices=["", "NHD", "HND"],
        default="",
        help=(
            "Set VLLM_KV_CACHE_LAYOUT for the vLLM worker. Empty keeps vLLM's "
            "default layout."
        ),
    )
    parser.add_argument("--l1-size-gb", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--raw-cuda-ipc", action="store_true")
    parser.add_argument(
        "--native-gpu-hot-cache",
        action="store_true",
        help=(
            "Pass --cuda-gpu-hot-cache to the native server. Ignored for "
            "Python MP."
        ),
    )
    parser.add_argument(
        "--use-layerwise",
        action="store_true",
        help="Set LMCACHE_USE_LAYERWISE=1 in each vLLM worker process.",
    )
    parser.add_argument("--worker-timeout-s", type=int, default=180)
    parser.add_argument(
        "--no-require-retrieve", dest="require_retrieve", action="store_false"
    )
    parser.set_defaults(require_retrieve=True)
    parser.add_argument(
        "--require-clean-native-stderr",
        action="store_true",
        help=(
            "Fail native smoke runs if the native server writes anything other "
            "than its startup line to stderr."
        ),
    )
    parser.add_argument(
        "--mp-trace-output",
        type=Path,
        default=None,
        help=(
            "Write metadata-only JSONL rows for real vLLM MP requests emitted "
            "by the smoke worker processes."
        ),
    )
    parser.add_argument(
        "--require-mp-trace-lifecycle",
        action="store_true",
        help=(
            "Fail if the metadata trace does not include the expected real "
            "vLLM REGISTER/STORE/LOOKUP/RETRIEVE lifecycle."
        ),
    )
    parser.add_argument(
        "--kvcache-checksum",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-kvcache-checksum-match",
        action="store_true",
        help=(
            "Query native /kvcache/check inside each vLLM worker and fail if "
            "reader RETRIEVE checksums differ from writer STORE checksums."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steady_state_rounds < 1:
        raise ValueError("--steady-state-rounds must be at least 1")
    if args.steady_state_warmup_rounds < 0:
        raise ValueError("--steady-state-warmup-rounds must be non-negative")
    if args.compare_python and args.require_kvcache_checksum_match:
        raise ValueError(
            "--require-kvcache-checksum-match cannot be used with --compare-python"
        )

    if args.worker:
        _run_worker(args)
        return

    report = run_compare(args) if args.compare_python else run_smoke(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

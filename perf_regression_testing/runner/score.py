# SPDX-License-Identifier: Apache-2.0
"""Score one run: the same metric set for every workload.

Inputs: ``aiperf/profile_export.jsonl`` and ``samples.jsonl``. The window is
``scoring.window_s`` from the first profiling request's start; a request (or
error) counts if it ends inside. Decode rate is ``(output tokens - 1) /
(latency - TTFT)``. Server counters are interpolated at the window edges.
"""

# Standard
from pathlib import Path
from typing import Any
import json

_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile.

    Args:
        xs: Values, any order.
        q: Quantile in [0, 1].

    Returns:
        The percentile, or None for an empty list.
    """
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Read AIPerf per-request records into flat dicts (seconds, tokens).

    Args:
        path: ``profile_export.jsonl``.

    Returns:
        One dict per record: phase, start, end, ttft, lat, osl, rate, err.
    """
    out = []
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            m, me = x["metadata"], x.get("metrics", {})

            def val(k: str, me: dict[str, Any] = me) -> Any:
                v = me.get(k)
                return v.get("value") if isinstance(v, dict) else None

            ttft, lat = val("time_to_first_token"), val("request_latency")
            ttft = ttft / 1e3 if ttft is not None else None
            lat = lat / 1e3 if lat is not None else None
            osl = val("usage_completion_tokens") or val("output_sequence_length") or 0
            rate = None
            if ttft is not None and lat is not None and osl > 1 and lat > ttft:
                rate = (osl - 1) / (lat - ttft)
            out.append(
                {
                    "phase": m.get("benchmark_phase"),
                    "start": m["request_start_ns"] / 1e9,
                    "end": m["request_end_ns"] / 1e9,
                    "ttft": ttft,
                    "lat": lat,
                    "osl": osl,
                    "rate": rate,
                    "err": bool(x.get("error")) or m.get("was_cancelled", False),
                }
            )
    return out


def window_bounds(
    reqs: list[dict[str, Any]], window_s: float
) -> tuple[float, float, list[dict[str, Any]]]:
    """Compute the scored window and the requests that fall in it.

    Args:
        reqs: Output of :func:`load_requests`.
        window_s: Window length from the first profiling request's start.

    Returns:
        (start, end, profiling requests that end in the window, errored ones
        included).
    """
    pop = [r for r in reqs if r["phase"] == "profiling"]
    ws = min(r["start"] for r in pop)
    we = ws + window_s
    return ws, we, [r for r in pop if ws <= r["end"] <= we]


def _at(samples: list[dict[str, Any]], t: float, key: str) -> float:
    prev = None
    for s in samples:
        if key not in s:
            continue
        if s["t"] >= t:
            if prev is None:
                return s[key]
            f = (t - prev["t"]) / (s["t"] - prev["t"])
            return prev[key] + f * (s[key] - prev[key])
        prev = s
    if prev is None:
        raise KeyError(f"{key} never sampled")
    return prev[key]


def score(run_dir: Path, window_s: float) -> dict[str, Any]:
    """Compute the run's metrics.

    Args:
        run_dir: Run dir holding ``aiperf/`` and ``samples.jsonl``.
        window_s: The workload's ``scoring.window_s``.

    Returns:
        Metrics, keyed by name.

    Raises:
        KeyError: a server counter was never sampled.
        ValueError: there are no profiling requests.
    """
    reqs = load_requests(run_dir / "aiperf" / "profile_export.jsonl")
    ws, we, win = window_bounds(reqs, window_s)
    ok = [r for r in win if not r["err"] and r["ttft"] is not None]
    ttft = [r["ttft"] for r in ok]
    rates = [r["rate"] for r in ok if r["rate"] is not None]
    with open(run_dir / "samples.jsonl") as f:
        samples = [json.loads(line) for line in f]

    def delta(key: str) -> float:
        return _at(samples, we, key) - _at(samples, ws, key)

    src = {k: delta(f"vllm:prompt_tokens_by_source/{k}") for k in _SOURCES}
    mem = [s for s in samples if ws <= s["t"] <= we and "mem_anon" in s]
    total = sum(src.values())
    return {
        "window_start": ws,
        "window_s": we - ws,
        "requests": len(ok),
        "errors": sum(r["err"] for r in win),
        "ttft_p50": pct(ttft, 0.5),
        "ttft_p95": pct(ttft, 0.95),
        "decode_p10": pct(rates, 0.1),
        "decode_p50": pct(rates, 0.5),
        "prompt_tokens_by_source": src,
        # Served by the cache layer (L1 and L2), the same meter everywhere.
        "external_token_share": src["external_kv_transfer"] / total if total else None,
        # Token throughput over the window, from the server's counters: every
        # prompt token whatever its source, and every generated token.
        "input_tok_s": total / (we - ws),
        "output_tok_s": delta("vllm:generation_tokens_total") / (we - ws),
        "preemptions": delta("vllm:num_preemptions_total"),
        # Peak over the window of the vLLM container's own memory and of the
        # page cache charged to it (a [deployments] vllm_memory_gb caps both).
        "vllm_mem_peak_gb": {
            k: max(s[f"mem_{k}"] for s in mem) / 2**30 for k in ("anon", "file")
        }
        if mem
        else None,
    }

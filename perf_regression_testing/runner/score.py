# SPDX-License-Identifier: Apache-2.0
"""Score one run: the same metric set for every workload.

Inputs: ``aiperf/profile_export.jsonl``, ``samples.jsonl`` and the prompt ids
the AIPerf endpoint logs in :data:`PROMPT_DIR`. The window is
``scoring.window_s`` from the first profiling request's start; a request (or
error) counts if it ends inside. Decode rate is ``(output tokens - 1) /
(latency - TTFT)``. Server counters are interpolated at the window edges.

Disk write rates: ``warmup`` from the first I/O sample (AIPerf's start) to the
window start, ``steady`` over the window, which follows the warmup.

The ideal cache hit share is what an unbounded token-granular prefix cache
would serve: each prompt, in start order from the server's first request, hits
its longest common prefix with any earlier prompt, at most all but its last
token, which a server always computes. Only prompts that start in the window
count.
"""

# Standard
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any
import hashlib
import json
import os

_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
# Under the run dir: one ``<x_request_id>.i32`` of prompt ids per request.
PROMPT_DIR = Path("aiperf", "prt_live_history", "prompts")
# Tokens per hashed block in the prefix search; the result is exact either way.
_BLOCK = 64
_ITEM = array("i").itemsize


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
        One dict per record: id, phase, start, end, ttft, lat, osl, rate, err.
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
                    "id": m.get("x_request_id"),
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


def ideal_hit_share(
    prompt_dir: Path, reqs: list[dict[str, Any]], ws: float, we: float
) -> float | None:
    """Share of the window's prompt tokens an unbounded prefix cache would serve.

    Full ``_BLOCK``-token blocks are matched by chained hash; the first
    unmatched block is compared token by token with every block that followed
    the same prefix in an earlier prompt.

    Args:
        prompt_dir: :data:`PROMPT_DIR` of the run.
        reqs: Output of :func:`load_requests`, every phase.
        ws: Window start.
        we: Window end.

    Returns:
        Hit tokens over prompt tokens of the requests whose first token lands
        in ``[ws, we]``, or None when none of them has a prompt file. vLLM
        counts a prompt by source when its prefill ends, so this is the
        population ``cache_hit_share`` covers: one still in prefill at ``we``
        is in neither. A request without a prompt file never reached the
        server's prefill and is left out.
    """
    width = _ITEM * _BLOCK
    # Chained hash of a full-block prefix -> the blocks (or tails) after it.
    after: defaultdict[bytes, set[bytes]] = defaultdict(set)
    hit = total = 0
    for r in sorted(reqs, key=lambda r: r["start"]):
        path = prompt_dir / f"{r['id']}.i32"
        if not path.exists():
            continue
        data = path.read_bytes()
        blocks = [data[i : i + width] for i in range(0, len(data), width)]
        common, h = 0, b""
        for block in blocks:
            nexts = after.get(h, set())
            if block not in nexts:
                common += max(
                    (len(os.path.commonprefix([block, c])) for c in nexts), default=0
                )
                break
            common += len(block)
            h = hashlib.blake2b(h + block, digest_size=8).digest()
        h = b""
        for block in blocks:
            after[h].add(block)
            h = hashlib.blake2b(h + block, digest_size=8).digest()
        if r["ttft"] is not None and ws <= r["start"] + r["ttft"] <= we:
            n = len(data) // _ITEM
            hit += min(common // _ITEM, n - 1)
            total += n
    return hit / total if total else None


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
    drafts = (
        delta("vllm:spec_decode_num_drafts_total")
        if any("vllm:spec_decode_num_drafts_total" in s for s in samples)
        else 0
    )
    disk: dict[str, Any] = {}
    io = [s for s in samples if "io_wbytes" in s]
    if io:
        t0 = io[0]["t"]
        disk["disk_write_gb_s_warmup"] = (
            (_at(samples, ws, "io_wbytes") - io[0]["io_wbytes"]) / 1e9 / (ws - t0)
        )
        disk["disk_write_gb_s_steady"] = delta("io_wbytes") / 1e9 / (we - ws)
        # A mean over the window hides a rate still climbing as L1 fills.
        edges = [ws + (we - ws) * i / 4 for i in range(5)]
        disk["disk_write_gb_s_steady_quarters"] = [
            (_at(samples, b, "io_wbytes") - _at(samples, a, "io_wbytes"))
            / 1e9
            / (b - a)
            for a, b in zip(edges, edges[1:], strict=False)
        ]
    return disk | {
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
        # Every prompt token not computed: prefix cache plus cache layer.
        "cache_hit_share": 1 - src["local_compute"] / total if total else None,
        "ideal_hit_share": ideal_hit_share(run_dir / PROMPT_DIR, reqs, ws, we),
        # Token throughput over the window, from the server's counters: every
        # prompt token whatever its source, and every generated token.
        "input_tok_s": total / (we - ws),
        "output_tok_s": delta("vllm:generation_tokens_total") / (we - ws),
        "preemptions": delta("vllm:num_preemptions_total"),
        # Tokens per verification step: the target's token plus accepted drafts.
        "spec_accept_len": 1
        + delta("vllm:spec_decode_num_accepted_tokens_total") / drafts
        if drafts
        else None,
        # Peak over the window of the vLLM container's own memory and of the
        # page cache charged to it (a [deployments] vllm_memory_gb caps both).
        "vllm_mem_peak_gb": {
            k: max(s[f"mem_{k}"] for s in mem) / 2**30 for k in ("anon", "file")
        }
        if mem
        else None,
    }

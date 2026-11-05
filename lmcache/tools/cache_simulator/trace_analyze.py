# SPDX-License-Identifier: Apache-2.0
"""
Offline cache analysis of request traces exported by the MP server.

With ``--enable-token-trace`` the MP server emits one ``mp.request.trace``
span per lookup carrying the prompt token ids (``lmcache.input_ids``), the
model name (``lmcache.model``) and the request's ``session_id``; the span's
start time is the lookup timestamp. An OpenTelemetry Collector ``file``
exporter persists those spans as OTLP JSON, one resource batch per line.

This module replays such a trace through a simulated cache for a grid of
``(chunk_size, capacity)`` pairs and reports, per cell:

* **prefix hit tokens** -- the leading run of chained chunk hashes present
  in an LRU cache, exactly as the prefix lookup counts them;
* **non-prefix hit tokens** -- tokens outside that prefix that CacheBlend
  would retrieve: the real :class:`BlendTokenRangeMatcher` is probed with
  every lookup and its matches are reconciled the way the blend lookup
  does (prefix coverage first, then non-overlapping segments outside it).

Only matched tokens are reported; what CacheBlend recomputes on top of a
match is not modelled. Capacity is a token budget, since the trace carries
no KV shape.
"""

# Standard
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import html
import json
import logging
import struct

# First Party
from lmcache.cli.metrics import Metrics
from lmcache.tools.cache_simulator.lru_cache import LRUCacheFast
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.modules.blend.matcher import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.token_hasher import TokenHasher

#: Name of the span the tracing subscriber emits per traced request.
TRACE_SPAN_NAME = "mp.request.trace"

#: Instrumentation scope of the Dynamo frontend's request-trace ``otel`` sink.
DYNAMO_TRACE_SCOPE = "dynamo.request_trace"

_SIZE_SUFFIXES: dict[str, int] = {"k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}
_DEFAULT_CAPACITY_START = 4096


@dataclass(frozen=True)
class TraceRecord:
    """One traced request: all the analyzer needs from any trace format.

    A format is whatever a producer and its collector pipeline emit (OTLP
    spans, OTLP logs, flat JSON lines); its adapter in :data:`TRACE_FORMATS`
    turns each input line into these records and nothing else about the
    input matters. In order of importance:

    Attributes:
        input_ids: Prompt token ids as the engine prefilled them, chat
            template included. The replay works on nothing else.
        timestamp_ns: Request time in nanoseconds; fixes the replay order.
        request_id: Collapses a request recorded more than once to its
            earliest record. The LMCache server emits one span per lookup,
            and the engine looks a request up again after preemption, so
            that format needs it; a producer that records each request once
            can use any unique value.
        model: Model name, only for the ``--model`` filter; may be empty.
    """

    timestamp_ns: int
    model: str
    input_ids: list[int]
    request_id: str


@dataclass
class ReplayResult:
    """Outcome of replaying a trace through one ``(chunk_size, capacity)``.

    Token totals include the trailing partial chunk of every request; it
    is never cached, so it always counts as a miss.

    Attributes:
        chunk_size: Tokens per cached chunk.
        capacity_tokens: Cache capacity in tokens.
        capacity_chunks: Cache capacity in chunks (``capacity_tokens //
            chunk_size``, at least one).
        total_requests: Requests replayed.
        total_tokens: Sum of request lengths.
        prefix_hit_tokens: Tokens covered by the leading run of cached
            chunks, summed over requests.
        non_prefix_hit_tokens: Tokens CacheBlend would retrieve outside
            the prefix, summed over requests.
        eviction_count: LRU evictions during the replay.
        per_request_prefix_hit_tokens: Prefix hit tokens per request, in
            replay order.
        per_request_non_prefix_hit_tokens: Non-prefix hit tokens per
            request, in replay order.
    """

    chunk_size: int
    capacity_tokens: int
    capacity_chunks: int
    total_requests: int
    total_tokens: int
    prefix_hit_tokens: int
    non_prefix_hit_tokens: int
    eviction_count: int
    per_request_prefix_hit_tokens: list[int]
    per_request_non_prefix_hit_tokens: list[int]

    @property
    def prefix_hit_rate(self) -> float:
        """``prefix_hit_tokens / total_tokens`` (0 for an empty trace)."""
        return self.prefix_hit_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def non_prefix_hit_rate(self) -> float:
        """``non_prefix_hit_tokens / total_tokens`` (0 for an empty trace)."""
        if not self.total_tokens:
            return 0.0
        return self.non_prefix_hit_tokens / self.total_tokens

    def summary(self) -> dict[str, Any]:
        """Aggregate fields as a JSON-serializable dict (no per-request lists)."""
        return {
            "chunk_size": self.chunk_size,
            "capacity_tokens": self.capacity_tokens,
            "capacity_chunks": self.capacity_chunks,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefix_hit_rate": self.prefix_hit_rate,
            "non_prefix_hit_tokens": self.non_prefix_hit_tokens,
            "non_prefix_hit_rate": self.non_prefix_hit_rate,
            "requests_with_non_prefix_hits": sum(
                1 for t in self.per_request_non_prefix_hit_tokens if t > 0
            ),
            "eviction_count": self.eviction_count,
        }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def parse_size(text: str) -> int:
    """Parse a token count such as ``4096``, ``64k`` or ``1M``.

    Suffixes ``K``, ``M``, ``G`` and ``T`` are binary multiples; an
    optional trailing ``i``, ``iB`` or ``B`` is ignored.

    Args:
        text: The size to parse.

    Returns:
        The size as an integer.

    Raises:
        argparse.ArgumentTypeError: If *text* is not a size.
    """
    s = text.strip().lower()
    for tail in ("ib", "b", "i"):
        if s.endswith(tail) and len(s) > len(tail):
            s = s[: -len(tail)]
            break
    multiplier = 1
    if s and s[-1] in _SIZE_SUFFIXES:
        multiplier = _SIZE_SUFFIXES[s[-1]]
        s = s[:-1]
    try:
        value = float(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a size: {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {text!r}")
    return int(value * multiplier)


def _attribute_value(value: dict[str, Any]) -> Any:
    """Decode one OTLP JSON ``AnyValue``.

    OTLP's JSON mapping encodes 64-bit integers as decimal strings; both
    the string and the plain-int spelling are accepted.
    """
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_attribute_value(v) for v in value["arrayValue"]["values"]]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    raise ValueError(f"unsupported OTLP attribute value: {value!r}")


def _iter_spans(batch: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield every span in one OTLP JSON ``ExportTraceServiceRequest``."""
    for resource_spans in batch.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            yield from scope_spans.get("spans", [])


def _lmcache_records(line: dict[str, Any]) -> Iterable[TraceRecord]:
    """Records in one line of the ``lmcache`` trace format.

    A line is an OTLP JSON export batch; the format is the MP server's
    ``mp.request.trace`` span with ``lmcache.model``, ``lmcache.input_ids``
    and ``session_id``. Other spans in the batch are skipped.
    ``lmcache.input_ids`` is the OTLP integer array as emitted, or one JSON
    string of the same integers, which is how a collector ``transform``
    may carry it compactly; both decode to the same ids.
    """
    for span in _iter_spans(line):
        if span.get("name") != TRACE_SPAN_NAME:
            continue
        attrs = {a["key"]: _attribute_value(a["value"]) for a in span["attributes"]}
        ids = attrs["lmcache.input_ids"]
        if isinstance(ids, str):
            ids = json.loads(ids)
        yield TraceRecord(
            timestamp_ns=int(span["startTimeUnixNano"]),
            model=attrs["lmcache.model"],
            input_ids=[int(t) for t in ids],
            request_id=str(attrs["session_id"]),
        )


def _chat_text(request: dict[str, Any]) -> str:
    """A chat request's text in prompt order: tools, then each message.

    Chat templates render the tools ahead of the conversation. Each message
    adds its role and its text (the ``text`` parts of a list content).
    """
    parts: list[str] = []
    if request.get("tools"):
        parts.append(json.dumps(request["tools"], ensure_ascii=False, sort_keys=True))
    for message in request["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        parts.append(f"{message['role']}\n{content or ''}")
    return "\n".join(parts)


def _dynamo_records(line: dict[str, Any]) -> Iterable[TraceRecord]:
    """Records in one line of the ``dynamo`` trace format.

    A line is an OTLP JSON logs batch from the Dynamo frontend's request
    trace sink. A ``request_payload`` record holds the chat request as text,
    so its ids are that text packed four UTF-8 bytes per integer (about one
    per token) and results are an estimate. Other records, and the marker
    Dynamo sends without the request when a record is over its size limit,
    are skipped.
    """
    for resource_logs in line.get("resourceLogs", []):
        for scope_logs in resource_logs.get("scopeLogs", []):
            if scope_logs.get("scope", {}).get("name") != DYNAMO_TRACE_SCOPE:
                continue
            for log in scope_logs.get("logRecords", []):
                attrs = {
                    a["key"]: _attribute_value(a["value"]) for a in log["attributes"]
                }
                if attrs["event_type"] != "request_payload":
                    continue
                request = json.loads(attrs["payload"])["payload"].get("request")
                if request is None:
                    continue
                data = _chat_text(request).encode("utf-8")
                data += b"\0" * (-len(data) % 4)
                yield TraceRecord(
                    timestamp_ns=int(log["timeUnixNano"]),
                    model=attrs["model"],
                    input_ids=list(struct.unpack(f">{len(data) // 4}I", data)),
                    request_id=attrs["rid"],
                )


@dataclass(frozen=True)
class TraceFormat:
    """A producer format: how to read it and what it expects to find.

    Attributes:
        read: Yields the :class:`TraceRecord` entries in one parsed input
            line (none if the line holds no record of this format).
        expects: What the input must contain, for the error when it does
            not.
    """

    read: Callable[[dict[str, Any]], Iterable[TraceRecord]]
    expects: str


#: Producer formats ``--trace-format`` accepts.
TRACE_FORMATS: dict[str, TraceFormat] = {
    "lmcache": TraceFormat(
        read=_lmcache_records,
        expects=(
            f"{TRACE_SPAN_NAME} spans with lmcache.input_ids, which the MP "
            "server emits under --enable-token-trace and the collector's file "
            "exporter writes"
        ),
    ),
    "dynamo": TraceFormat(
        read=_dynamo_records,
        expects=(
            f"{DYNAMO_TRACE_SCOPE} request_payload log records, which the Dynamo "
            "frontend's request trace otel sink sends and the collector's file "
            "exporter writes"
        ),
    ),
}


def load_trace_records(
    paths: Sequence[Path],
    trace_format: str = "lmcache",
    model: str | None = None,
) -> list[TraceRecord]:
    """Read request records from JSON-lines files in a producer format.

    Each non-empty line is parsed as JSON and handed to the format's
    adapter. A request recorded more than once (for example an engine
    lookup repeated after preemption) keeps only its earliest record, so
    every request counts once.

    Args:
        paths: Files to read, in any order.
        trace_format: Producer format of the input, a key of
            :data:`TRACE_FORMATS`.
        model: When given, keep only requests for this model name.

    Returns:
        One record per request id, sorted by timestamp (ties keep file
        order).

    Raises:
        ValueError: If no line yielded a record. The message says what the
            files did hold (for OTLP span batches, the span names), so a
            wrong file, a server started without the token trace, or a
            collector pipeline that renamed the span can be told apart.
    """
    fmt = TRACE_FORMATS[trace_format]
    records: list[TraceRecord] = []
    lines = 0
    span_names: Counter[str] = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                lines += 1
                line = json.loads(raw)
                span_names.update(s.get("name", "") for s in _iter_spans(line))
                records.extend(
                    r for r in fmt.read(line) if model is None or r.model == model
                )
    if not records:
        files = ", ".join(str(p) for p in paths)
        held = f"{lines} line(s)"
        if span_names:
            names = ", ".join(f"{n} ({c})" for n, c in span_names.most_common(8))
            held += f" with {sum(span_names.values())} OTLP span(s) named {names}"
        raise ValueError(
            f"No {trace_format!r} trace records in {files}: read {held}. "
            f"Expected {fmt.expects}; see the trace-analyze docs."
        )
    records.sort(key=lambda r: r.timestamp_ns)
    first: dict[str, TraceRecord] = {}
    for r in records:
        first.setdefault(r.request_id, r)
    return list(first.values())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def select_non_prefix_segments(
    matches: Iterable[CBMatchResult], prefix_tokens: int
) -> list[CBMatchResult]:
    """Reconcile fingerprint matches against the prefix coverage.

    Mirrors the blend lookup: matches starting inside the prefix are
    already served by the prefix leg and are dropped first; of the rest a
    non-overlapping subset is kept, the earlier ``cur_st`` winning on
    overlap. Filtering before de-duplicating matters: a prefix-covered
    match could otherwise shadow a usable one and then be dropped anyway.

    Args:
        matches: Raw matcher results, in any order.
        prefix_tokens: Tokens covered by the prefix hit.

    Returns:
        Kept matches in ascending ``cur_st`` order.
    """
    kept: list[CBMatchResult] = []
    covered_end = -1
    for r in sorted(
        (r for r in matches if r.cur_st >= prefix_tokens), key=lambda r: r.cur_st
    ):
        if r.cur_st >= covered_end:
            kept.append(r)
            covered_end = r.cur_ed
    return kept


def hash_records(
    records: Sequence[TraceRecord], hasher: TokenHasher
) -> list[list[bytes]]:
    """Chained chunk hashes for every record (trailing partial chunk dropped).

    Args:
        records: Requests to hash.
        hasher: Hasher fixing chunk size and hash algorithm.

    Returns:
        One hash list per record, in the same order.
    """
    return [hasher.compute_chunk_hashes(r.input_ids) for r in records]


def replay(
    records: Sequence[TraceRecord],
    hashes: Sequence[list[bytes]],
    chunk_size: int,
    capacity_tokens: int,
) -> ReplayResult:
    """Replay a trace through one LRU cache plus the CacheBlend matcher.

    For each request, in order: the leading run of cached chunk hashes is
    the prefix hit; the matcher is probed with the full prompt and its
    matches are reconciled with :func:`select_non_prefix_segments`; the
    request's chunks are then registered with the matcher and inserted in
    the LRU (prefix hits are touched, not re-inserted), and every chunk the
    LRU evicts is removed from the matcher so later probes cannot match it.

    Args:
        records: Requests in replay order.
        hashes: Chunk hashes per request, as from :func:`hash_records`.
        chunk_size: Tokens per chunk the hashes were computed with.
        capacity_tokens: Cache capacity in tokens.

    Returns:
        Aggregate and per-request hit accounting.
    """
    capacity_chunks = max(1, capacity_tokens // chunk_size)
    cache = LRUCacheFast(capacity_chunks)
    matcher = BlendTokenRangeMatcher(chunk_size)

    total_tokens = 0
    prefix_total = 0
    non_prefix_total = 0
    per_prefix: list[int] = []
    per_non_prefix: list[int] = []

    for record, chunk_hashes in zip(records, hashes, strict=True):
        hit_prefix = 0
        for h in chunk_hashes:
            if not cache.contains(h):
                break
            hit_prefix += 1
        prefix_tokens = hit_prefix * chunk_size

        non_prefix_tokens = 0
        if chunk_hashes:
            segments = select_non_prefix_segments(
                matcher.match_sub_sequence(record.input_ids), prefix_tokens
            )
            # Segments are non-overlapping, so their lengths sum to the
            # token coverage.
            non_prefix_tokens = sum(s.cur_ed - s.cur_st for s in segments)

            matcher.on_new_token_hashes(record.input_ids, chunk_hashes)
            for i, h in enumerate(chunk_hashes):
                if i < hit_prefix:
                    cache.access(h)
                    continue
                evicted = cache.insert(h)
                if evicted is not None:
                    matcher.remove_chunks([evicted])

        total_tokens += len(record.input_ids)
        prefix_total += prefix_tokens
        non_prefix_total += non_prefix_tokens
        per_prefix.append(prefix_tokens)
        per_non_prefix.append(non_prefix_tokens)

    return ReplayResult(
        chunk_size=chunk_size,
        capacity_tokens=capacity_tokens,
        capacity_chunks=capacity_chunks,
        total_requests=len(records),
        total_tokens=total_tokens,
        prefix_hit_tokens=prefix_total,
        non_prefix_hit_tokens=non_prefix_total,
        eviction_count=cache.eviction_count,
        per_request_prefix_hit_tokens=per_prefix,
        per_request_non_prefix_hit_tokens=per_non_prefix,
    )


def analyze(
    records: Sequence[TraceRecord],
    chunk_sizes: Sequence[int],
    capacities_tokens: Sequence[int],
    hash_algorithm: str = "blake3",
) -> list[ReplayResult]:
    """Replay *records* for every ``(chunk_size, capacity)`` in the grid.

    Hashes are computed once per chunk size and shared by its capacities.

    Args:
        records: Requests sorted by timestamp.
        chunk_sizes: Chunk sizes to simulate.
        capacities_tokens: Cache capacities in tokens to simulate.
        hash_algorithm: Hash algorithm name for :class:`TokenHasher`.

    Returns:
        One result per grid cell, chunk sizes outer, capacities inner.
    """
    results: list[ReplayResult] = []
    for chunk_size in chunk_sizes:
        hasher = TokenHasher(chunk_size, hash_algorithm)
        hashes = hash_records(records, hasher)
        for capacity in capacities_tokens:
            results.append(replay(records, hashes, chunk_size, capacity))
    return results


def default_capacities(total_tokens: int) -> list[int]:
    """Half-octave steps from 4096 tokens up to the first covering the trace.

    Each step multiplies by the square root of two, rounded to a multiple
    of 1024 tokens, so the slider has two points per doubling.

    Args:
        total_tokens: Tokens in the trace; the last capacity is at least
            this large, so the sweep ends at a cache that never evicts.

    Returns:
        Ascending capacities in tokens, never empty.
    """
    capacities = [_DEFAULT_CAPACITY_START]
    step = 1
    while capacities[-1] < total_tokens:
        exact = _DEFAULT_CAPACITY_START * 2 ** (step / 2)
        capacities.append(round(exact / 1024) * 1024)
        step += 1
    return capacities


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LMCache trace analysis</title>
<style>
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5f6368; --line: #e0e0e0;
  --tile: #f6f7f9; --accent: #2f6fed;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #121417; --fg: #e8eaed; --muted: #9aa0a6; --line: #2c3036;
    --tile: #1b1f24; --accent: #8ab4f8;
  }
}
body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.controls { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  margin: 0 0 20px; }
@media (max-width: 600px) { .controls { grid-template-columns: 1fr; } }
label { display: block; font-weight: 600; margin-bottom: 4px; }
label span { font-weight: 400; color: var(--muted); float: right; }
input[type=range] { width: 100%; accent-color: var(--accent); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin: 0 0 24px; }
.tile { background: var(--tile); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 14px; }
.tile .k { color: var(--muted); font-size: 13px; }
.tile .v { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--muted); font-size: 13px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; font-size: 13px; }
tr.sel td { background: var(--tile); font-weight: 600; }
</style>
</head>
<body>
<main>
<h1>LMCache trace analysis</h1>
<p class="sub" id="meta">__META__</p>
<div class="controls">
  <div><label for="cap">Cache capacity <span id="cap-v">__CAP_V__</span></label>
    <input type="range" id="cap" min="0" max="__CAP_MAX__" value="__CAP_MAX__"
      step="1"></div>
  <div><label for="cs">Chunk size <span id="cs-v">__CS_V__</span></label>
    <input type="range" id="cs" min="0" max="__CS_MAX__" value="0" step="1"></div>
</div>
<div class="tiles">
  <div class="tile"><div class="k">Prefix hit rate</div>
    <div class="v" id="t-phr">__T_PHR__</div>
    <div class="d" id="t-pht">__T_PHT__</div></div>
  <div class="tile"><div class="k">Non-prefix (CacheBlend) hit rate</div>
    <div class="v" id="t-npr">__T_NPR__</div>
    <div class="d" id="t-npt">__T_NPT__</div></div>
  <div class="tile"><div class="k">Requests</div>
    <div class="v" id="t-req">__T_REQ__</div>
    <div class="d" id="t-reqnp">__T_REQNP__</div></div>
  <div class="tile"><div class="k">Total tokens</div>
    <div class="v" id="t-tok">__T_TOK__</div>
    <div class="d" id="t-ev">__T_EV__</div></div>
</div>
<table>
  <thead><tr><th>Capacity</th><th>Chunks</th><th>Prefix hit rate</th>
    <th>Prefix hit tokens</th><th>Non-prefix hit rate</th>
    <th>Non-prefix hit tokens</th><th>Evictions</th>
  </tr></thead>
  <tbody id="rows">__ROWS__</tbody>
</table>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  var data = JSON.parse(document.getElementById("data").textContent);
  var caps = data.capacities_tokens, sizes = data.chunk_sizes;
  var $ = function (id) { return document.getElementById(id); };
  var fmt = function (n) { return n.toLocaleString("en-US"); };
  var pct = function (x) { return (100 * x).toFixed(2) + "%"; };
  var capLabel = function (t) { return fmt(t) + " tokens"; };
  var cell = function (cs, cap) { return data.cells[cs + ":" + cap]; };
  function render() {
    var cap = caps[+$("cap").value], cs = sizes[+$("cs").value], c = cell(cs, cap);
    $("cap-v").textContent = capLabel(cap);
    $("cs-v").textContent = fmt(cs) + " tokens";
    $("t-phr").textContent = pct(c.prefix_hit_rate);
    $("t-pht").textContent =
      fmt(c.prefix_hit_tokens) + " of " + fmt(c.total_tokens) + " tokens";
    $("t-npr").textContent = pct(c.non_prefix_hit_rate);
    $("t-npt").textContent =
      fmt(c.non_prefix_hit_tokens) + " of " + fmt(c.total_tokens) + " tokens";
    $("t-req").textContent = fmt(c.total_requests);
    $("t-reqnp").textContent =
      fmt(c.requests_with_non_prefix_hits) + " with non-prefix hits";
    $("t-tok").textContent = fmt(c.total_tokens);
    $("t-ev").textContent =
      fmt(c.eviction_count) + " evictions, " + fmt(c.capacity_chunks) + " chunk slots";
    var rows = "";
    caps.forEach(function (k) {
      var r = cell(cs, k);
      var cells = [capLabel(k), fmt(r.capacity_chunks), pct(r.prefix_hit_rate),
        fmt(r.prefix_hit_tokens), pct(r.non_prefix_hit_rate),
        fmt(r.non_prefix_hit_tokens), fmt(r.eviction_count)];
      rows += "<tr" + (k === cap ? ' class="sel"' : "") + "><td>" +
        cells.join("</td><td>") + "</td></tr>";
    });
    $("rows").innerHTML = rows;
  }
  $("cap").addEventListener("input", render);
  $("cs").addEventListener("input", render);
})();
</script>
</body>
</html>
"""


def render_html(results: Sequence[ReplayResult], meta: str) -> str:
    """Render the self-contained HTML report.

    Two sliders (token budget, chunk size) select one precomputed grid
    cell and show its prefix and non-prefix hit rates, request count and
    token totals, plus a table of every budget at the selected chunk size.
    The default cell (smallest chunk size, largest budget) is rendered
    into the markup, so the page reads even where scripts do not run.

    Args:
        results: Grid cells from :func:`analyze`.
        meta: One line describing the trace, shown under the title.

    Returns:
        The HTML document.
    """
    chunk_sizes = sorted({r.chunk_size for r in results})
    capacities = sorted({r.capacity_tokens for r in results})
    data = {
        "meta": meta,
        "chunk_sizes": chunk_sizes,
        "capacities_tokens": capacities,
        "cells": {f"{r.chunk_size}:{r.capacity_tokens}": r.summary() for r in results},
    }
    # A closing script tag inside the JSON would end the data block early.
    payload = json.dumps(data).replace("</", "<\\/")

    default_cs, default_cap = chunk_sizes[0], capacities[-1]
    cells = {(r.chunk_size, r.capacity_tokens): r for r in results}
    c = cells[(default_cs, default_cap)]
    rows = []
    for cap in capacities:
        r = cells[(default_cs, cap)]
        row_cells = [
            f"{cap:,} tokens",
            f"{r.capacity_chunks:,}",
            f"{r.prefix_hit_rate:.2%}",
            f"{r.prefix_hit_tokens:,}",
            f"{r.non_prefix_hit_rate:.2%}",
            f"{r.non_prefix_hit_tokens:,}",
            f"{r.eviction_count:,}",
        ]
        cls = ' class="sel"' if cap == default_cap else ""
        rows.append(f"<tr{cls}><td>" + "</td><td>".join(row_cells) + "</td></tr>")
    static = {
        "__DATA__": payload,
        "__META__": html.escape(meta),
        "__CAP_MAX__": str(len(capacities) - 1),
        "__CS_MAX__": str(len(chunk_sizes) - 1),
        "__CAP_V__": f"{default_cap:,} tokens",
        "__CS_V__": f"{default_cs:,} tokens",
        "__T_PHR__": f"{c.prefix_hit_rate:.2%}",
        "__T_PHT__": f"{c.prefix_hit_tokens:,} of {c.total_tokens:,} tokens",
        "__T_NPR__": f"{c.non_prefix_hit_rate:.2%}",
        "__T_NPT__": f"{c.non_prefix_hit_tokens:,} of {c.total_tokens:,} tokens",
        "__T_REQ__": f"{c.total_requests:,}",
        "__T_REQNP__": (
            f"{c.summary()['requests_with_non_prefix_hits']:,} with non-prefix hits"
        ),
        "__T_TOK__": f"{c.total_tokens:,}",
        "__T_EV__": (
            f"{c.eviction_count:,} evictions, {c.capacity_chunks:,} chunk slots"
        ),
        "__ROWS__": "".join(rows),
    }
    page = _HTML_TEMPLATE
    for key, value in static.items():
        page = page.replace(key, value)
    return page


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_trace_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``trace-analyze`` flags onto *parser*.

    Args:
        parser: The sub-parser to add flags to.
    """
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        type=Path,
        help="OTLP JSON file(s) written by the collector's file exporter",
    )
    parser.add_argument(
        "--html",
        default="trace_analysis.html",
        metavar="FILE",
        help="Where to write the HTML report (default: trace_analysis.html)",
    )
    parser.add_argument(
        "--chunk-size",
        nargs="+",
        type=int,
        default=[256],
        metavar="TOKENS",
        help="Chunk size(s) to simulate (default: 256)",
    )
    parser.add_argument(
        "--capacity",
        nargs="+",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help=(
            "Cache token budgets to simulate; K/M suffixes are binary (64k). "
            "Default: half-octave steps from 4096 tokens up to the trace size."
        ),
    )
    parser.add_argument(
        "--trace-format",
        choices=sorted(TRACE_FORMATS),
        default="lmcache",
        help=(
            "Producer format of the input spans (default: lmcache, the MP "
            "server's mp.request.trace span)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Only analyze requests for this model name",
    )


def run_trace_analyze(args: argparse.Namespace, metrics: Metrics) -> None:
    """Load the trace, replay the grid, write the report and the summary.

    Args:
        args: Parsed arguments registered by :func:`add_trace_analyze_arguments`.
        metrics: Receives the summary (sections ``trace`` and ``grid``).

    Raises:
        SystemExit: If the input holds no trace records for the format.
    """
    # Per-lookup matcher and per-hasher construction logs would drown the
    # summary; only problems are worth printing here.
    for name in (
        "lmcache.v1.multiprocess.modules.blend.matcher",
        "lmcache.v1.multiprocess.token_hasher",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        records = load_trace_records(
            args.input, trace_format=args.trace_format, model=args.model
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    total_tokens = sum(len(r.input_ids) for r in records)
    if args.capacity is None:
        capacities = default_capacities(total_tokens)
    else:
        capacities = list(args.capacity)
    capacities = sorted(set(capacities))
    chunk_sizes = sorted(set(args.chunk_size))

    results = analyze(records, chunk_sizes, capacities)

    models = sorted({r.model for r in records})
    meta = (
        f"{len(records):,} requests, {total_tokens:,} tokens, "
        f"model(s): {', '.join(models)}"
    )
    Path(args.html).write_text(render_html(results, meta), encoding="utf-8")

    trace = metrics.add_section("trace", "Trace")
    trace.add("input", "Input", ", ".join(str(p) for p in args.input))
    trace.add("requests", "Requests", len(records))
    trace.add("total_tokens", "Total tokens", total_tokens)
    trace.add("models", "Models", ", ".join(models))
    trace.add("html", "HTML report", str(args.html))

    grid = metrics.add_table(
        "grid",
        "Grid",
        ("chunk_size", "Chunk"),
        ("capacity_tokens", "Capacity (tokens)"),
        ("prefix_hit_rate", "Prefix hit rate"),
        ("prefix_hit_tokens", "Prefix hit tokens"),
        ("non_prefix_hit_rate", "Non-prefix hit rate"),
        ("non_prefix_hit_tokens", "Non-prefix hit tokens"),
        ("eviction_count", "Evictions"),
    )
    for r in results:
        grid.add_row(
            chunk_size=r.chunk_size,
            capacity_tokens=r.capacity_tokens,
            prefix_hit_rate=f"{r.prefix_hit_rate:.2%}",
            prefix_hit_tokens=r.prefix_hit_tokens,
            non_prefix_hit_rate=f"{r.non_prefix_hit_rate:.2%}",
            non_prefix_hit_tokens=r.non_prefix_hit_tokens,
            eviction_count=r.eviction_count,
        )
    metrics.emit()

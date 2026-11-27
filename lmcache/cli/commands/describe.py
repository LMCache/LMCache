# SPDX-License-Identifier: Apache-2.0
"""``lmcache describe`` — show detailed status of a running LMCache service.

Usage::

    lmcache describe kvcache --url http://localhost:8000
"""

# Standard
from http.client import IncompleteRead
from typing import TYPE_CHECKING, Iterable
import argparse
import math
import sys
import urllib.error
import urllib.request

# Third Party
from prometheus_client.parser import text_string_to_metric_families

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.http import (
    DEFAULT_URLS,
    CLIHTTPError,
)
from lmcache.cli.http import fetch_json as _fetch_json
from lmcache.cli.http import (
    normalize_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.metrics import Metrics

# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------


class DescribeError(Exception):
    """Raised when the describe command cannot fetch or parse status data."""


_RUNNING_REQUESTS = "vllm:num_requests_running"


# Lookup counters exported by the MP server's metrics subscriber
# (``lmcache/v1/mp_observability/subscribers/metrics/lookup.py``), named by
# Prometheus family as they appear on ``GET /metrics``.
LOOKUP_REQUESTED_TOKENS = "lmcache_mp_lookup_requested_tokens"
LOOKUP_HIT_TOKENS = "lmcache_mp_lookup_hit_tokens"
LOOKUP_HIT_L1_TOKENS = "lmcache_mp_lookup_hit_l1_tokens"
LOOKUP_HIT_L2_TOKENS = "lmcache_mp_lookup_hit_l2_tokens"
LOOKUP_PROMPT_TOKENS = "lmcache_mp_lookup_prompt_tokens"
LOOKUP_REQUEST_HIT_TOKENS = "lmcache_mp_lookup_request_hit_tokens"
HIT_COUNTER_NAMES: tuple[str, ...] = (
    LOOKUP_REQUESTED_TOKENS,
    LOOKUP_HIT_TOKENS,
    LOOKUP_HIT_L1_TOKENS,
    LOOKUP_HIT_L2_TOKENS,
    LOOKUP_PROMPT_TOKENS,
    LOOKUP_REQUEST_HIT_TOKENS,
)


def fetch_json(url: str, timeout: int = 10) -> dict:
    """GET *url* and return the parsed JSON body.

    Raises:
        DescribeError: On network/HTTP errors.
    """
    try:
        return _fetch_json(url, timeout=timeout)
    except CLIHTTPError as exc:
        raise DescribeError(str(exc)) from exc


def fetch_health(url: str, timeout: int = 10) -> bool:
    """Return whether *url* responds with HTTP 200.

    A lightweight liveness probe for endpoints (e.g. the vLLM ``/health``
    route) that return an empty body rather than JSON, so :func:`fetch_json`
    cannot be used.

    Args:
        url: Full health-check URL to GET.
        timeout: Socket timeout in seconds.

    Returns:
        ``True`` if the server responds with HTTP 200, ``False`` on any
        non-200 status or connection error.
    """
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url), timeout=timeout
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def fetch_running_requests(url: str, timeout: int = 10) -> int | None:
    """Return the number of in-flight requests from a vLLM ``/metrics`` page.

    Parses the Prometheus ``vllm:num_requests_running`` gauge, summing the
    value across all reported series (e.g. one per model). This is best
    effort: the metric is informational, so any failure to fetch or parse
    degrades to ``None`` (rendered as ``N/A``) rather than raising.

    Args:
        url: Full ``/metrics`` URL to GET.
        timeout: Socket timeout in seconds.

    Returns:
        The total running-request count, or ``None`` if the endpoint is
        unreachable or the metric is absent (e.g. metrics disabled or an
        unsupported engine version).
    """
    totals = fetch_metric_totals(url, [_RUNNING_REQUESTS], timeout=timeout)
    running = totals.get(_RUNNING_REQUESTS)
    return int(running) if running is not None else None


def fetch_metric_totals(
    url: str, names: Iterable[str], timeout: int = 10
) -> dict[str, float]:
    """Sum Prometheus metric families from a ``/metrics`` page.

    Each requested family is summed across all its label series (e.g. one
    per model or cache salt). Counters are named by family, without the
    ``_total`` suffix (``lmcache_mp_lookup_hit_tokens`` reads
    ``lmcache_mp_lookup_hit_tokens_total``); their ``_created`` timestamps
    are not counted. This is best effort, for informational rows.

    Args:
        url: Full ``/metrics`` URL to GET.
        names: Metric family names to read.
        timeout: Socket timeout in seconds.

    Returns:
        A mapping from each requested family found on the page to its sum.
        Families absent from the page or with a non-finite sum (``NaN`` or
        ``Inf``) are omitted, and an unreachable, truncated or unparsable
        page yields an empty mapping.
    """
    wanted = set(names)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url), timeout=timeout
        ) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, OSError, IncompleteRead, UnicodeDecodeError):
        return {}

    totals: dict[str, float] = {}
    try:
        for family in text_string_to_metric_families(text):
            if family.name not in wanted:
                continue
            # A counter's value is its ``_total`` sample; a gauge's sample
            # carries the family name itself.
            values = [
                sample.value
                for sample in family.samples
                if sample.name in (family.name, f"{family.name}_total")
            ]
            if values:
                totals[family.name] = totals.get(family.name, 0.0) + sum(values)
    except ValueError:
        # Malformed exposition text; treat the whole page as unavailable.
        return {}
    # The exposition format allows NaN and +/-Inf, which no count can hold.
    return {name: total for name, total in totals.items() if math.isfinite(total)}


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def fmt_l1_used(used_bytes: int, total_bytes: int) -> str:
    """Format L1 usage as GB with its share of capacity, e.g. ``0.33 (8.3%)``."""
    pct = used_bytes / total_bytes * 100 if total_bytes > 0 else 0.0
    return f"{used_bytes / 1024**3:.2f} ({pct:.1f}%)"


def fmt_hit_rate(hits: int | None, total: int | None) -> float | None:
    """Return *hits* / *total* as a percentage rounded to one decimal.

    Args:
        hits: Tokens that hit, or ``None`` if unknown.
        total: Tokens looked up, or ``None`` if unknown.

    Returns:
        The percentage, or ``None`` if either count is unknown or *total*
        is 0 (no lookups yet), so the row renders as ``N/A``.
    """
    if hits is None or not total:
        return None
    return round(hits / total * 100, 1)


def fmt_health(is_healthy: object) -> str | None:
    """Format a boolean health flag as ``'OK'`` / ``'UNHEALTHY'``."""
    if is_healthy is None:
        return None
    return "OK" if is_healthy else "UNHEALTHY"


def safe_get(data: dict, *keys, default=None):  # type: ignore[type-arg]
    """Walk nested dicts by *keys*, returning *default* on any miss."""
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


# -------------------------------------------------------------------
# KVCache describer
# -------------------------------------------------------------------


class KVCacheDescriber:
    """Builds the ``describe kvcache`` output from a ``/status`` response.

    Each ``add_*`` method populates one logical section. The orchestrating
    :meth:`describe` calls them in order and emits the result.  Adding a
    new section is a one-method change — no other code needs to know
    about it.
    """

    def __init__(
        self,
        metrics: "Metrics",
        data: dict,
        base_url: str,
        hit_totals: dict[str, float],
    ) -> None:
        """Bind the inputs for one ``describe kvcache`` report.

        Args:
            metrics: The output collector to populate.
            data: The server's ``/status`` response.
            base_url: The normalized server URL, shown in the overview.
            hit_totals: Lookup counter sums from the server's ``/metrics``, as
                returned by :func:`fetch_metric_totals` for
                :data:`HIT_COUNTER_NAMES`; missing names render as ``N/A``.
        """
        self.metrics = metrics
        self.data = data
        self.base_url = base_url
        self.hit_totals = hit_totals

    def describe(self) -> None:
        """Run all section builders and emit."""
        self.add_overview()
        self.add_l1_storage()
        self.add_cache_hits()
        self.add_models()
        self.add_l2_adapters()
        self.metrics.emit()

    # -- sections --------------------------------------------------------

    def add_overview(self) -> None:
        """Top-level engine overview."""
        self.metrics.add("health", "Health", fmt_health(self.data.get("is_healthy")))
        self.metrics.add("url", "URL", self.base_url)
        self.metrics.add("engine_type", "Engine type", self.data.get("engine_type"))
        self.metrics.add("chunk_size", "Chunk size", self.data.get("chunk_size"))

    def add_l1_storage(self) -> None:
        """L1 totals across all L1s, plus one section per L1 when there are several.

        Reads the per-tag ``l1_managers`` and ``l1_eviction_controllers`` maps,
        which ``/status`` sends for any number of L1s. The single-L1 keys
        (``l1_manager``, ``l1_eviction_controller``) are omitted on multi-L1
        servers, so they cannot describe every deployment.
        """
        managers = safe_get(self.data, "storage_manager", "l1_managers") or {}
        evictions = (
            safe_get(self.data, "storage_manager", "l1_eviction_controllers") or {}
        )
        capacity_gb: float | None = None
        used_gb: str | None = None
        objects: int | None = None
        if managers:
            total = sum(m["memory_total_bytes"] for m in managers.values())
            used = sum(m["memory_used_bytes"] for m in managers.values())
            capacity_gb = round(total / (1024**3), 2)
            used_gb = fmt_l1_used(used, total)
            objects = sum(m["total_object_count"] for m in managers.values())
        # Each L1 may set its own policy; list each distinct one once.
        policies = dict.fromkeys(e["eviction_policy"] for e in evictions.values())

        self.metrics.add("l1_capacity_gb", "L1 capacity (GB)", capacity_gb)
        self.metrics.add("l1_used_gb", "L1 used (GB)", used_gb)
        self.metrics.add(
            "eviction_policy", "Eviction policy", ", ".join(policies) or None
        )
        self.metrics.add("cached_objects", "Cached objects", objects)
        self.metrics.add(
            "active_sessions", "Active sessions", self.data.get("active_sessions")
        )

        if len(managers) > 1:
            for tag, manager in managers.items():
                self._add_l1_section(tag, manager, evictions[tag])

    def add_cache_hits(self) -> None:
        """Lookup hit counts and rates since the server started.

        Adds the ``cache_hits`` section from :attr:`hit_totals`. Token counts
        missing from ``/metrics`` are ``None`` (``N/A``); a rate is ``None``
        when either operand is missing or its denominator is 0.
        """
        requested = self._hit_tokens(LOOKUP_REQUESTED_TOKENS)
        hit = self._hit_tokens(LOOKUP_HIT_TOKENS)
        sec = self.metrics.add_section("cache_hits", "Cache hits (since server start)")
        sec.add("lookup_requested_tokens", "Lookup requested (tokens)", requested)
        sec.add("lookup_hit_tokens", "Lookup hits (tokens)", hit)
        sec.add("lookup_hit_rate", "Lookup hit rate (%)", fmt_hit_rate(hit, requested))
        sec.add(
            "l1_hit_tokens", "L1 hits (tokens)", self._hit_tokens(LOOKUP_HIT_L1_TOKENS)
        )
        sec.add(
            "l2_hit_tokens", "L2 hits (tokens)", self._hit_tokens(LOOKUP_HIT_L2_TOKENS)
        )
        sec.add(
            "request_hit_rate",
            "Request hit rate (%)",
            fmt_hit_rate(
                self._hit_tokens(LOOKUP_REQUEST_HIT_TOKENS),
                self._hit_tokens(LOOKUP_PROMPT_TOKENS),
            ),
        )

    def add_models(self) -> None:
        """Per-model KV cache layout sections.

        Each model gets one section with context-wide fields, followed by
        one ``kernel_groups`` list entry per kernel group carrying that
        group's identity and geometry.
        """
        gpu_meta = self.data.get("cache_context_meta", {})
        if not gpu_meta:
            return

        # Deduplicate by (model_name, world_size) — multiple GPU IDs
        # may share the same model.
        seen: dict[tuple[str, int], dict] = {}
        for gpu_id, meta in gpu_meta.items():
            key = (meta["model_name"], meta["world_size"])
            if key not in seen:
                seen[key] = {
                    "gpu_ids": [],
                    "layout": meta.get("kv_cache_layout"),
                }
            seen[key]["gpu_ids"].append(gpu_id)

        for idx, ((model_name, world_size), info) in enumerate(seen.items()):
            section_key = f"model_{idx}"
            self.metrics.add_list_section("models", section_key, f"Model: {model_name}")
            sec = self.metrics[section_key]
            sec.add("model", "Model", model_name)
            sec.add("world_size", "World size", world_size)
            sec.add("gpu_ids", "GPU IDs", ", ".join(info["gpu_ids"]))

            layout = info.get("layout")
            if not layout:
                continue
            for _key, _label in (
                ("num_layers", "Num layers"),
                ("num_blocks", "Num blocks"),
                ("cache_size_per_token", "Cache size per token (bytes)"),
            ):
                if _key in layout:
                    sec.add(_key, _label, layout[_key])

            self._add_kernel_groups(idx, model_name, layout.get("kernel_groups", []))

    def _add_kernel_groups(
        self, model_idx: int, model_name: str, kernel_groups: list
    ) -> None:
        """Emit one ``kernel_groups`` list section per kernel group.

        Args:
            model_idx: Index of the owning model section (keeps section keys
                unique across models).
            model_name: Human-readable model name, shown in each group header.
            kernel_groups: The model layout's ``kernel_groups`` list (each a
                dict produced by ``GPUCacheContext.report_status``).
        """
        for group in kernel_groups:
            kg_idx = group.get("kernel_group_idx")
            section_key = f"model_{model_idx}_kg_{kg_idx}"
            self.metrics.add_list_section(
                "kernel_groups",
                section_key,
                f"Kernel group {kg_idx} ({model_name})",
            )
            sec = self.metrics[section_key]
            sec.add("model", "Model", model_name)
            for _key, _label in (
                ("kernel_group_idx", "Kernel group index"),
                ("engine_group_idx", "Engine group index"),
                ("object_group_idx", "Object group index"),
                ("num_layers", "Num layers"),
                ("tokens_per_block", "Tokens per block"),
                ("slots_per_block", "Slots per block"),
                ("dtype", "Dtype"),
                ("is_mla", "MLA"),
                ("attention_backend", "Attention backend"),
                ("engine_kv_shape", "Engine KV shape"),
                ("engine_kv_concrete_shape", "Engine KV tensor shape"),
            ):
                if _key in group:
                    sec.add(_key, _label, group[_key])

    def add_l2_adapters(self) -> None:
        """L2 adapter sections."""
        l2_adapters = safe_get(self.data, "storage_manager", "l2_adapters") or []
        for idx, adapter in enumerate(l2_adapters):
            adapter_type = adapter.get("type", "Unknown")
            section_key = f"l2_{idx}"
            self.metrics.add_list_section(
                "l2_adapters", section_key, f"L2: {adapter_type}"
            )
            sec = self.metrics[section_key]
            sec.add("type", "Type", adapter_type)
            sec.add("health", "Health", fmt_health(adapter.get("is_healthy")))

            if "backend" in adapter:
                sec.add("backend", "Backend", adapter["backend"])
            if "base_path" in adapter:
                sec.add("base_path", "Base path", adapter["base_path"])
            if "stored_object_count" in adapter:
                sec.add(
                    "stored_object_count",
                    "Stored objects",
                    adapter["stored_object_count"],
                )

            cap = adapter.get("max_capacity_bytes")
            used = adapter.get("current_size_bytes")
            if cap is not None and used is not None:
                pct = used / cap * 100 if cap > 0 else 0.0
                sec.add(
                    "used",
                    "Used",
                    f"{fmt_bytes(used)} / {fmt_bytes(cap)} ({pct:.1f}%)",
                )

            pool_size = adapter.get("pool_size")
            pool_free = adapter.get("pool_free_slots")
            if pool_size is not None and pool_free is not None:
                pool_used = pool_size - pool_free
                pct = pool_used / pool_size * 100 if pool_size > 0 else 0.0
                sec.add(
                    "pool_used",
                    "Pool used",
                    f"{pool_used} / {pool_size} ({pct:.1f}%)",
                )

    def _add_l1_section(self, tag: str, manager: dict, eviction: dict) -> None:
        """Emit one ``l1_managers`` list section for the L1 tagged *tag*.

        Args:
            tag: The L1's configured tag.
            manager: That L1's entry in ``storage_manager.l1_managers``.
            eviction: That L1's entry in ``storage_manager.l1_eviction_controllers``.
        """
        self.metrics.add_list_section("l1_managers", f"l1_{tag}", f"L1: {tag}")
        sec = self.metrics[f"l1_{tag}"]
        sec.add("tag", "Tag", tag)
        sec.add("backend", "Backend", manager["backend"])
        sec.add(
            "capacity_gb",
            "Capacity (GB)",
            round(manager["memory_total_bytes"] / (1024**3), 2),
        )
        sec.add(
            "used_gb",
            "Used (GB)",
            fmt_l1_used(manager["memory_used_bytes"], manager["memory_total_bytes"]),
        )
        sec.add("cached_objects", "Cached objects", manager["total_object_count"])
        sec.add("eviction_policy", "Eviction policy", eviction["eviction_policy"])

    def _hit_tokens(self, name: str) -> int | None:
        """Return the summed counter *name* as a token count, or ``None``."""
        value = self.hit_totals.get(name)
        return int(value) if value is not None else None


# -------------------------------------------------------------------
# Engine describer
# -------------------------------------------------------------------


class EngineDescriber:
    """Builds the ``describe engine`` output from vLLM server responses.

    Reads model identity and context window from the engine's
    ``/v1/models`` response and combines them with a ``/health`` liveness
    result to render a concise status view.
    """

    def __init__(
        self,
        metrics: "Metrics",
        models_data: dict,
        is_healthy: bool,
        running_requests: int | None,
        base_url: str,
    ) -> None:
        self.metrics = metrics
        self.models_data = models_data
        self.is_healthy = is_healthy
        self.running_requests = running_requests
        self.base_url = base_url

    def describe(self) -> None:
        """Run all section builders and emit."""
        self.add_overview()
        self.metrics.emit()

    def add_overview(self) -> None:
        """Model identity, context window, health status, and load."""
        model = self._first_model()
        self.metrics.add("model", "Model", model.get("id") if model else None)
        self.metrics.add(
            "max_context",
            "Max context (tokens)",
            model.get("max_model_len") if model else None,
        )
        self.metrics.add("status", "Status", fmt_health(self.is_healthy))
        self.metrics.add("running_requests", "Running requests", self.running_requests)

    def _first_model(self) -> dict | None:
        """Return the first model entry from a ``/v1/models`` response."""
        data = self.models_data.get("data") or []
        return data[0] if data else None


# -------------------------------------------------------------------
# Command
# -------------------------------------------------------------------


class DescribeCommand(BaseCommand):
    """Show detailed status of a running LMCache service."""

    def name(self) -> str:
        return "describe"

    def help(self) -> str:
        return "Show detailed status of a running LMCache service."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            choices=["kvcache", "engine"],
            help="What to describe.",
        )
        parser.add_argument(
            "--url",
            default=None,
            help=(
                "Server URL (default: http://localhost:8080 for kvcache, "
                "http://localhost:8000 for engine)."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        if getattr(args, "url", None) is None:
            args.url = DEFAULT_URLS[args.target]
        if args.target == "kvcache":
            self._describe_kvcache(args)
        elif args.target == "engine":
            self._describe_engine(args)

    def _describe_kvcache(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        try:
            data = fetch_json(f"{base_url}/status")
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        hit_totals = fetch_metric_totals(f"{base_url}/metrics", HIT_COUNTER_NAMES)
        metrics = self.create_metrics("LMCache KV Cache Service", args, width=50)
        KVCacheDescriber(metrics, data, base_url, hit_totals).describe()

    def _describe_engine(self, args: argparse.Namespace) -> None:
        base_url = normalize_url(args.url)
        try:
            models = fetch_json(f"{base_url}/v1/models")
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        is_healthy = fetch_health(f"{base_url}/health")
        running_requests = fetch_running_requests(f"{base_url}/metrics")
        metrics = self.create_metrics("Inference Engine", args, width=50)
        EngineDescriber(
            metrics, models, is_healthy, running_requests, base_url
        ).describe()

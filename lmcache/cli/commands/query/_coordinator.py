# SPDX-License-Identifier: Apache-2.0
"""Read-only coordinator queries behind ``lmcache query coordinator``.

One entry per queryable coordinator endpoint. Each knows its path, which
extra arguments it needs, and how to shape the reply into a
:class:`~lmcache.cli.metrics.Metrics` report -- so the terminal gets an
aligned table and ``--format json`` gets the same data structurally.

Only reads live here. The coordinator's mutating routes (``POST /events``,
``POST /instances``, the ``/cache`` actions) are either server-to-coordinator
plumbing or belong to commands that own the action.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.describe import fmt_bytes
from lmcache.cli.metrics import Metrics
from lmcache.logging import init_logger

logger = init_logger(__name__)

_TIMEOUT = 10


def normalize_url(url: str) -> str:
    """Ensure *url* has a scheme and no trailing slash."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def _fetch(url: str, timeout: int = _TIMEOUT) -> str:
    """GET *url* and return the body as text.

    Args:
        url: Full URL to request.
        timeout: Seconds to wait.

    Returns:
        The response body.

    Raises:
        SystemExit: On a connection failure or a non-2xx status, reported as
            a CLI error rather than a traceback.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError, ValueError):
            pass
        logger.error("Coordinator returned %s: %s", e.code, detail)
        sys.exit(1)
    except TimeoutError as e:
        logger.error("Timed out contacting %s after %ds (%s)", url, timeout, e)
        sys.exit(1)
    except urllib.error.URLError as e:
        logger.error(
            "Cannot reach %s -- is the coordinator running? (%s)", url, e.reason
        )
        sys.exit(1)


def get_text(url: str) -> str:
    """GET *url* and return the body unparsed.

    For endpoints that are not JSON -- ``/metrics`` is Prometheus text.

    Args:
        url: Full URL to request.

    Returns:
        The response body.
    """
    return _fetch(url)


def get_json(url: str) -> Any:
    """GET *url* and parse the reply as JSON.

    Args:
        url: Full URL to request.

    Returns:
        The parsed body.

    Raises:
        SystemExit: If the reply is not JSON.
    """
    body = _fetch(url)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        logger.error("Coordinator reply was not JSON: %s", body[:200])
        sys.exit(1)


def _ratio(value: float | None) -> str:
    """Render a usage ratio, keeping ``null`` distinct from zero."""
    return "unknown" if value is None else f"{value * 100:.1f}%"


def _capacity(value: int) -> str:
    """Render a declared capacity, keeping "undeclared" distinct from zero."""
    return fmt_bytes(value) if value else "--"


# -- Renderers ---------------------------------------------------------------
# Each takes the parsed reply and fills a Metrics report.


def _render_usage(body: Any, metrics: Metrics) -> None:
    """Per-compartment occupancy, busiest first."""
    table = metrics.add_table(
        "usage",
        "Usage",
        ("instance", "instance"),
        ("compartment", "compartment"),
        ("used", "used"),
        ("capacity", "capacity"),
        ("ratio", "ratio"),
    )
    # A single-instance reply has no "instances" key; treat it as a fleet of one.
    instances = body.get("instances", [body]) if "instances" in body else [body]
    rows = [(i["instance_id"], m) for i in instances for m in i["modules"]]
    rows += [("(fleet-shared)", m) for m in body.get("shared_modules", [])]
    rows.sort(key=lambda row: -(row[1]["usage_ratio"] or 0))
    for name, module in rows:
        table.add_row(
            instance=name,
            compartment=f"{module['tier']}/{module['backend']}",
            used=fmt_bytes(module["used_bytes"]),
            capacity=_capacity(module["capacity_bytes"]),
            ratio=_ratio(module["usage_ratio"]),
        )


def _render_instances(body: Any, metrics: Metrics) -> None:
    """Fleet membership."""
    table = metrics.add_table(
        "instances",
        "Instances",
        ("instance", "instance"),
        ("address", "address"),
        ("mq_port", "mq port"),
        ("p2p", "p2p url"),
    )
    for instance in body.get("instances", []):
        table.add_row(
            instance=instance["instance_id"],
            address=f"{instance['ip']}:{instance['http_port']}",
            mq_port=instance.get("mq_port") or "--",
            p2p=instance.get("p2p_advertised_url") or "--",
        )


def _render_health(body: Any, metrics: Metrics) -> None:
    """Liveness."""
    metrics.add("status", "Status", body.get("status", "unknown"))


def _render_directory_stats(body: Any, metrics: Metrics) -> None:
    """Key-directory size, and how much of it is fragment-matchable."""
    metrics.add("num_keys", "Keys", body.get("num_keys"))
    metrics.add("num_placements", "Placements", body.get("num_placements"))
    blend = body.get("blend") or {}
    section = metrics.add_section("blend", "Blend index")
    for key, label in (
        ("num_contents", "Contents"),
        ("num_chunks", "Chunks"),
        ("num_claims", "Claims"),
        ("num_namespaces", "Namespaces"),
        ("table_size", "Table size"),
    ):
        section.add(key, label, blend.get(key))


def _render_keys(body: Any, metrics: Metrics) -> None:
    """A page of directory keys and where each one lives.

    The chunk hash is truncated: 64 hex characters per row would push the
    placements column off the terminal, and a 12-character prefix is enough
    to correlate with a log line.
    """
    metrics.add("total", "Matching keys", body.get("total", 0))
    table = metrics.add_table(
        "keys",
        "Keys",
        ("chunk", "chunk"),
        ("model", "model"),
        ("kv_rank", "rank"),
        ("cache_salt", "salt"),
        ("placements", "placements"),
    )
    for info in body.get("keys", []):
        key = info.get("key") or {}
        placements = ", ".join(
            f"{p.get('instance_id') or '(shared)'}:{p.get('tier')}/{p.get('backend')}"
            for p in info.get("placements", [])
        )
        table.add_row(
            chunk=str(key.get("chunk_hash_hex", ""))[:12],
            model=key.get("model_name") or "--",
            kv_rank=key.get("kv_rank"),
            cache_salt=key.get("cache_salt") or "(default)",
            placements=placements or "--",
        )


def _render_quota(body: Any, metrics: Metrics) -> None:
    """Per-salt usage against quota. Accepts the list or single-salt reply."""
    if "by_cache_salt" not in body:  # single-salt reply
        for key, label in (
            ("cache_salt", "Cache salt"),
            ("quota_limit_gb", "Quota (GiB)"),
            ("quota_exists", "Quota set"),
            ("usage_gb", "Usage (GiB)"),
        ):
            metrics.add(key, label, body.get(key))
        return
    metrics.add("total_gb", "Total usage (GiB)", body.get("total_gb"))
    table = metrics.add_table(
        "by_cache_salt",
        "By cache salt",
        ("cache_salt", "cache salt"),
        ("usage_gb", "usage GiB"),
        ("quota_limit_gb", "quota GiB"),
        ("quota_exists", "quota set"),
    )
    for row in body.get("by_cache_salt", []):
        table.add_row(
            cache_salt=row.get("cache_salt") or "(default)",
            usage_gb=f"{row.get('usage_gb', 0.0):.2f}",
            quota_limit_gb=f"{row.get('quota_limit_gb', 0.0):.2f}",
            quota_exists="yes" if row.get("quota_exists") else "no",
        )


def _render_quota_config(body: Any, metrics: Metrics) -> None:
    """The default quota applied to salts with no explicit entry."""
    limit = body.get("default_limit_gb")
    metrics.add(
        "default_limit_gb",
        "Default limit (GiB)",
        "none (exempt)" if limit is None else limit,
    )


def _render_prefetch(body: Any, metrics: Metrics) -> None:
    """One prefetch request's progress."""
    for key, value in body.items():
        metrics.add(key, key.replace("_", " ").capitalize(), value)


def _render_status(body: Any, metrics: Metrics) -> None:
    """The coordinator's debugging snapshot, one block per component.

    Tables print without their label, so each one's first column names
    what its rows are.
    """
    source = body["event_source"]
    section = metrics.add_section("event_source", "Event source")
    section.add("source_name", "Source", source["source_name"])
    section.add("replay_capability", "Replay", source["replay_capability"])
    # Only the Kafka source reports what it has read.
    reads = source.get("reads")
    if reads is not None:
        for key, label in (
            ("records_applied", "Records applied"),
            ("records_undecodable", "Records undecodable"),
            ("records_failed", "Records failed"),
            ("partitions_assigned", "Partitions assigned"),
        ):
            section.add(key, label, reads[key])
        table = metrics.add_table(
            "partitions",
            "Partitions",
            ("partition", "partition"),
            ("next_offset", "next offset"),
            ("unread_records", "unread"),
        )
        partitions = set(reads["next_offsets"]) | set(reads["unread_records"])
        for partition in sorted(partitions, key=int):
            table.add_row(
                partition=partition,
                next_offset=reads["next_offsets"].get(partition),
                unread_records=reads["unread_records"].get(partition),
            )

    ingest = body["ingest"]
    section = metrics.add_section("ingest", "Ingest")
    for key, label in (
        ("batches_applied", "Batches applied"),
        ("batches_duplicate", "Batches duplicate"),
        ("batches_stale", "Batches stale"),
        ("batches_missing", "Batches missing"),
        ("events_dropped", "Events dropped by servers"),
    ):
        section.add(key, label, ingest[key])
    table = metrics.add_table(
        "streams",
        "Streams",
        ("server", "server"),
        ("incarnation", "incarnation"),
        ("last_seq", "last seq"),
        ("view", "view"),
        ("missing_batches", "missing"),
        ("events_dropped", "dropped"),
        ("seconds_since_last_event", "since event s"),
        ("event_delivery_delay_seconds", "delivery delay s"),
    )
    for server, stream in sorted(ingest["streams"].items()):
        table.add_row(
            server=server,
            incarnation=stream["incarnation"],
            last_seq=stream["last_seq"],
            view="incomplete" if stream["gap_detected"] else "complete",
            missing_batches=stream["missing_batches"],
            events_dropped=stream["events_dropped"],
            seconds_since_last_event=stream["seconds_since_last_event"],
            event_delivery_delay_seconds=stream["event_delivery_delay_seconds"],
        )

    table = metrics.add_table(
        "consumers",
        "Consumers",
        ("consumer", "consumer"),
        ("batches_delivered", "delivered"),
        ("average_apply_ms", "avg apply ms"),
        ("consume_failures", "consume failures"),
        ("fence_failures", "fence failures"),
    )
    for name, consumer in sorted(body["consumers"].items()):
        delivered = consumer["batches_delivered"]
        table.add_row(
            consumer=name,
            batches_delivered=delivered,
            average_apply_ms=(
                consumer["apply_seconds"] / delivered * 1000 if delivered else None
            ),
            consume_failures=consumer["consume_failures"],
            fence_failures=consumer["fence_failures"],
        )

    table = metrics.add_table(
        "heartbeats",
        "Heartbeats",
        ("server", "registered server"),
        ("seconds_since_heartbeat", "since heartbeat s"),
    )
    for server, age in sorted(body["registry"]["seconds_since_heartbeat"].items()):
        table.add_row(server=server, seconds_since_heartbeat=age)

    table = metrics.add_table(
        "controllers",
        "Controllers",
        ("controller", "controller"),
        ("started", "started"),
    )
    for name, started in sorted(body["controllers"].items()):
        table.add_row(controller=name, started="yes" if started else "no")

    # The error is last: it is free text of any length, and as the final
    # column it cannot push the others out of line.
    table = metrics.add_table(
        "background_loops",
        "Background loops",
        ("loop", "loop"),
        ("interval_seconds", "every s"),
        ("passes_ok", "ok"),
        ("passes_failed", "failed"),
        ("seconds_since_last_success", "since success s"),
        ("failing", "failing"),
        ("last_error", "last error"),
    )
    for name, loop in sorted(body["background_loops"].items()):
        table.add_row(
            loop=name,
            interval_seconds=loop["interval_seconds"],
            passes_ok=loop["passes_ok"],
            passes_failed=loop["passes_failed"],
            seconds_since_last_success=loop["seconds_since_last_success"],
            failing="yes" if _still_failing(loop) else "no",
            last_error=loop["last_error"] or "--",
        )

    section = metrics.add_section("event_loop", "Event loop")
    section.add(
        "blocked_seconds", "Blocked (s, total)", body["event_loop"]["blocked_seconds"]
    )


@dataclass(frozen=True)
class CoordinatorApi:
    """One queryable coordinator endpoint.

    Attributes:
        summary: One-line description, shown in ``--api`` help.
        path: Callable building the request path from parsed arguments.
        render: Fills a report from the parsed reply.
        requires: Argument names this endpoint needs, e.g. ``("instance",)``.
        raw: True when the reply is not JSON and is printed verbatim
            (``/metrics`` is Prometheus text).
    """

    summary: str
    path: Callable[[Any], str]
    render: Callable[[Any, Metrics], None] = field(default=_render_health)
    requires: tuple[str, ...] = ()
    raw: bool = False


APIS: dict[str, CoordinatorApi] = {
    "usage": CoordinatorApi(
        summary="per-instance memory usage against declared capacity",
        path=lambda a: (
            f"/instances/{a.instance}/usage" if a.instance else "/instances/usage"
        ),
        render=_render_usage,
    ),
    "instances": CoordinatorApi(
        summary="registered MP servers",
        path=lambda a: "/instances",
        render=_render_instances,
    ),
    "health": CoordinatorApi(
        summary="coordinator liveness",
        path=lambda a: "/healthz",
        render=_render_health,
    ),
    "directory": CoordinatorApi(
        summary="key-directory size and blend-index stats",
        path=lambda a: "/directory/stats",
        render=_render_directory_stats,
    ),
    "keys": CoordinatorApi(
        summary="a page of directory keys and their placements",
        path=lambda a: f"/directory/keys?limit={a.limit}",
        render=_render_keys,
    ),
    "quota": CoordinatorApi(
        summary="per-salt usage against quota",
        path=lambda a: (
            f"/quota/{a.cache_salt}" if a.cache_salt is not None else "/quota"
        ),
        render=_render_quota,
    ),
    "quota-config": CoordinatorApi(
        summary="default quota for salts with no explicit entry",
        path=lambda a: "/quota/config",
        render=_render_quota_config,
    ),
    "prefetch": CoordinatorApi(
        summary="status of one prefetch request",
        path=lambda a: f"/cache/prefetches/{a.instance}/{a.request_id}",
        render=_render_prefetch,
        requires=("instance", "request_id"),
    ),
    "status": CoordinatorApi(
        summary="debugging snapshot: ingest, consumers, loops, controllers",
        path=lambda a: "/status",
        render=_render_status,
    ),
    "metrics": CoordinatorApi(
        summary="Prometheus metrics, verbatim",
        path=lambda a: "/metrics",
        raw=True,
    ),
}


def _still_failing(loop: dict[str, Any]) -> bool:
    """Whether a loop's latest pass failed.

    A later success leaves ``last_error`` in place, so the error alone does
    not say the loop is failing now; its time against the last success does.
    """
    failed_at = loop["last_error_timestamp"]
    succeeded_at = loop["last_success_timestamp"]
    return failed_at is not None and (succeeded_at is None or failed_at > succeeded_at)

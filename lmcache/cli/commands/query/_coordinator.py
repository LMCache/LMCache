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
    "metrics": CoordinatorApi(
        summary="Prometheus metrics, verbatim",
        path=lambda a: "/metrics",
        raw=True,
    ),
}

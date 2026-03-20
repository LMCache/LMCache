# SPDX-License-Identifier: Apache-2.0
"""``lmcache describe`` — show detailed status of a running LMCache service.

Usage::

    lmcache describe kvcache --url http://localhost:8000
"""

# Standard
import argparse
import json
import sys
import urllib.error
import urllib.request

# First Party
from lmcache.cli.commands.base import BaseCommand


class DescribeError(Exception):
    """Raised when the describe command cannot fetch or parse status data."""


def _normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """GET *url* and return the parsed JSON body.

    Raises:
        DescribeError: On network/HTTP errors.
    """
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            body = exc.read().decode()
            try:
                detail = json.loads(body).get("error", body)
            except (json.JSONDecodeError, AttributeError):
                detail = body
            raise DescribeError(f"Server unhealthy: {detail}") from exc
        raise DescribeError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DescribeError(f"Cannot connect to {url}: {exc.reason}") from exc
    except OSError as exc:
        raise DescribeError(f"Cannot connect to {url}: {exc}") from exc


def _fmt_used_gb(used_bytes: int, ratio: float) -> str:
    """Format L1 used memory as ``'XX.XX (YY.Y%)'``."""
    gb = used_bytes / (1024**3)
    pct = ratio * 100
    return f"{gb:.2f} ({pct:.1f}%)"


def _safe_get(data: dict, *keys, default=None):  # type: ignore[type-arg]
    """Walk nested dicts by *keys*, returning *default* on any miss."""
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


class DescribeCommand(BaseCommand):
    """Show detailed status of a running LMCache service."""

    def name(self) -> str:
        return "describe"

    def help(self) -> str:
        return "Show detailed status of a running LMCache service."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            choices=["kvcache"],
            help="What to describe.",
        )
        parser.add_argument(
            "--url",
            help="LMCache HTTP server URL (default to http://localhost:8080).",
            default="http://localhost:8080",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.target == "kvcache":
            self._describe_kvcache(args)

    def _describe_kvcache(self, args: argparse.Namespace) -> None:
        base_url = _normalize_url(args.url)
        status_url = f"{base_url}/api/status"

        try:
            data = _fetch_json(status_url)
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        metrics = self.create_metrics("LMCache KV Cache Service", args, width=48)

        # Health
        is_healthy = data.get("is_healthy")
        health = (
            "OK" if is_healthy else ("UNHEALTHY" if is_healthy is not None else None)
        )
        metrics.add("health", "Health", health)

        # URL
        metrics.add("url", "URL", base_url)

        # Engine type
        metrics.add("engine_type", "Engine type", data.get("engine_type"))

        # Chunk size
        metrics.add("chunk_size", "Chunk size", data.get("chunk_size"))

        # L1 capacity (GB)
        total_bytes = _safe_get(
            data, "storage_manager", "l1_manager", "memory_total_bytes"
        )
        if total_bytes is not None:
            capacity_gb = total_bytes / (1024**3)
            metrics.add("l1_capacity_gb", "L1 capacity (GB)", round(capacity_gb, 2))
        else:
            metrics.add("l1_capacity_gb", "L1 capacity (GB)", None)

        # L1 used (GB)
        used_bytes = _safe_get(
            data, "storage_manager", "l1_manager", "memory_used_bytes"
        )
        usage_ratio = _safe_get(
            data, "storage_manager", "l1_manager", "memory_usage_ratio"
        )
        if used_bytes is not None and usage_ratio is not None:
            metrics.add(
                "l1_used_gb", "L1 used (GB)", _fmt_used_gb(used_bytes, usage_ratio)
            )
        else:
            metrics.add("l1_used_gb", "L1 used (GB)", None)

        # Eviction policy
        eviction_policy = _safe_get(
            data, "storage_manager", "eviction_controller", "eviction_policy"
        )
        metrics.add("eviction_policy", "Eviction policy", eviction_policy)

        # Cached objects
        cached_objects = _safe_get(
            data, "storage_manager", "l1_manager", "total_object_count"
        )
        metrics.add("cached_objects", "Cached objects", cached_objects)

        # Active sessions
        metrics.add("active_sessions", "Active sessions", data.get("active_sessions"))

        # Per-model KV cache layout sections
        gpu_meta = data.get("gpu_context_meta", {})
        if gpu_meta:
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
                metrics.add_section(section_key, f"Model: {model_name}")
                sec = metrics[section_key]
                sec.add("world_size", "World size", world_size)
                sec.add("gpu_ids", "GPU IDs", ", ".join(info["gpu_ids"]))

                layout = info.get("layout")
                if layout:
                    sec.add("num_layers", "Num layers", layout["num_layers"])
                    sec.add("block_size", "Block size", layout["block_size"])
                    sec.add(
                        "hidden_dim_size",
                        "Hidden dim size",
                        layout["hidden_dim_size"],
                    )
                    sec.add("dtype", "Dtype", layout["dtype"])
                    sec.add("is_mla", "MLA", layout["is_mla"])
                    sec.add("num_blocks", "Num blocks", layout["num_blocks"])

        metrics.emit()

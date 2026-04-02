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
from lmcache.cli.metrics import Metrics

# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------


class DescribeError(Exception):
    """Raised when the describe command cannot fetch or parse status data."""


def normalize_url(url: str) -> str:
    """Ensure *url* has an ``http://`` or ``https://`` scheme."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def fetch_json(url: str, timeout: int = 10) -> dict:
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


def fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


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
    """Builds the ``describe kvcache`` output from a ``/api/status`` response.

    Each ``add_*`` method populates one logical section. The orchestrating
    :meth:`describe` calls them in order and emits the result.  Adding a
    new section is a one-method change — no other code needs to know
    about it.
    """

    def __init__(self, metrics: Metrics, data: dict, base_url: str) -> None:
        self.metrics = metrics
        self.data = data
        self.base_url = base_url

    def describe(self) -> None:
        """Run all section builders and emit."""
        self.add_overview()
        self.add_l1_storage()
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
        """L1 cache capacity, usage, eviction, and object count."""
        total_bytes = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_total_bytes"
        )
        if total_bytes is not None:
            self.metrics.add(
                "l1_capacity_gb",
                "L1 capacity (GB)",
                round(total_bytes / (1024**3), 2),
            )
        else:
            self.metrics.add("l1_capacity_gb", "L1 capacity (GB)", None)

        used_bytes = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_used_bytes"
        )
        usage_ratio = safe_get(
            self.data, "storage_manager", "l1_manager", "memory_usage_ratio"
        )
        if used_bytes is not None and usage_ratio is not None:
            gb = used_bytes / (1024**3)
            pct = usage_ratio * 100
            self.metrics.add("l1_used_gb", "L1 used (GB)", f"{gb:.2f} ({pct:.1f}%)")
        else:
            self.metrics.add("l1_used_gb", "L1 used (GB)", None)

        self.metrics.add(
            "eviction_policy",
            "Eviction policy",
            safe_get(
                self.data,
                "storage_manager",
                "eviction_controller",
                "eviction_policy",
            ),
        )
        self.metrics.add(
            "cached_objects",
            "Cached objects",
            safe_get(self.data, "storage_manager", "l1_manager", "total_object_count"),
        )
        self.metrics.add(
            "active_sessions", "Active sessions", self.data.get("active_sessions")
        )

    def add_models(self) -> None:
        """Per-model KV cache layout sections."""
        gpu_meta = self.data.get("gpu_context_meta", {})
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
            sec.add(
                "attention_backend",
                "Attention backend",
                layout.get("attention_backend"),
            )
            sec.add("gpu_kv_shape", "GPU KV shape", layout.get("gpu_kv_shape"))
            sec.add(
                "gpu_kv_concrete_shape",
                "GPU KV tensor shape",
                layout.get("gpu_kv_concrete_shape"),
            )
            sec.add("num_layers", "Num layers", layout["num_layers"])
            sec.add("block_size", "Block size", layout["block_size"])
            sec.add("hidden_dim_size", "Hidden dim size", layout["hidden_dim_size"])
            sec.add("dtype", "Dtype", layout["dtype"])
            sec.add("is_mla", "MLA", layout["is_mla"])
            sec.add("num_blocks", "Num blocks", layout["num_blocks"])
            sec.add(
                "cache_size_per_token",
                "Cache size per token (bytes)",
                layout["cache_size_per_token"],
            )

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
        base_url = normalize_url(args.url)
        try:
            data = fetch_json(f"{base_url}/api/status")
        except DescribeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        metrics = self.create_metrics("LMCache KV Cache Service", args, width=50)
        KVCacheDescriber(metrics, data, base_url).describe()

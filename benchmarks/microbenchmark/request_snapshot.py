# SPDX-License-Identifier: Apache-2.0
"""Benchmark snapshot preparation and per-server ZMQ encoding.

Run with LMCache and vLLM installed::

    CUDA_VISIBLE_DEVICES='' python -m benchmarks.microbenchmark.request_snapshot \
        --output /tmp/request-snapshot.json

The connector uses an in-memory adapter and an empty transfer schedule.
Cadence is simulated; wall timers measure copying and encoding separately.
Transport queues, RPC framing, network, receivers and GPU work are excluded.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch
import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time

# Third Party
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
import torch  # noqa: F401 -- initialize torch before native LMCache imports

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_module
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.v1.multiprocess.custom_types import RequestSnapshot, RequestSnapshotEntry
from lmcache.v1.multiprocess.transport.zmq_impl.mq import msgspec_encode


def connector_config(mode: str, interval_ms: int) -> VllmConfig:
    """Build scheduler configuration for the reporting mode."""
    transfer = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "lmcache.mp.request_snapshot": mode != "disabled",
            "lmcache.mp.request_snapshot_interval_ms": (
                interval_ms if mode == "interval" else 0
            ),
        },
    )
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="snapshot-benchmark", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=True, policy="fcfs"
            ),
            kv_transfer_config=transfer,
        ),
    )


def empty_schedule() -> SchedulerOutput:
    """Exclude KV transfer work from the measured callback."""
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
        ),
    )


def measure(
    workload: Workload, mode: str, steps: int, step_ms: int, interval_ms: int
) -> dict[str, object]:
    """Measure snapshot preparation and encoding for one workload.

    Args:
        workload: Request count, initial context length and server count.
        mode: ``disabled``, ``per_step`` or ``interval``.
        steps: Callback count.
        step_ms: Simulated step spacing, without sleeping.
        interval_ms: Minimum interval used only in ``interval`` mode.

    Returns:
        Timings, report counts and encoded byte counts.
    """
    clock = Clock()
    sink = SnapshotSink()
    token_lists = [list(range(workload.tokens)) for _ in range(workload.requests)]
    schedule = empty_schedule()
    callback_ns: list[int] = []
    encoding_ns: list[int] = []
    emitting_callback_ns: list[int] = []
    report_steps: list[int] = []
    encoded_bytes = 0
    with (
        patch.object(connector_module, "LMCacheMPSchedulerAdapter", return_value=sink),
        patch.object(connector_module, "print_banner_once"),
        patch.object(connector_module, "time", clock),
    ):
        connector = LMCacheMPConnector(
            connector_config(mode, interval_ms), KVConnectorRole.SCHEDULER
        )
        try:
            for index, tokens in enumerate(token_lists):
                connector.on_new_request(
                    cast(
                        Request,
                        SimpleNamespace(
                            request_id=f"request-{index}",
                            all_token_ids=tokens,
                            status=RequestStatus.RUNNING,
                            priority=0,
                            arrival_time=0.0,
                            resumable=False,
                        ),
                    )
                )
            gc.collect()
            for step in range(steps):
                clock.now = step * step_ms / 1000.0
                for tokens in token_lists:
                    tokens.append((workload.tokens + step) % 100_000)
                sink.snapshot = None
                started = time.perf_counter_ns()
                connector.build_connector_meta(schedule)
                elapsed = time.perf_counter_ns() - started
                callback_ns.append(elapsed)
                snapshot = sink.snapshot
                if snapshot is not None:
                    emitting_callback_ns.append(elapsed)
                    report_steps.append(step)
                    started = time.perf_counter_ns()
                    for _ in range(workload.servers):
                        payload = msgspec_encode(snapshot, RequestSnapshot)
                        encoded_bytes += len(payload)
                    encoding_ns.append(time.perf_counter_ns() - started)
                    # Exclude payload cleanup from callback timing.
                    sink.snapshot = None
                    del snapshot, payload
        finally:
            connector.shutdown()
    return {
        **asdict(workload),
        "mode": mode,
        "steps": steps,
        "report_steps": report_steps,
        "reports": len(report_steps),
        "encoded_bytes": encoded_bytes,
        "callback_ns": callback_ns,
        "encoding_ns": encoding_ns,
        "callback_mean_us": statistics.mean(callback_ns) / 1000,
        "emitting_callback_p95_us": (
            sorted(emitting_callback_ns)[
                math.ceil(len(emitting_callback_ns) * 0.95) - 1
            ]
            / 1000
            if emitting_callback_ns
            else None
        ),
        "encoding_total_ms": sum(encoding_ns) / 1_000_000,
        "callback_plus_encoding_us_per_step": (sum(callback_ns) + sum(encoding_ns))
        / steps
        / 1000,
    }


def main() -> None:
    """Run repeated comparisons and save raw measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("steps", "repeats", "warmup", "step_ms", "interval_ms"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    workloads = (
        Workload(16, 4096, 1),
        Workload(16, 32768, 1),
        Workload(16, 32768, 4),
        Workload(64, 32768, 4),
    )
    modes = ("disabled", "per_step", "interval")
    results: list[dict[str, object]] = []
    for workload in workloads:
        for mode in modes:
            measure(workload, mode, args.warmup, args.step_ms, args.interval_ms)
        for repeat in range(args.repeats):
            offset = repeat % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                result = measure(
                    workload, mode, args.steps, args.step_ms, args.interval_ms
                )
                result["repeat"] = repeat
                results.append(result)
                print(
                    f"{workload.requests}x{workload.tokens}, S={workload.servers}, "
                    f"{mode}, repeat={repeat}: {result['reports']} reports, "
                    f"callback={result['callback_mean_us']:.3f} us/step, "
                    f"encoding={result['encoding_total_ms']:.3f} ms",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "step_ms": args.step_ms,
                "interval_ms": args.interval_ms,
                "warmup_steps": args.warmup,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        Path(connector_module.__file__),
                    )
                },
                "scope": __doc__,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


@dataclass(frozen=True)
class Workload:
    """Request count, context length and server count."""

    requests: int
    tokens: int
    servers: int


@dataclass
class Clock:
    """Simulated time for the interval check."""

    now: float = 0.0

    def monotonic(self) -> float:
        """Return simulated seconds."""
        return self.now


@dataclass
class SnapshotSink:
    """Collect snapshots for encoding outside the callback timer."""

    lmcache_tokens_per_chunk: int = 256
    sequence: int = 0
    snapshot: RequestSnapshot | None = None

    def report_request_snapshot(
        self, requests: tuple[RequestSnapshotEntry, ...], scheduling_policy: str
    ) -> bool:
        """Capture the snapshot for encoding and return True to confirm submission."""
        self.sequence += 1
        self.snapshot = RequestSnapshot(
            "benchmark",
            self.sequence,
            "snapshot-benchmark",
            scheduling_policy,
            requests,
        )
        return True

    def shutdown(self) -> None:
        """Release the last captured snapshot."""
        self.snapshot = None


if __name__ == "__main__":
    main()

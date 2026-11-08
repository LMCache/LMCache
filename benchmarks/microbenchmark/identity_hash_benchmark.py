# SPDX-License-Identifier: Apache-2.0
"""Measure identity construction and repeated hashing through public cache APIs.

Run the same command in separate baseline and candidate checkouts. This measures
CPU metadata processing, not GPU transfer or model inference. L1 mode allocates
pinned host memory and requires an available accelerator runtime.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
import argparse
import json
import platform
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import DramL1Manager


def make_keys(count: int, tags: dict[str, str]) -> list[ObjectKey]:
    """Return independently constructed identities for the same object range."""
    return [
        ObjectKey(i.to_bytes(32, "little"), "benchmark", 0, tags=tags)
        for i in range(count)
    ]


def measure(
    operation: Callable[[list[ObjectKey]], None],
    stored: list[ObjectKey],
    tags: dict[str, str],
    batch_size: int,
    mode: str,
    rounds: int,
) -> list[float]:
    """Time an operation, optionally including construction of fresh keys.

    Reused mode passes stored identities. Fresh mode constructs equal identities
    outside timing; construct mode includes that construction in timing.
    """
    samples: list[float] = []
    reused = stored[:batch_size]
    for index in range(rounds + 10):
        batch = make_keys(batch_size, tags) if mode == "fresh" else reused
        start = time.perf_counter_ns()
        if mode == "construct":
            batch = make_keys(batch_size, tags)
        operation(batch)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if index >= 10:
            samples.append(elapsed_ms)
    return samples


def run_case(
    kind: str, tags_count: int, population: int, batch_size: int, rounds: int
) -> list[dict[str, object]]:
    """Measure LRU updates or a resident L1 reserve/touch/release lifecycle.

    All L1 operation results and returned bytes are validated. Pool allocation,
    initial admission and final victim-order checks are outside timing.
    """
    tags = {f"tag-{i}": f"value-{i}" for i in range(tags_count)}
    stored = make_keys(population, tags)
    policy = LRUEvictionPolicy()
    manager: DramL1Manager | None = None
    operation: Callable[[list[ObjectKey]], None]
    if kind == "lru":
        policy.on_keys_created(stored)
        operation = policy.on_keys_touched
    else:
        pool_bytes = max(64 << 20, population * 8192)
        manager = DramL1Manager(
            L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=pool_bytes,
                    use_lazy=False,
                    shm_name="",
                )
            )
        )
        manager.register_listener(L1EvictionPolicy(policy))
        layout = MemoryLayoutDesc(shapes=[torch.Size([256])], dtypes=[torch.uint8])
        reserved = manager.reserve_write(stored, [False] * population, layout)
        for status, obj in reserved.values():
            if status != L1Error.SUCCESS or obj is None:
                raise RuntimeError("L1 population failed")
            obj.tensor.fill_(37)
        if any(
            status != L1Error.SUCCESS
            for status in manager.finish_write(stored).values()
        ):
            raise RuntimeError("L1 admission failed")
        l1 = manager

        def l1_read(batch: list[ObjectKey]) -> None:
            results = l1.reserve_read(batch)
            l1.touch_keys(batch)
            released = l1.finish_read(batch)
            if any(status != L1Error.SUCCESS for status, _ in results.values()):
                raise RuntimeError("L1 read failed")
            if any(status != L1Error.SUCCESS for status in released.values()):
                raise RuntimeError("L1 release failed")

        operation = l1_read
    rows = []
    try:
        for mode in ["reused", "fresh", "construct"]:
            samples = measure(operation, stored, tags, batch_size, mode, rounds)
            rows.append(
                dict(
                    kind=kind,
                    tags=tags_count,
                    mode=mode,
                    population=population,
                    batch_size=batch_size,
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                )
            )
        expected = list(reversed(stored[batch_size:])) + list(
            reversed(stored[:batch_size])
        )
        if policy.get_eviction_candidates(population) != expected:
            raise RuntimeError("LRU victim order differs from the expected order")
        if manager is not None:
            results = manager.reserve_read(stored[:batch_size])
            for status, obj in results.values():
                if (
                    status != L1Error.SUCCESS
                    or obj is None
                    or not torch.all(obj.tensor == 37)
                ):
                    raise RuntimeError("L1 payload check failed")
            manager.finish_read(stored[:batch_size])
    finally:
        if manager is not None:
            manager.close()
    return rows


def main() -> None:
    """Run the selected component benchmark and save every timed sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["lru", "l1", "all"], default="lru")
    parser.add_argument("--population", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=51)
    parser.add_argument("--tags", type=int, nargs="+", default=[0, 4, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 0 < args.batch_size <= args.population
        or args.rounds < 1
        or min(args.tags) < 0
    ):
        parser.error("require population >= batch-size > 0, rounds > 0 and tags >= 0")
    kinds = ["lru", "l1"] if args.kind == "all" else [args.kind]
    rows = [
        row
        for kind in kinds
        for count in args.tags
        for row in run_case(kind, count, args.population, args.batch_size, args.rounds)
    ]
    args.output.write_text(
        json.dumps(dict(python=platform.python_version(), rows=rows), indent=2)
    )


if __name__ == "__main__":
    main()

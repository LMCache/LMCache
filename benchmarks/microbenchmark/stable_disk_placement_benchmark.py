# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Benchmark stable multi-disk placement and restart directory scanning.

The placement comparison uses the modulo rule from PR #4260 as the baseline
and the rendezvous rule used by the stable ``striped`` policy as the candidate.
It reports routing throughput, balance, and the fraction of keys remapped when
one disk is added. The scan case creates a temporary fs-native directory and
measures recovery metadata discovery separately from file creation.

Run with::

    python benchmarks/microbenchmark/stable_disk_placement_benchmark.py \
        --keys 100000 --disks 8 --scan-files 10000
"""

# Standard
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
import argparse
import json
import math
import tempfile
import time
import uuid

# Third Party
import blake3

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_key_codec import object_key_to_filename
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
    scan_existing_cache_entries,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    rendezvous_adapter_indices_for_keys,
)

T = TypeVar("T")


def _make_keys(count: int) -> list[ObjectKey]:
    return [
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(index),
            model_name="benchmark/model",
            kv_rank=index % 8,
            object_group_id=index % 4,
        )
        for index in range(count)
    ]


def _make_adapters(count: int) -> list[AdapterDescriptor]:
    adapters: list[AdapterDescriptor] = []
    for index in range(count):
        config = FSNativeL2AdapterConfig(base_path=f"/benchmark/disk-{index}")
        config.placement_id = str(uuid.uuid5(uuid.NAMESPACE_URL, config.base_path))
        adapters.append(AdapterDescriptor(index=index, config=config))
    return adapters


def _modulo_owner_ids(
    keys: list[ObjectKey],
    adapters: list[AdapterDescriptor],
) -> list[str]:
    """Return owners selected by the PR #4260 ``BLAKE3(key) % N`` rule."""
    sorted_adapters = sorted(adapters, key=lambda adapter: adapter.index)
    owners: list[str] = []
    for key in keys:
        digest = blake3.blake3(str(key).encode()).digest(length=8)
        slot = int.from_bytes(digest, "big") % len(sorted_adapters)
        placement_id = sorted_adapters[slot].placement_id
        assert placement_id is not None
        owners.append(placement_id)
    return owners


def _rendezvous_owner_ids(
    keys: list[ObjectKey],
    adapters: list[AdapterDescriptor],
) -> list[str]:
    by_index = {adapter.index: adapter for adapter in adapters}
    owners: list[str] = []
    for adapter_index in rendezvous_adapter_indices_for_keys(keys, adapters):
        placement_id = by_index[adapter_index].placement_id
        assert placement_id is not None
        owners.append(placement_id)
    return owners


def _best_seconds(function: Callable[[], T], repetitions: int) -> float:
    best = float("inf")
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def _remap_ratio(before: list[str], after: list[str]) -> float:
    return sum(left != right for left, right in zip(before, after, strict=True)) / len(
        before
    )


def _distribution_cv(owners: list[str], disk_count: int) -> float:
    counts = list(Counter(owners).values())
    counts.extend([0] * (disk_count - len(counts)))
    mean = len(owners) / disk_count
    variance = sum((count - mean) ** 2 for count in counts) / disk_count
    return math.sqrt(variance) / mean


def benchmark_placement(
    key_count: int,
    disk_count: int,
    repetitions: int,
) -> dict[str, object]:
    """Benchmark routing and add-one-disk remapping.

    Args:
        key_count: Number of deterministic object keys.
        disk_count: Number of disks before the topology change.
        repetitions: Timing repetitions; the best wall time is reported.

    Returns:
        JSON-serializable placement metrics for modulo and rendezvous hashing.

    Raises:
        ValueError: If an argument is not positive.
    """
    if key_count <= 0 or disk_count <= 0 or repetitions <= 0:
        raise ValueError("key_count, disk_count, and repetitions must be positive")

    keys = _make_keys(key_count)
    before = _make_adapters(disk_count)
    after = _make_adapters(disk_count + 1)

    modulo_before = _modulo_owner_ids(keys, before)
    modulo_after = _modulo_owner_ids(keys, after)
    rendezvous_before = _rendezvous_owner_ids(keys, before)
    rendezvous_after = _rendezvous_owner_ids(keys, after)

    modulo_seconds = _best_seconds(lambda: _modulo_owner_ids(keys, before), repetitions)
    rendezvous_seconds = _best_seconds(
        lambda: _rendezvous_owner_ids(keys, before), repetitions
    )
    return {
        "keys": key_count,
        "disks_before": disk_count,
        "disks_after": disk_count + 1,
        "ideal_add_disk_remap_ratio": 1 / (disk_count + 1),
        "modulo": {
            "remap_ratio": _remap_ratio(modulo_before, modulo_after),
            "keys_per_second": key_count / modulo_seconds,
            "distribution_cv": _distribution_cv(modulo_before, disk_count),
        },
        "rendezvous": {
            "remap_ratio": _remap_ratio(rendezvous_before, rendezvous_after),
            "keys_per_second": key_count / rendezvous_seconds,
            "distribution_cv": _distribution_cv(rendezvous_before, disk_count),
        },
    }


def benchmark_recovery_scan(file_count: int) -> dict[str, object]:
    """Benchmark recursive recovery scanning over generated cache files.

    Args:
        file_count: Number of valid ``.data`` files to generate.

    Returns:
        JSON-serializable scan latency and throughput metrics. File creation is
        intentionally excluded from the timed region.

    Raises:
        ValueError: If ``file_count`` is not positive.
    """
    if file_count <= 0:
        raise ValueError("file_count must be positive")

    keys = _make_keys(file_count)
    with tempfile.TemporaryDirectory(prefix="lmcache-recovery-bench-") as temp_dir:
        root = Path(temp_dir)
        for key in keys:
            (root / object_key_to_filename(key)).touch()

        started = time.perf_counter()
        recovered, skipped = scan_existing_cache_entries(temp_dir)
        elapsed = time.perf_counter() - started

    return {
        "files": file_count,
        "recovered": len(recovered),
        "skipped": skipped,
        "seconds": elapsed,
        "files_per_second": file_count / elapsed,
    }


def main() -> None:
    """Parse command-line arguments, run benchmarks, and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=100_000)
    parser.add_argument("--disks", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--scan-files", type=int, default=10_000)
    args = parser.parse_args()

    results = {
        "placement": benchmark_placement(args.keys, args.disks, args.repetitions),
        "recovery_scan": benchmark_recovery_scan(args.scan_files),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

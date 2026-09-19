# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the MP ObjectKey cache-identity fence."""

# Future
from __future__ import annotations

# Standard
from argparse import ArgumentParser, Namespace
from math import ceil
from statistics import median
from time import perf_counter_ns
import json

# First Party
from lmcache.v1.cache_identity import (
    CACHE_IDENTITY_REVISION_TAG,
    BaseCacheIdentity,
    CacheIdentity,
    CacheRepresentationIdentity,
    materialize_cache_identity_revision,
)
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


def _parse_args() -> Namespace:
    """Parse benchmark dimensions and repetition counts."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=512)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--object-groups", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def _identity() -> CacheIdentity:
    """Build the fixed revision bundle used by the benchmark."""
    return CacheIdentity(
        base=BaseCacheIdentity(
            model_revision="model-a",
            tokenizer_revision="tokenizer-a",
            weight_revision="rollout-step-4200",
            adapter_revision="policy-lora-17",
        ),
        representation=CacheRepresentationIdentity(
            topology_fingerprint="hybrid:mla=61,attention=3,tp=8",
            backend_revision="flashmla-2.3",
            kv_dtype="fp8_e4m3",
            quantization_revision="per-head-scale-v2",
            drop_algorithm_id="dapo-attention-score",
            drop_policy_revision="keep-0.75-v3",
        ),
    )


def _key(world_size: int, *, versioned: bool) -> IPCCacheServerKey:
    """Construct one scheduler-side IPC key for the benchmark."""
    return IPCCacheServerKey.from_token_ids(
        model_name="benchmark-model",
        world_size=world_size,
        worker_id=None,
        token_ids=[],
        request_configs=_identity().to_request_configs() if versioned else None,
    )


def _measure_key_expansion(
    legacy_key: IPCCacheServerKey,
    versioned_key: IPCCacheServerKey,
    chunk_hashes: list[bytes],
    group_ids: list[int],
    *,
    iterations: int,
    warmup: int,
) -> tuple[list[int], list[int]]:
    """Measure interleaved legacy and versioned full key expansions."""
    for _ in range(warmup):
        ipc_key_to_object_keys(legacy_key, chunk_hashes, group_ids)
        ipc_key_to_object_keys(versioned_key, chunk_hashes, group_ids)
    legacy_samples: list[int] = []
    versioned_samples: list[int] = []
    for iteration in range(iterations):
        ordered = (
            ((legacy_key, legacy_samples), (versioned_key, versioned_samples))
            if iteration % 2 == 0
            else ((versioned_key, versioned_samples), (legacy_key, legacy_samples))
        )
        for key, samples in ordered:
            start = perf_counter_ns()
            ipc_key_to_object_keys(key, chunk_hashes, group_ids)
            samples.append(perf_counter_ns() - start)
    return legacy_samples, versioned_samples


def _measure_materialization(
    request_configs: dict[str, object], *, iterations: int, warmup: int
) -> list[int]:
    """Measure one structured-identity materialization per request."""
    for _ in range(warmup):
        materialize_cache_identity_revision(request_configs)
    samples: list[int] = []
    for _ in range(iterations):
        start = perf_counter_ns()
        materialize_cache_identity_revision(request_configs)
        samples.append(perf_counter_ns() - start)
    return samples


def _percentile(samples: list[int], percentile: float) -> int:
    """Return a nearest-rank percentile for non-empty samples."""
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def main() -> int:
    """Run the benchmark, verify isolation invariants, and print JSON."""
    args = _parse_args()
    if (
        min(
            args.chunks,
            args.world_size,
            args.object_groups,
            args.iterations,
        )
        < 1
        or args.warmup < 0
    ):
        raise SystemExit("benchmark dimensions must be positive and warmup >= 0")

    chunk_hashes = [index.to_bytes(32, "big") for index in range(args.chunks)]
    group_ids = list(range(args.object_groups))
    legacy_key = _key(args.world_size, versioned=False)
    versioned_key = _key(args.world_size, versioned=True)

    legacy_output = ipc_key_to_object_keys(legacy_key, chunk_hashes, group_ids)
    versioned_output = ipc_key_to_object_keys(versioned_key, chunk_hashes, group_ids)
    if legacy_output[0][0].chunk_hash != chunk_hashes[0]:
        raise RuntimeError("legacy chunk hash changed")
    if versioned_output[0][0].chunk_hash == chunk_hashes[0]:
        raise RuntimeError("versioned chunk hash was not namespaced")
    if versioned_output[0][0].chunk_hash != versioned_output[-1][0].chunk_hash:
        raise RuntimeError("object groups derived different identity hashes")

    structured_configs: dict[str, object] = {
        **_identity().to_request_configs(),
        "lmcache.ttl": 60,
    }
    materialized = materialize_cache_identity_revision(structured_configs)
    if materialized is None or CACHE_IDENTITY_REVISION_TAG not in materialized:
        raise RuntimeError("structured identity was not materialized")
    if materialized.get("lmcache.ttl") != 60:
        raise RuntimeError("identity materialization dropped unrelated config")

    legacy_samples, versioned_samples = _measure_key_expansion(
        legacy_key,
        versioned_key,
        chunk_hashes,
        group_ids,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    materialization_samples = _measure_materialization(
        structured_configs,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    object_keys_per_run = args.chunks * args.world_size * args.object_groups
    legacy_median = median(legacy_samples)
    versioned_median = median(versioned_samples)
    result = {
        "schema_version": 1,
        "dimensions": {
            "chunks": args.chunks,
            "world_size": args.world_size,
            "object_groups": args.object_groups,
            "object_keys_per_run": object_keys_per_run,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "legacy": {
            "median_ns_per_run": legacy_median,
            "p95_ns_per_run": _percentile(legacy_samples, 0.95),
            "median_ns_per_object_key": legacy_median / object_keys_per_run,
        },
        "versioned": {
            "median_ns_per_run": versioned_median,
            "p95_ns_per_run": _percentile(versioned_samples, 0.95),
            "median_ns_per_object_key": versioned_median / object_keys_per_run,
        },
        "incremental": {
            "median_ns_per_run": versioned_median - legacy_median,
            "median_ns_per_chunk": (versioned_median - legacy_median) / args.chunks,
            "median_percent": (
                (versioned_median - legacy_median) / legacy_median * 100
            ),
        },
        "request_materialization": {
            "median_ns_per_request": median(materialization_samples),
            "p95_ns_per_request": _percentile(materialization_samples, 0.95),
        },
        "invariants": {
            "legacy_hash_unchanged": True,
            "versioned_hash_namespaced": True,
            "hash_reused_across_object_groups": True,
            "unrelated_request_config_preserved": True,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

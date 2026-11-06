# SPDX-License-Identifier: Apache-2.0
"""Measure native direct KV transfers with CUDA batch-copy support.

Run this same script with each revision's CUDA extension and identical flags::

    PYTHONPATH=. python benchmarks/microbenchmark/direct_copy_benchmark.py \
        --rounds 31 --loops 3 --output /tmp/direct-copy.json

Pinned objects, GPU buffers and native descriptors are prepared outside timing.
Each sample reports host submission and stream-complete wall time separately.
This measures transfers, not model throughput. Use alternating revision order
and retain all samples when sharing a GPU with other workloads.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import platform
import random
import statistics
import subprocess
import time

# Third Party
import torch

# First Party
from lmcache import device_ops, torch_dev
import lmcache.lmcache_native as native


@dataclass(frozen=True)
class Geometry:
    """Geometry of one layer-major kernel group and its host chunks."""

    block_tokens: int
    chunk_tokens: int
    layers: int
    kv_planes: int
    heads: int
    head_size: int
    chunks: int


CASES = {
    "nhd_16": Geometry(16, 16, 28, 2, 4, 128, 128),
    "nhd_256": Geometry(16, 256, 28, 2, 4, 128, 8),
    "nhd_1024": Geometry(16, 1024, 28, 2, 4, 128, 2),
    "mla_1024": Geometry(1024, 1024, 4, 1, 1, 576, 8),
}


def measure(geometry: Geometry, rounds: int, loops: int) -> list[dict[str, object]]:
    """Measure both transfer directions for one allocated geometry.

    Args:
        geometry: Layer-major paged buffer and chunk dimensions.
        rounds: Number of timed samples per direction.
        loops: Transfers per sample; each transfer waits for its stream.

    Returns:
        Direction, geometry, medians and every timing sample in milliseconds.

    Raises:
        RuntimeError: If the native executor submits an unexpected copy count.
    """
    g = geometry
    dtype = torch.bfloat16
    blocks_per_chunk = g.chunk_tokens // g.block_tokens
    nb = g.chunks * blocks_per_chunk + 8
    device = torch.device("cuda:0")
    shape = (
        (g.kv_planes, nb, g.block_tokens, g.heads, g.head_size)
        if g.kv_planes == 2
        else (nb, g.block_tokens, g.head_size)
    )
    paged = [torch.rand(shape, dtype=dtype, device=device) for _ in range(g.layers)]
    object_bytes = (
        g.kv_planes * g.layers * g.chunk_tokens * g.heads * g.head_size * dtype.itemsize
    )
    objects = [
        torch.empty(object_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(g.chunks)
    ]
    ids = random.Random(73).sample(range(nb), g.chunks * blocks_per_chunk)
    desc = native.PageBufferShapeDesc()
    desc.kv_size, desc.nl, desc.nb = g.kv_planes, g.layers, nb
    desc.bs, desc.nh, desc.hs = g.block_tokens, g.heads, g.head_size
    desc.element_size, desc.block_stride_elems = dtype.itemsize, 0
    fmt = (
        native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        if g.kv_planes == 2
        else native.EngineKVFormat.NL_X_NB_BS_HS
    )
    specs = [
        device_ops.DirectCopyGroupSpec(
            [tensor.data_ptr() for tensor in paged], desc, fmt, g.chunk_tokens, 0, ids
        )
    ]
    plans = [
        device_ops.DirectCopyObject(
            obj.data_ptr(), i * object_bytes, object_bytes, i, [0]
        )
        for i, obj in enumerate(objects)
    ]
    # Split counts at the same virtual pin boundaries as the production allocator.
    alignment = 64 << 20
    expected_copies = 0
    block_bytes = g.block_tokens * g.heads * g.head_size * dtype.itemsize
    for i in range(g.chunks):
        for entry in range(g.kv_planes * g.layers * blocks_per_chunk):
            offset = i * object_bytes + entry * block_bytes
            expected_copies += (
                1 + (offset + block_bytes - 1) // alignment - (offset // alignment)
            )
    stream = torch_dev.Stream(device=device)
    torch_dev.synchronize(device)
    results: list[dict[str, object]] = []
    for direction in (native.TransferDirection.D2H, native.TransferDirection.H2D):
        with torch_dev.stream(stream):
            for _ in range(5):
                count = device_ops.execute_direct_copy_transfer(
                    direction, device, alignment, specs, plans
                )
                stream.synchronize()
                if count != expected_copies:
                    raise RuntimeError(
                        f"Expected {expected_copies} copies, got {count}"
                    )
            samples = []
            for _ in range(rounds):
                submit_ns = total_ns = 0
                for _ in range(loops):
                    start = time.perf_counter_ns()
                    device_ops.execute_direct_copy_transfer(
                        direction, device, alignment, specs, plans
                    )
                    submitted = time.perf_counter_ns()
                    stream.synchronize()
                    finish = time.perf_counter_ns()
                    submit_ns += submitted - start
                    total_ns += finish - start
                samples.append(
                    {
                        "submit_ms": submit_ns / loops / 1e6,
                        "total_ms": total_ns / loops / 1e6,
                    }
                )
        results.append(
            {
                "direction": str(direction),
                "geometry": vars(g),
                "bytes": object_bytes * g.chunks,
                "copies": expected_copies,
                "median_submit_ms": statistics.median(s["submit_ms"] for s in samples),
                "median_total_ms": statistics.median(s["total_ms"] for s in samples),
                "samples": samples,
            }
        )
    return results


def main() -> None:
    """Parse cases and sample counts, print medians, and optionally save JSON.

    Raises:
        RuntimeError: If CUDA batch-copy support is unavailable.
        OSError: If saving results or invoking git fails.
        subprocess.CalledProcessError: If the current directory is not a repository.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.loops) <= 0:
        parser.error("rounds and loops must be positive")
    if not device_ops.batch_memcpy_supported():
        raise RuntimeError("Requires a CUDA/HIP build with usable batch-copy support")
    torch.manual_seed(123)
    results = []
    for name in args.cases:
        for result in measure(CASES[name], args.rounds, args.loops):
            result["case"] = name
            results.append(result)
            print(
                f"{name} {result['direction']} "
                f"submit={result['median_submit_ms']:.3f}ms "
                f"total={result['median_total_ms']:.3f}ms",
                flush=True,
            )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "runtime": torch.version.cuda,
                    "device": torch_dev.get_device_name(0),
                    "rounds": args.rounds,
                    "loops": args.loops,
                    "results": results,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

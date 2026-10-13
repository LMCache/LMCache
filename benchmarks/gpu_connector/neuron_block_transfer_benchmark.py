# SPDX-License-Identifier: Apache-2.0
"""Benchmark Neuron block KV transfers on the MP (engine-driven) path.

Times :meth:`DeviceOps.multi_layer_block_kv_transfer`, the single op every KV
movement in the multiprocess path dispatches to. ``gather_paged_kv_to_cpu`` and
``scatter_cpu_to_paged_kv`` in
``lmcache/v1/multiprocess/transfer_context/base.py`` call it for the store and
retrieve paths respectively, so this measures what ``submit_store`` and
``submit_retrieve`` spend moving bytes.

What this does **not** cover: the SHM writes and ZMQ round trips to the MP
server in ``commit_store`` / ``prepare_retrieve``. These are KV-movement
timings, not end-to-end store/retrieve latency.

Block ids are randomly scattered by default, modelling vLLM's block allocator.
That matters because the Neuron implementation copies runs of *consecutive*
blocks with ``narrow`` + ``copy_``, so its cost tracks fragmentation. Use
``--contiguous-blocks`` to see the best case.

``--verify`` checks a store/retrieve round trip is bit-exact and that
unselected blocks are untouched, instead of timing. A throughput number from a
path that moves the wrong bytes is worthless, so run ``--verify`` first at the
shapes you intend to measure.

Must run on Neuron hardware. Importing ``libtorch_neuronx_lite`` registers the
``neuron`` privateuse1 backend and must happen before any ``device="neuron"``
tensor is allocated.

Example:

    python benchmarks/gpu_connector/neuron_block_transfer_benchmark.py \
        --num-layers 16 --num-heads 8 --head-size 64 \
        --block-size 32 --chunk-tokens 32,128,256,512
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Callable, Sequence
import argparse
import json
import random
import statistics
import time

# Registering the Neuron privateuse1 backend (renamed to "neuron") is a side
# effect of importing libtorch_neuronx_lite. It must happen before any
# ``device="neuron..."`` tensor is allocated, so the import is unconditional
# but tolerated as absent off-hardware.
try:
    # Third Party
    import libtorch_neuronx_lite  # noqa: F401
except ImportError:
    pass

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import make_page_buffer_shape_desc
from lmcache.v1.platform import resolve_device_ops
import lmcache.lmcache_native as lmcache_native

#: vllm-neuron's per-layer layout, ``[2, NB, NH, BS, HS]``.
_HND_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """One direction at one chunk size."""

    direction: str
    chunk_tokens: int
    num_blocks: int
    median_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "direction": self.direction,
            "chunk_tokens": self.chunk_tokens,
            "num_blocks": self.num_blocks,
            "median_ms": round(self.median_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--cache-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument(
        "--chunk-tokens",
        type=str,
        default="32,128,256,512",
        help="Comma-separated LMCache chunk sizes in tokens.",
    )
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--device", type=str, default="neuron:0")
    parser.add_argument(
        "--contiguous-blocks",
        action="store_true",
        help="Allocate one consecutive run of blocks instead of scattering.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check round-trip correctness instead of timing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of a table.",
    )
    return parser.parse_args()


def _build_paged_layers(args: argparse.Namespace) -> list[torch.Tensor]:
    """Allocate the per-layer paged cache, each ``[2, NB, NH, BS, HS]``."""
    dtype = _DTYPES[args.dtype]
    return [
        torch.randn(
            2,
            args.cache_blocks,
            args.num_heads,
            args.block_size,
            args.head_size,
            dtype=dtype,
            device=args.device,
        )
        for _ in range(args.num_layers)
    ]


def _build_block_ids(
    args: argparse.Namespace, num_blocks: int, rng: random.Random
) -> list[int]:
    """Pick the blocks one chunk covers.

    Scattered by default: vLLM's allocator hands out whichever blocks happen to
    be free, in allocation order, not a consecutive run.
    """
    if num_blocks > args.cache_blocks:
        raise ValueError(
            f"chunk needs {num_blocks} blocks, cache_blocks={args.cache_blocks}"
        )
    if args.contiguous_blocks:
        start = rng.randrange(args.cache_blocks - num_blocks + 1)
        return list(range(start, start + num_blocks))
    return rng.sample(range(args.cache_blocks), num_blocks)


def _build_chunk(args: argparse.Namespace, chunk_tokens: int) -> torch.Tensor:
    """Allocate the host chunk in ``[2, L, T, NH*HS]`` layout."""
    return torch.zeros(
        2,
        args.num_layers,
        chunk_tokens,
        args.num_heads * args.head_size,
        dtype=_DTYPES[args.dtype],
        device="cpu",
    )


def _make_transfer(
    args: argparse.Namespace,
    paged_layers: Sequence[torch.Tensor],
    chunk: torch.Tensor,
    block_ids: Sequence[int],
    chunk_tokens: int,
    direction: "lmcache_native.TransferDirection",
) -> Callable[[], None]:
    """Bind one ``multi_layer_block_kv_transfer`` call, MP-path style.

    The shape descriptor is built exactly as ``gather_paged_kv_to_cpu`` builds
    it, so the benchmark exercises the real entry point rather than the
    implementation behind it.
    """
    device_ops = resolve_device_ops(paged_layers[0].device.type)
    # A sequence, not a dict: the format spec indexes this by integer layer id.
    shape_desc = make_page_buffer_shape_desc(
        list(paged_layers),
        _HND_FORMAT,
        layer_idx=0,
        num_layers_in_group=len(paged_layers),
        num_blocks=args.cache_blocks,
        block_size=args.block_size,
    )
    layers = list(paged_layers)
    blocks = list(block_ids)

    def _run() -> None:
        device_ops.multi_layer_block_kv_transfer(
            layers,
            [chunk],
            blocks,
            paged_layers[0].device,
            direction,
            shape_desc,
            chunk_tokens,
            _HND_FORMAT,
            0,
        )

    return _run


def _time_call(
    fn: Callable[[], None], *, iters: int, warmup_iters: int
) -> tuple[float, float, float]:
    """Return ``(median_ms, min_ms, max_ms)`` over ``iters`` calls.

    Per-call timing rather than total/iters: the Neuron path is synchronous, so
    each call is independently measurable and the spread is worth seeing.
    """
    for _ in range(warmup_iters):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples), max(samples)


def _verify(args: argparse.Namespace, rng: random.Random) -> int:
    """Assert a gather/scatter round trip is bit-exact and side-effect free."""
    chunk_tokens = max(int(t) for t in args.chunk_tokens.split(","))
    num_blocks = chunk_tokens // args.block_size
    paged_layers = _build_paged_layers(args)
    original = [layer.clone() for layer in paged_layers]
    block_ids = _build_block_ids(args, num_blocks, rng)
    chunk = _build_chunk(args, chunk_tokens)

    _make_transfer(
        args,
        paged_layers,
        chunk,
        block_ids,
        chunk_tokens,
        lmcache_native.TransferDirection.D2H,
    )()
    for layer in paged_layers:
        layer.zero_()
    _make_transfer(
        args,
        paged_layers,
        chunk,
        block_ids,
        chunk_tokens,
        lmcache_native.TransferDirection.H2D,
    )()

    selected = set(block_ids)
    untouched = [b for b in range(args.cache_blocks) if b not in selected][:16]
    failures = []
    for idx, (got, want) in enumerate(zip(paged_layers, original, strict=True)):
        got_host, want_host = got.cpu(), want.cpu()
        for kv in (0, 1):
            for block in block_ids:
                if not torch.equal(got_host[kv, block], want_host[kv, block]):
                    failures.append(f"layer {idx} kv {kv} block {block} not restored")
            for block in untouched:
                if not torch.equal(
                    got_host[kv, block], torch.zeros_like(got_host[kv, block])
                ):
                    failures.append(f"layer {idx} kv {kv} block {block} was written")

    if failures:
        print(f"VERIFY FAILED ({len(failures)} problems), first 5:")
        for line in failures[:5]:
            print(f"  {line}")
        return 1
    print(
        f"VERIFY OK: {len(block_ids)} blocks round-tripped bit-exact across "
        f"{args.num_layers} layers; {len(untouched)} sampled unselected blocks "
        f"left untouched"
    )
    return 0


def main() -> int:
    """Run the benchmark or the verification."""
    args = parse_args()
    rng = random.Random(args.seed)

    if args.verify:
        return _verify(args, rng)

    paged_layers = _build_paged_layers(args)
    results: list[BenchmarkResult] = []
    for token_spec in args.chunk_tokens.split(","):
        chunk_tokens = int(token_spec)
        num_blocks = chunk_tokens // args.block_size
        if num_blocks == 0:
            raise ValueError(
                f"chunk-tokens={chunk_tokens} is below block-size={args.block_size}"
            )
        block_ids = _build_block_ids(args, num_blocks, rng)
        chunk = _build_chunk(args, chunk_tokens)
        for name, direction in (
            ("store_d2h", lmcache_native.TransferDirection.D2H),
            ("retrieve_h2d", lmcache_native.TransferDirection.H2D),
        ):
            median, low, high = _time_call(
                _make_transfer(
                    args, paged_layers, chunk, block_ids, chunk_tokens, direction
                ),
                iters=args.iters,
                warmup_iters=args.warmup_iters,
            )
            results.append(
                BenchmarkResult(name, chunk_tokens, num_blocks, median, low, high)
            )

    config = {
        "device": args.device,
        "dtype": args.dtype,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_size": args.head_size,
        "block_size": args.block_size,
        "cache_blocks": args.cache_blocks,
        "blocks": "contiguous" if args.contiguous_blocks else "scattered",
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
    }

    if args.json:
        print(json.dumps({"config": config, "results": [r.as_dict() for r in results]}))
        return 0

    print(" ".join(f"{k}={v}" for k, v in config.items()))
    print()
    header = (
        f"{'direction':14}{'chunk':>7}{'blocks':>8}{'median':>11}{'min':>10}{'max':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.direction:14}{r.chunk_tokens:>7}{r.num_blocks:>8}"
            f"{r.median_ms:>9.2f}ms{r.min_ms:>8.2f}ms{r.max_ms:>8.2f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

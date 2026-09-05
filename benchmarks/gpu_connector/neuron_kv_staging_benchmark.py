# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the Neuron KV staging path.

Times ``NeuronKVBlockStager`` against synthetic paged-KV tensors shaped like a
real vllm-neuron deployment (HND layout, ``NL_X_TWO_NB_NH_BS_HS``).

The benchmark mirrors how the cache engine actually drives the stager, because
an earlier version of this file did not and reported a throughput two orders of
magnitude better than serving:

* **Chunked calls.** The engine calls ``from_gpu`` once per chunk of
  ``chunk_size`` tokens, not once per request, so a request is
  ``ceil(num_tokens / chunk_size)`` staging calls. Timing one whole-sequence
  call amortizes all per-call fixed cost exactly once and hides it.
* **Scattered blocks.** Slot mappings come from vLLM's block allocator, so the
  gathered block ids are non-contiguous. A dense ``arange`` slot mapping lets
  ``index_select`` degenerate into a strided slice and measures the best case.
* **No warm-only reporting.** Per-request cost is reported in full. A
  ``warm mean (2..N)`` over a repeated identical shape discards precisely the
  first-touch and per-shape device tracing cost that dominates serving.

Both directions are measured: ``transfer_into_key_value`` (D2H, the store path)
and ``transfer_from_key_value`` (H2D, the retrieve path).

Must run on Neuron hardware (import ``libtorch_neuronx_lite`` registers the
``neuron`` device); off-hardware it runs on CPU tensors as a functional check.

Example:

    python benchmarks/gpu_connector/neuron_kv_staging_benchmark.py \
        --requests 5 \
        --num-layers 16 \
        --num-tokens 1202 \
        --chunk-size 32 \
        --num-heads 4 \
        --head-size 64 \
        --block-size 16 \
        --device neuron
"""

# Standard
from dataclasses import dataclass
import argparse
import json
import random
import time

# Third Party
import torch

# Registering the Neuron privateuse1 backend (renamed to "neuron") is a side
# effect of importing libtorch_neuronx_lite. It must happen before any
# ``device="neuron..."`` tensor is allocated, so the import is unconditional
# but tolerated as absent off-hardware.
try:
    import libtorch_neuronx_lite  # noqa: F401
except ImportError:
    pass

# First Party
from lmcache.v1.gpu_connector.neuron_kv_staging import NeuronKVBlockStager
import lmcache.lmcache_native as lmcache_native


@dataclass(frozen=True)
class BenchConfig:
    """Shape and run parameters for the staging benchmark."""

    requests: int
    num_layers: int
    num_tokens: int
    chunk_size: int
    num_heads: int
    head_size: int
    block_size: int
    cache_blocks: int
    device: str
    dtype: torch.dtype
    seed: int


@dataclass(frozen=True)
class DirectionStats:
    """Timing results for one transfer direction."""

    calls_per_request: int
    request_secs: list[float]
    mean_request_sec: float
    mean_call_sec: float
    gib_per_sec: float


def _build_layer_tensors(config: BenchConfig) -> list[torch.Tensor]:
    """Allocate per-layer paged KV tensors in ``NL_X_TWO_NB_NH_BS_HS`` layout.

    The cache is sized to ``cache_blocks`` rather than to the request, so the
    gathered blocks are a sparse subset of a realistically large cache.

    :param config: Benchmark configuration.
    :returns: One tensor per layer of shape
        ``[2, cache_blocks, num_heads, block_size, head_size]`` on the target
        device.
    """
    shape = (
        2,
        config.cache_blocks,
        config.num_heads,
        config.block_size,
        config.head_size,
    )
    return [
        torch.randn(shape, dtype=config.dtype, device=config.device)
        for _ in range(config.num_layers)
    ]


def _build_scattered_slot_mapping(
    config: BenchConfig, rng: random.Random
) -> torch.Tensor:
    """Build a slot mapping over randomly scattered cache blocks.

    Models vLLM's block allocator: the request's tokens occupy whichever blocks
    happen to be free, in allocation order, not a contiguous run.

    :param config: Benchmark configuration.
    :param rng: Seeded source of randomness for block selection.
    :returns: A CPU int64 tensor of length ``num_tokens`` mapping each token to
        a slot in a scattered set of blocks.
    :raises ValueError: If the request needs more blocks than the cache holds.
    """
    blocks_needed = -(-config.num_tokens // config.block_size)  # ceil
    if blocks_needed > config.cache_blocks:
        raise ValueError(
            f"num_tokens={config.num_tokens} needs {blocks_needed} blocks, "
            f"cache_blocks={config.cache_blocks}"
        )
    chosen = rng.sample(range(config.cache_blocks), blocks_needed)
    slots = [
        block_id * config.block_size + offset
        for block_id in chosen
        for offset in range(config.block_size)
    ]
    return torch.tensor(slots[: config.num_tokens], dtype=torch.long, device="cpu")


def _build_key_value(config: BenchConfig, num_tokens: int) -> torch.Tensor:
    """Allocate the CPU staging tensor in ``[2, NL, NT, HS]`` layout.

    :param config: Benchmark configuration.
    :param num_tokens: Number of token slots this chunk covers.
    :returns: A CPU tensor of shape ``[2, num_layers, num_tokens, hidden]``.
    """
    hidden = config.num_heads * config.head_size
    return torch.empty(
        (2, config.num_layers, num_tokens, hidden), dtype=config.dtype, device="cpu"
    )


def _staged_gib(config: BenchConfig) -> float:
    """Return the payload size moved per request, in GiB.

    :param config: Benchmark configuration.
    :returns: Bytes of KV staged per request (both K and V, all layers, valid
        tokens only), expressed in GiB.
    """
    elt = torch.tensor([], dtype=config.dtype).element_size()
    hidden = config.num_heads * config.head_size
    total_bytes = 2 * config.num_layers * config.num_tokens * hidden * elt
    return total_bytes / (1024**3)


def _chunk_bounds(config: BenchConfig) -> list[tuple[int, int]]:
    """Split a request's tokens into the chunks the cache engine would use.

    :param config: Benchmark configuration.
    :returns: A list of ``(start, end)`` token offsets per chunk.
    """
    return [
        (start, min(start + config.chunk_size, config.num_tokens))
        for start in range(0, config.num_tokens, config.chunk_size)
    ]


def _time_direction(
    config: BenchConfig,
    layer_tensors: list[torch.Tensor],
    store: bool,
) -> DirectionStats:
    """Time one transfer direction across whole requests.

    A fresh stager is built per direction so cache state is not carried in from
    the other direction's run.

    :param config: Benchmark configuration.
    :param layer_tensors: Per-layer paged KV tensors on the target device.
    :param store: Time the D2H store path when true, the H2D retrieve path when
        false.
    :returns: Timing statistics for the direction.
    """
    rng = random.Random(config.seed)
    stager = NeuronKVBlockStager()
    fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    bounds = _chunk_bounds(config)
    gib = _staged_gib(config)

    request_secs: list[float] = []
    for _ in range(config.requests):
        # A new slot mapping per request: block ids differ every time in
        # serving, so any per-selection caching must not be handed a
        # artificially stable input.
        slot_mapping = _build_scattered_slot_mapping(config, rng)
        chunks = [
            (_build_key_value(config, end - start), slot_mapping[start:end])
            for start, end in bounds
        ]

        start_time = time.perf_counter()
        for key_value, chunk_slots in chunks:
            if store:
                stager.transfer_into_key_value(
                    key_value=key_value,
                    layer_tensors=layer_tensors,
                    slot_mapping=chunk_slots,
                    engine_kv_format=fmt,
                    block_size=config.block_size,
                    head_size=config.head_size,
                )
            else:
                stager.transfer_from_key_value(
                    key_value=key_value,
                    layer_tensors=layer_tensors,
                    slot_mapping=chunk_slots,
                    engine_kv_format=fmt,
                    block_size=config.block_size,
                    head_size=config.head_size,
                )
        request_secs.append(time.perf_counter() - start_time)

    mean_request = sum(request_secs) / len(request_secs)
    return DirectionStats(
        calls_per_request=len(bounds),
        request_secs=request_secs,
        mean_request_sec=mean_request,
        mean_call_sec=mean_request / len(bounds) if bounds else 0.0,
        gib_per_sec=gib / mean_request if mean_request > 0 else 0.0,
    )


def run_benchmark(config: BenchConfig) -> dict[str, object]:
    """Run the staging benchmark in both directions.

    :param config: Benchmark configuration.
    :returns: A dict of per-direction timing statistics plus the payload size.
    """
    layer_tensors = _build_layer_tensors(config)
    return {
        "staged_gib": _staged_gib(config),
        "store_d2h": _time_direction(config, layer_tensors, store=True),
        "retrieve_h2d": _time_direction(config, layer_tensors, store=False),
    }


def _format_direction(name: str, stats: DirectionStats) -> str:
    """Render one direction's statistics as an indented block.

    :param name: Human-readable direction label.
    :param stats: Timing statistics to render.
    :returns: A multi-line string.
    """
    per_request = ", ".join(f"{sec:.4f}" for sec in stats.request_secs)
    return (
        f"  {name}:\n"
        f"    calls/request:    {stats.calls_per_request}\n"
        f"    mean/request:     {stats.mean_request_sec:.4f}s "
        f"({stats.gib_per_sec:.4f} GiB/s)\n"
        f"    mean/call:        {stats.mean_call_sec * 1e3:.2f}ms\n"
        f"    per-request secs: [{per_request}]"
    )


def _parse_args() -> tuple[BenchConfig, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=1202)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--cache-blocks",
        type=int,
        default=2048,
        help="Total blocks in the paged cache; blocks are scattered within it.",
    )
    parser.add_argument("--device", type=str, default="neuron:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    config = BenchConfig(
        requests=args.requests,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        chunk_size=args.chunk_size,
        num_heads=args.num_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        cache_blocks=args.cache_blocks,
        device=args.device,
        dtype=torch.bfloat16,
        seed=args.seed,
    )
    return config, args.output_json


def main() -> None:
    config, output_json = _parse_args()
    stats = run_benchmark(config)
    store = stats["store_d2h"]
    retrieve = stats["retrieve_h2d"]
    assert isinstance(store, DirectionStats)
    assert isinstance(retrieve, DirectionStats)
    print(
        f"neuron_kv_staging: device={config.device} "
        f"layers={config.num_layers} tokens={config.num_tokens} "
        f"chunk_size={config.chunk_size} "
        f"staged={stats['staged_gib']:.4f}GiB/request\n"
        f"{_format_direction('store (D2H)', store)}\n"
        f"{_format_direction('retrieve (H2D)', retrieve)}"
    )
    if output_json:
        payload = {
            "staged_gib": stats["staged_gib"],
            "store_d2h": vars(store),
            "retrieve_h2d": vars(retrieve),
        }
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

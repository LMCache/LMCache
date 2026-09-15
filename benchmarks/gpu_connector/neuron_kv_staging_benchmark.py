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

``--verify`` checks correctness instead of speed: it drives a full store/retrieve
round trip on device and asserts the selected blocks come back bit-exact while
every other block is left alone. A throughput number from a path that moves the
wrong bytes is worthless, so run ``--verify`` first at the shapes you intend to
measure.

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
    # Third Party
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
    skip_prefix_tokens: int


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
                    skip_prefix_n_tokens=0,
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


def verify_round_trip(config: BenchConfig) -> None:
    """Assert a store/retrieve round trip restores KV exactly, on device.

    The staging path only moves bytes -- there is no arithmetic in it -- so a
    correct round trip is bit-exact even in bfloat16, and any mismatch is a real
    indexing or layout bug rather than tolerable drift.

    The whole cache is overwritten with unrelated data between the store and the
    retrieve. Without that, a retrieve that silently wrote nothing would still
    compare equal and the check would pass on a completely broken scatter. The
    overwrite also gives the untouched-blocks assertion something to detect: it
    catches a scatter that writes outside the blocks it was asked for, which is
    the failure mode that corrupts another request's KV rather than erroring.

    Both the store and the retrieve are driven chunk by chunk with a scattered
    slot mapping, matching how the cache engine calls the connector.

    :param config: Benchmark configuration; shapes and device come from it.
    :raises AssertionError: If any selected block fails to round trip, or if any
        block outside the selection was modified.
    """
    rng = random.Random(config.seed)
    stager = NeuronKVBlockStager()
    fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS

    layer_tensors = _build_layer_tensors(config)
    original = [layer.to("cpu") for layer in layer_tensors]

    slot_mapping = _build_scattered_slot_mapping(config, rng)
    chunks = [
        (_build_key_value(config, end - start), slot_mapping[start:end])
        for start, end in _chunk_bounds(config)
    ]

    for key_value, chunk_slots in chunks:
        stager.transfer_into_key_value(
            key_value=key_value,
            layer_tensors=layer_tensors,
            slot_mapping=chunk_slots,
            engine_kv_format=fmt,
            block_size=config.block_size,
            head_size=config.head_size,
        )

    # Overwrite the cache so the retrieve has to actually put the data back.
    poison = [torch.randn_like(reference) for reference in original]
    for layer, poisoned in zip(layer_tensors, poison, strict=True):
        layer.copy_(poisoned.to(layer.device))

    # The first chunk carries a prefix skip, as it does whenever vLLM's own
    # prefix cache already holds the head of the request. Those slots must come
    # back untouched: their blocks can be shared with another running request,
    # so writing them corrupts KV that is not ours.
    skip = config.skip_prefix_tokens
    skipped_slots = (
        chunks[0][1][:skip] if chunks else torch.tensor([], dtype=torch.long)
    )

    for index, (key_value, chunk_slots) in enumerate(chunks):
        stager.transfer_from_key_value(
            key_value=key_value,
            layer_tensors=layer_tensors,
            slot_mapping=chunk_slots,
            engine_kv_format=fmt,
            block_size=config.block_size,
            head_size=config.head_size,
            skip_prefix_n_tokens=skip if index == 0 else 0,
        )

    # Which token slots the retrieve was expected to write, at slot granularity
    # rather than block granularity: a partially covered tail block must keep its
    # unmapped slots and a skipped prefix slot must keep its old value, so
    # checking whole blocks would miss both failures.
    covered = torch.zeros(
        config.cache_blocks * config.block_size, dtype=torch.bool, device="cpu"
    )
    covered[slot_mapping] = True
    covered[skipped_slots] = False
    covered_blocks = covered.view(config.cache_blocks, config.block_size)

    # One layer is pulled back at a time. Holding every layer at once costs
    # num_layers * cache size on the host, which at serving shapes is tens of
    # GiB and turns a correctness check into an OOM.
    for layer_index, layer in enumerate(layer_tensors):
        actual = layer.to("cpu")
        reference = original[layer_index]
        poisoned = poison[layer_index]

        # Broadcast the slot mask over [2, NB, NH, BS, HS].
        mask = covered_blocks[None, :, None, :, None]
        expected = torch.where(mask, reference, poisoned)
        if not torch.equal(actual, expected):
            wrong = (actual != expected).any(dim=(0, 2, 4))  # -> [NB, BS]
            bad_blocks = wrong.any(dim=1).nonzero().flatten().tolist()
            block = bad_blocks[0]
            bad_slots = wrong[block].nonzero().flatten().tolist()
            raise AssertionError(
                f"layer {layer_index}: {len(bad_blocks)} block(s) wrong after "
                f"round trip, first is block {block} at within-block slots "
                f"{bad_slots} (mapped slots there: "
                f"{covered_blocks[block].nonzero().flatten().tolist()}); "
                f"{(actual != expected).sum().item()} elements differ in total"
            )

    num_selected = int(covered_blocks.any(dim=1).sum())
    num_partial = int((covered_blocks.any(dim=1) & ~covered_blocks.all(dim=1)).sum())
    print(
        f"verify: OK -- {int(covered.sum())} token slots across {num_selected} "
        f"blocks round tripped exactly on {config.num_layers} layers "
        f"({num_partial} partially covered block(s) kept their unmapped slots); "
        f"{config.cache_blocks - num_selected} blocks untouched; "
        f"{len(chunks)} chunks each direction"
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


def _parse_args() -> tuple[BenchConfig, str, bool]:
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
    parser.add_argument(
        "--skip-prefix-tokens",
        type=int,
        default=3,
        help="Leading tokens of the first chunk to treat as already held by "
        "vLLM's prefix cache during --verify. Those slots must come back "
        "unmodified. Only used by --verify.",
    )
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check a store/retrieve round trip is bit-exact, then exit without "
        "timing. Run this before trusting any throughput number.",
    )
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
        skip_prefix_tokens=args.skip_prefix_tokens,
    )
    return config, args.output_json, args.verify


def main() -> None:
    config, output_json, verify = _parse_args()
    if verify:
        verify_round_trip(config)
        return
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

# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the Neuron NIXL KV staging path.

Times ``NeuronKVBlockStager.transfer_into_key_value`` over repeated
iterations against synthetic paged-KV tensors shaped like a real
vllm-neuron deployment (HND layout, ``NL_X_TWO_NB_NH_BS_HS``).

The staging path gathers the selected KV blocks into a contiguous device
tensor with ``torch.index_select`` and copies that to CPU in one shot, so
per-iteration time reflects the on-device gather plus the device-to-host
copy. The "cold" (first) and "warm" (rest) split is retained to expose any
one-time device warmup cost.

Must run on Neuron hardware (import ``libtorch_neuronx_lite`` registers the
``neuron`` device); off-hardware it runs on CPU tensors as a functional
check.

Example:

    python benchmarks/gpu_connector/neuron_kv_staging_benchmark.py \
        --iterations 10 \
        --num-layers 16 \
        --num-tokens 3043 \
        --num-heads 4 \
        --head-size 64 \
        --block-size 16 \
        --device neuron
"""

# Standard
from dataclasses import dataclass
import argparse
import json
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

    iterations: int
    num_layers: int
    num_tokens: int
    num_heads: int
    head_size: int
    block_size: int
    device: str
    dtype: torch.dtype


def _build_layer_tensors(config: BenchConfig, num_blocks: int) -> list[torch.Tensor]:
    """Allocate per-layer paged KV tensors in ``NL_X_TWO_NB_NH_BS_HS`` layout.

    :param config: Benchmark configuration.
    :param num_blocks: Number of KV blocks per layer.
    :returns: One tensor per layer of shape
        ``[2, num_blocks, num_heads, block_size, head_size]`` on the target
        device.
    """
    shape = (2, num_blocks, config.num_heads, config.block_size, config.head_size)
    return [
        torch.randn(shape, dtype=config.dtype, device=config.device)
        for _ in range(config.num_layers)
    ]


def _build_slot_mapping(config: BenchConfig, num_blocks: int) -> torch.Tensor:
    """Build a contiguous slot mapping covering ``num_tokens`` slots.

    :param config: Benchmark configuration.
    :param num_blocks: Number of KV blocks available.
    :returns: A CPU int64 tensor of length ``num_tokens`` mapping each token
        to a distinct slot in ``[0, num_blocks * block_size)``.
    """
    capacity = num_blocks * config.block_size
    if config.num_tokens > capacity:
        raise ValueError(
            f"num_tokens={config.num_tokens} exceeds slot capacity={capacity}"
        )
    return torch.arange(config.num_tokens, dtype=torch.long, device="cpu")


def _build_key_value(config: BenchConfig) -> torch.Tensor:
    """Allocate the CPU destination tensor in ``[2, NL, NT, HS]`` layout.

    :param config: Benchmark configuration.
    :returns: A CPU tensor of shape
        ``[2, num_layers, num_tokens, num_heads * head_size]``.
    """
    hidden = config.num_heads * config.head_size
    shape = (2, config.num_layers, config.num_tokens, hidden)
    return torch.empty(shape, dtype=config.dtype, device="cpu")


def _staged_gib(config: BenchConfig) -> float:
    """Return the payload size moved per iteration, in GiB.

    :param config: Benchmark configuration.
    :returns: Bytes of KV actually staged (both K and V, all layers, valid
        tokens only), expressed in GiB.
    """
    elt = torch.tensor([], dtype=config.dtype).element_size()
    hidden = config.num_heads * config.head_size
    total_bytes = 2 * config.num_layers * config.num_tokens * hidden * elt
    return total_bytes / (1024**3)


def run_benchmark(config: BenchConfig) -> dict[str, float]:
    """Run the staging benchmark and return timing statistics.

    :param config: Benchmark configuration.
    :returns: A dict with cold/warm/mean per-iteration seconds and the
        effective warm-path throughput in GiB/s.
    """
    num_blocks = -(-config.num_tokens // config.block_size)  # ceil
    layer_tensors = _build_layer_tensors(config, num_blocks)
    slot_mapping = _build_slot_mapping(config, num_blocks)

    stager = NeuronKVBlockStager()
    gib = _staged_gib(config)

    per_iter: list[float] = []
    for _ in range(config.iterations):
        key_value = _build_key_value(config)
        start = time.perf_counter()
        stager.transfer_into_key_value(
            key_value=key_value,
            layer_tensors=layer_tensors,
            slot_mapping=slot_mapping,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
            block_size=config.block_size,
            head_size=config.head_size,
        )
        per_iter.append(time.perf_counter() - start)

    cold = per_iter[0]
    warm_samples = per_iter[1:] if len(per_iter) > 1 else per_iter
    warm_mean = sum(warm_samples) / len(warm_samples)
    return {
        "iterations": float(config.iterations),
        "staged_gib": gib,
        "cold_sec": cold,
        "warm_mean_sec": warm_mean,
        "warm_gib_per_sec": gib / warm_mean if warm_mean > 0 else 0.0,
        "mean_sec": sum(per_iter) / len(per_iter),
    }


def _parse_args() -> tuple[BenchConfig, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=3043)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="neuron:0")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()
    config = BenchConfig(
        iterations=args.iterations,
        num_layers=args.num_layers,
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        device=args.device,
        dtype=torch.bfloat16,
    )
    return config, args.output_json


def main() -> None:
    config, output_json = _parse_args()
    stats = run_benchmark(config)
    print(
        f"neuron_kv_staging: device={config.device} "
        f"layers={config.num_layers} tokens={config.num_tokens} "
        f"staged={stats['staged_gib']:.4f}GiB\n"
        f"  cold (iter 1):      {stats['cold_sec']:.4f}s\n"
        f"  warm mean (2..N):   {stats['warm_mean_sec']:.4f}s "
        f"({stats['warm_gib_per_sec']:.3f} GiB/s)\n"
        f"  overall mean:       {stats['mean_sec']:.4f}s"
    )
    if output_json:
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

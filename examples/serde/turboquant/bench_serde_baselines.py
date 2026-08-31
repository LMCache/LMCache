# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

"""Microbenchmark fp8 and TurboQuant serde backends on synthetic KV tensors."""

# Standard
from dataclasses import dataclass
from typing import Any, cast
import argparse
import json
import math
import statistics
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.fp8 import (
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantDeserializer,
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)
from lmcache.v1.memory_management import MemoryObj


@dataclass
class _FakeMemoryObj:
    tensor: torch.Tensor


def sync(device: torch.device) -> None:
    """Synchronize benchmark work on accelerator devices."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the Pearson correlation between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom.item() == 0:
        return float("nan")
    return ((a @ b) / denom).item()


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for benchmark samples.

    Args:
        values: Non-empty collection of latency samples.
        quantile: Percentile expressed as a value in ``[0, 1]``.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``values`` is empty or ``quantile`` is out of range.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def transfer_time_ms(num_bytes: int, bandwidth_gbps: float) -> float:
    """Estimate one-way transfer time at a link bandwidth.

    Args:
        num_bytes: Payload size in bytes.
        bandwidth_gbps: Effective link bandwidth in gigabits per second.

    Returns:
        Estimated transfer time in milliseconds.

    Raises:
        ValueError: If ``num_bytes`` is negative or bandwidth is not positive.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")
    return num_bytes * 8 / (bandwidth_gbps * 1_000_000)


def build_service_profiles(
    *,
    raw_bytes: int,
    serialized_bytes: int,
    encode_ms: float,
    decode_ms: float,
    bandwidths_gbps: list[float],
) -> list[dict[str, float | bool]]:
    """Model codec latency and break-even behavior across link bandwidths.

    The estimate intentionally uses measured codec latency plus ideal payload
    transfer time. It excludes storage and protocol overhead, so consumers can
    combine the JSON output with deployment-specific measurements.

    Args:
        raw_bytes: Uncompressed payload size.
        serialized_bytes: Serialized payload size.
        encode_ms: Measured median serialization latency.
        decode_ms: Measured median deserialization latency.
        bandwidths_gbps: Effective link bandwidths to evaluate.

    Returns:
        One profile per requested bandwidth.

    Raises:
        ValueError: If sizes or latencies are invalid.
    """
    if raw_bytes <= 0:
        raise ValueError("raw_bytes must be positive")
    if serialized_bytes < 0:
        raise ValueError("serialized_bytes must be non-negative")
    if encode_ms < 0 or decode_ms < 0:
        raise ValueError("codec latencies must be non-negative")

    codec_ms = encode_ms + decode_ms
    saved_bytes = raw_bytes - serialized_bytes
    break_even_bandwidth_gbps = (
        math.inf if codec_ms == 0 else max(saved_bytes, 0) * 8 / (codec_ms * 1_000_000)
    )
    profiles: list[dict[str, float | bool]] = []
    for bandwidth_gbps in bandwidths_gbps:
        raw_transfer_ms = transfer_time_ms(raw_bytes, bandwidth_gbps)
        serialized_transfer_ms = transfer_time_ms(serialized_bytes, bandwidth_gbps)
        total_ms = codec_ms + serialized_transfer_ms
        speedup = math.inf if total_ms == 0 else raw_transfer_ms / total_ms
        profiles.append(
            {
                "bandwidth_gbps": bandwidth_gbps,
                "raw_transfer_ms": raw_transfer_ms,
                "serialized_transfer_ms": serialized_transfer_ms,
                "encode_transfer_decode_ms": total_ms,
                "speedup_vs_raw_transfer": speedup,
                "beneficial": total_ms < raw_transfer_ms,
                "break_even_bandwidth_gbps": break_even_bandwidth_gbps,
            }
        )
    return profiles


def make_serde(
    name: str, preset: str | None, fp8_dtype: str, head_dim: int, block_size: int
) -> tuple[Serializer, Deserializer]:
    """Create the synchronous serde pair for one benchmark configuration."""
    if name == "fp8":
        dtype = getattr(torch, fp8_dtype)
        return (
            Fp8QuantizationSerializer(dtype),
            Fp8QuantizationDeserializer(dtype),
        )

    if name == "turboquant":
        assert preset is not None
        cfg = TurboQuantSerdeConfig(
            preset=preset,
            head_dim=head_dim,
            block_size=block_size,
        )
        return TurboQuantSerializer(cfg), TurboQuantDeserializer(cfg)

    raise ValueError(f"unknown serde: {name}")


def benchmark_one(
    serde_name: str,
    preset: str | None,
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    head_dim: int,
    block_size: int,
    fp8_dtype: str,
    bandwidths_gbps: list[float],
) -> dict[str, Any]:
    torch.manual_seed(2026)
    original = torch.randn(shape, dtype=dtype, device=device)

    serializer, deserializer = make_serde(
        serde_name,
        preset,
        fp8_dtype,
        head_dim,
        block_size,
    )

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    n_bytes = serializer.estimate_serialized_size(layout)

    compressed = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    recovered = torch.empty_like(original)

    src = _FakeMemoryObj(original)
    enc = _FakeMemoryObj(compressed)
    dec = _FakeMemoryObj(recovered)
    # fp8 / TurboQuant ignore the key; pass an empty ObjectKey.
    key = ObjectKey(chunk_hash=b"", model_name="", kv_rank=0)

    for _ in range(warmup):
        written = serializer.serialize(cast(MemoryObj, src), cast(MemoryObj, enc), key)
        if written != n_bytes:
            raise RuntimeError(f"written={written}, expected={n_bytes}")
        deserializer.deserialize(cast(MemoryObj, enc), cast(MemoryObj, dec), key)
    sync(device)

    encode_times = []
    decode_times = []

    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        written = serializer.serialize(cast(MemoryObj, src), cast(MemoryObj, enc), key)
        sync(device)
        t1 = time.perf_counter()

        if written != n_bytes:
            raise RuntimeError(f"written={written}, expected={n_bytes}")

        deserializer.deserialize(cast(MemoryObj, enc), cast(MemoryObj, dec), key)
        sync(device)
        t2 = time.perf_counter()

        encode_times.append((t1 - t0) * 1000)
        decode_times.append((t2 - t1) * 1000)

    raw_bytes = original.numel() * original.element_size()
    orig_f = original.float()
    rec_f = recovered.float()

    encode_p50_ms = statistics.median(encode_times)
    decode_p50_ms = statistics.median(decode_times)
    result = {
        "serde": serde_name,
        "preset": preset or fp8_dtype,
        "shape": "x".join(map(str, shape)),
        "dtype": str(dtype).replace("torch.", ""),
        "raw_MB": raw_bytes / 1024 / 1024,
        "serialized_MB": n_bytes / 1024 / 1024,
        "compression_ratio": raw_bytes / n_bytes,
        "encode_ms": sum(encode_times) / len(encode_times),
        "encode_p50_ms": encode_p50_ms,
        "encode_p95_ms": percentile(encode_times, 0.95),
        "decode_ms": sum(decode_times) / len(decode_times),
        "decode_p50_ms": decode_p50_ms,
        "decode_p95_ms": percentile(decode_times, 0.95),
        "corr": corrcoef(orig_f, rec_f),
        "mean_abs_err": torch.mean(torch.abs(orig_f - rec_f)).item(),
        "max_abs_err": torch.max(torch.abs(orig_f - rec_f)).item(),
    }
    result["service_profiles"] = build_service_profiles(
        raw_bytes=raw_bytes,
        serialized_bytes=n_bytes,
        encode_ms=encode_p50_ms,
        decode_ms=decode_p50_ms,
        bandwidths_gbps=bandwidths_gbps,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        nargs="+",
        default=[10.0, 25.0, 100.0, 200.0],
        help="Effective one-way bandwidths used for service-aware estimates.",
    )
    parser.add_argument("--fp8-dtype", default="float8_e4m3fn")
    parser.add_argument(
        "--turboquant-presets",
        nargs="+",
        default=[
            "turboquant_k8v4",
            "turboquant_4bit_nc",
            "turboquant_k3v4_nc",
            "turboquant_3bit_nc",
        ],
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if any(bandwidth <= 0 for bandwidth in args.bandwidth_gbps):
        parser.error("--bandwidth-gbps values must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    device = torch.device(args.device)
    num_tokens = args.blocks * args.block_size
    hidden_dim = args.kv_heads * args.head_dim
    shape = torch.Size([2, args.layers, num_tokens, hidden_dim])

    configs: list[tuple[str, str | None]] = [("fp8", None)]
    configs += [("turboquant", p) for p in args.turboquant_presets]

    rows = [
        benchmark_one(
            serde_name=serde_name,
            preset=preset,
            shape=shape,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            head_dim=args.head_dim,
            block_size=args.block_size,
            fp8_dtype=args.fp8_dtype,
            bandwidths_gbps=args.bandwidth_gbps,
        )
        for serde_name, preset in configs
    ]

    print(json.dumps(rows, indent=2))
    print()

    headers = [
        "serde",
        "preset",
        "raw_MB",
        "serialized_MB",
        "compression_ratio",
        "encode_ms",
        "decode_ms",
        "corr",
        "mean_abs_err",
        "max_abs_err",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for r in rows:
        print(
            " | ".join(
                [
                    str(r["serde"]),
                    str(r["preset"]),
                    f"{r['raw_MB']:.2f}",
                    f"{r['serialized_MB']:.2f}",
                    f"{r['compression_ratio']:.2f}",
                    f"{r['encode_ms']:.3f}",
                    f"{r['decode_ms']:.3f}",
                    f"{r['corr']:.6f}",
                    f"{r['mean_abs_err']:.6f}",
                    f"{r['max_abs_err']:.6f}",
                ]
            )
        )

    print()
    profile_headers = [
        "serde",
        "preset",
        "bandwidth_Gbps",
        "raw_transfer_ms",
        "codec_transfer_ms",
        "speedup",
        "beneficial",
        "break_even_Gbps",
    ]
    print(" | ".join(profile_headers))
    print(" | ".join(["---"] * len(profile_headers)))
    for row in rows:
        for profile in row["service_profiles"]:
            print(
                " | ".join(
                    [
                        str(row["serde"]),
                        str(row["preset"]),
                        f"{profile['bandwidth_gbps']:.2f}",
                        f"{profile['raw_transfer_ms']:.3f}",
                        f"{profile['encode_transfer_decode_ms']:.3f}",
                        f"{profile['speedup_vs_raw_transfer']:.3f}",
                        str(profile["beneficial"]),
                        f"{profile['break_even_bandwidth_gbps']:.3f}",
                    ]
                )
            )


if __name__ == "__main__":
    main()

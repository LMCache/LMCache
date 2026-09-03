# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

"""Microbenchmark fp8 and TurboQuant serde backends on synthetic KV tensors."""

# Standard
from dataclasses import dataclass
from typing import Any
import argparse
import json
import time

# Third Party
from benchmark_tensor_utils import tensor_error_metrics
from benchmark_utils import non_negative_int, positive_int, summarize_timings
import torch
import triton

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.fp8 import (
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.distributed.serde.turboquant import (
    TurboQuantDeserializer,
    TurboQuantSerdeConfig,
    TurboQuantSerializer,
)


@dataclass
class _FakeMemoryObj:
    tensor: torch.Tensor


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_serde(
    name: str, preset: str | None, fp8_dtype: str, head_dim: int, block_size: int
):
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
    seed: int,
    metric_chunk_elements: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
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
        written = serializer.serialize(src, enc, key)
        if written != n_bytes:
            raise RuntimeError(f"written={written}, expected={n_bytes}")
        deserializer.deserialize(enc, dec, key)
    sync()

    encode_times = []
    decode_times = []

    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        written = serializer.serialize(src, enc, key)
        sync()
        t1 = time.perf_counter()

        if written != n_bytes:
            raise RuntimeError(f"written={written}, expected={n_bytes}")

        deserializer.deserialize(enc, dec, key)
        sync()
        t2 = time.perf_counter()

        encode_times.append((t1 - t0) * 1000)
        decode_times.append((t2 - t1) * 1000)

    raw_bytes = original.numel() * original.element_size()
    encode_summary = summarize_timings(encode_times, raw_bytes)
    decode_summary = summarize_timings(decode_times, raw_bytes)
    error_metrics = tensor_error_metrics(
        original,
        recovered,
        chunk_elements=metric_chunk_elements,
    )

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        compute_capability = ".".join(
            map(str, torch.cuda.get_device_capability(device))
        )
    else:
        device_name = str(device)
        compute_capability = "n/a"

    return {
        "serde": serde_name,
        "preset": preset or fp8_dtype,
        "shape": "x".join(map(str, shape)),
        "dtype": str(dtype).replace("torch.", ""),
        "raw_MB": raw_bytes / 1024 / 1024,
        "serialized_MB": n_bytes / 1024 / 1024,
        "compression_ratio": raw_bytes / n_bytes,
        "encode_ms": encode_summary.mean_ms,
        "encode_p50_ms": encode_summary.p50_ms,
        "encode_p95_ms": encode_summary.p95_ms,
        "encode_raw_GiB_s": encode_summary.raw_gib_per_s,
        "decode_ms": decode_summary.mean_ms,
        "decode_p50_ms": decode_summary.p50_ms,
        "decode_p95_ms": decode_summary.p95_ms,
        "decode_raw_GiB_s": decode_summary.raw_gib_per_s,
        "corr": error_metrics.corr,
        "mean_abs_err": error_metrics.mean_abs_err,
        "max_abs_err": error_metrics.max_abs_err,
        "warmup": warmup,
        "iterations": iters,
        "seed": seed,
        "metric_chunk_elements": metric_chunk_elements,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--layers", type=positive_int, default=24)
    parser.add_argument("--blocks", type=positive_int, default=4096)
    parser.add_argument("--block-size", type=positive_int, default=16)
    parser.add_argument("--kv-heads", type=positive_int, default=2)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--warmup", type=non_negative_int, default=3)
    parser.add_argument("--iters", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--metric-chunk-elements", type=positive_int, default=16 * 1024 * 1024
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
            seed=args.seed,
            metric_chunk_elements=args.metric_chunk_elements,
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
        "encode_p50_ms",
        "encode_p95_ms",
        "encode_raw_GiB_s",
        "decode_ms",
        "decode_p50_ms",
        "decode_p95_ms",
        "decode_raw_GiB_s",
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
                    f"{r['encode_p50_ms']:.3f}",
                    f"{r['encode_p95_ms']:.3f}",
                    f"{r['encode_raw_GiB_s']:.2f}",
                    f"{r['decode_ms']:.3f}",
                    f"{r['decode_p50_ms']:.3f}",
                    f"{r['decode_p95_ms']:.3f}",
                    f"{r['decode_raw_GiB_s']:.2f}",
                    f"{r['corr']:.6f}",
                    f"{r['mean_abs_err']:.6f}",
                    f"{r['max_abs_err']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()

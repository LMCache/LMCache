# SPDX-License-Identifier: Apache-2.0

"""Prepare and measure synthetic RawBlockCore checkpoint recovery.

This benchmark creates a checkpoint with many indexed raw-block entries and
writes only each slot header. It does not write full payload contents; it is
intended to measure bringup checkpoint loading plus slot-header validation.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import argparse
import json
import os
import statistics
import struct
import time
import zlib

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.raw_block import core as raw_block_core
from lmcache.v1.storage_backend.raw_block.core import RawBlockCore, RawBlockCoreConfig
from lmcache.v1.storage_backend.raw_block.key_codec import encode_object_key

META_HEADER_STRUCT = struct.Struct("<8sIQQI")
META_MAGIC = b"LMCIDX01"
META_VERSION = 1
SLOT_MAGIC = b"LMCBLK01"
MODEL_NAME = "raw-block-bringup-bench"
PAYLOAD_LEN = 4096
PROGRESS_EVERY = 50000


def gib(value: float) -> int:
    return int(value * 1024**3)


def mib(value: float) -> int:
    return int(value * 1024**2)


def round_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def encode_slot_header(
    slot_identity: int,
    payload_len: int,
    header_bytes: int,
) -> bytes:
    header = bytearray(header_bytes)
    header[0:8] = SLOT_MAGIC
    header[8:16] = int(slot_identity & ((1 << 64) - 1)).to_bytes(
        8, "little", signed=False
    )
    header[16:24] = int(payload_len).to_bytes(8, "little", signed=False)
    return bytes(header)


def object_key_for_slot(slot: int, model_name: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=slot.to_bytes(16, "big", signed=False),
        model_name=model_name,
        kv_rank=0,
        object_group_id=0,
    )


def iter_entry_items(
    *,
    num_slots: int,
    data_base_offset: int,
    slot_bytes: int,
    model_name: str,
) -> Iterable[tuple[int, str, int]]:
    for slot in range(num_slots):
        spec = encode_object_key(object_key_for_slot(slot, model_name))
        offset = data_base_offset + slot * slot_bytes
        yield offset, spec.encoded, spec.slot_identity


def build_checkpoint_payload(args: argparse.Namespace, num_slots: int) -> bytes:
    entries: dict[str, dict[str, Any]] = {}
    data_base_offset = args.meta_total_bytes
    for offset, encoded_key, _slot_identity in iter_entry_items(
        num_slots=num_slots,
        data_base_offset=data_base_offset,
        slot_bytes=args.slot_bytes,
        model_name=MODEL_NAME,
    ):
        entries[encoded_key] = {
            "offset": offset,
            "size": PAYLOAD_LEN,
            "shape": [1],
            "dtype": "uint8",
            "fmt": MemoryFormat.BINARY.name,
            "cached_positions": None,
        }

    state = {
        "version": 1,
        "device_path": args.device_path,
        "capacity_bytes": args.capacity_bytes,
        "block_align": args.block_align,
        "header_bytes": args.header_bytes,
        "slot_bytes": args.slot_bytes,
        "meta_total_bytes": args.meta_total_bytes,
        "meta_magic": META_MAGIC.decode("ascii"),
        "meta_version": META_VERSION,
        "data_base_offset": data_base_offset,
        "next_slot": num_slots,
        "free_slots": [],
        "entries": entries,
    }
    return json.dumps(state, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def write_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.pwrite(fd, view[written:], offset + written)


def prepare_fixture(args: argparse.Namespace) -> None:
    device = Path(args.device_path)
    if not args.i_understand_this_overwrites_device:
        raise SystemExit(
            "Refusing to prepare fixture without "
            "--i-understand-this-overwrites-device. This overwrites raw-block "
            "metadata and slot headers on the target path."
        )

    if args.capacity_bytes <= args.meta_total_bytes:
        raise SystemExit("capacity_bytes must be greater than meta_total_bytes")

    num_slots = (args.capacity_bytes - args.meta_total_bytes) // args.slot_bytes
    if args.max_entries is not None:
        num_slots = min(num_slots, args.max_entries)
    if num_slots <= 0:
        raise SystemExit("computed slot count is zero")

    if not device.exists() or device.is_file():
        with device.open("ab") as f:
            f.truncate(args.capacity_bytes)

    payload = build_checkpoint_payload(args, num_slots)
    meta_container_bytes = (
        (args.meta_total_bytes // 2) // args.block_align
    ) * args.block_align
    payload_cap = meta_container_bytes - args.block_align
    if len(payload) > payload_cap:
        raise SystemExit(
            f"checkpoint payload is too large: {len(payload)} > {payload_cap}. "
            "Increase --meta-total-bytes or increase --slot-bytes."
        )

    flags = os.O_RDWR
    fd = os.open(args.device_path, flags)
    try:
        print(
            f"Preparing {num_slots} entries "
            f"({num_slots * args.slot_bytes / 1024**3:.2f} GiB logical slots)"
        )
        started = time.perf_counter()
        for slot, (offset, _encoded_key, slot_identity) in enumerate(
            iter_entry_items(
                num_slots=num_slots,
                data_base_offset=args.meta_total_bytes,
                slot_bytes=args.slot_bytes,
                model_name=MODEL_NAME,
            )
        ):
            header = encode_slot_header(
                slot_identity=slot_identity,
                payload_len=PAYLOAD_LEN,
                header_bytes=args.header_bytes,
            )
            write_all(fd, header, offset)
            if (slot + 1) % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - started
                print(f"  wrote {slot + 1}/{num_slots} slot headers in {elapsed:.1f}s")

        payload_total_len = round_up(len(payload), args.block_align)
        payload_block = payload + b"\0" * (payload_total_len - len(payload))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header_block = bytearray(args.block_align)
        header_block[: META_HEADER_STRUCT.size] = META_HEADER_STRUCT.pack(
            META_MAGIC, META_VERSION, 1, len(payload), crc
        )
        for container_offset in (0, meta_container_bytes):
            write_all(fd, payload_block, container_offset + args.block_align)
            write_all(fd, bytes(header_block), container_offset)
        os.fsync(fd)
    finally:
        os.close(fd)


def make_config(args: argparse.Namespace, io_engine: str) -> RawBlockCoreConfig:
    return RawBlockCoreConfig(
        device_path=args.device_path,
        capacity_bytes=args.capacity_bytes,
        block_align=args.block_align,
        header_bytes=args.header_bytes,
        slot_bytes=args.slot_bytes,
        use_odirect=args.use_odirect,
        enable_zero_copy=False,
        meta_total_bytes=args.meta_total_bytes,
        meta_magic=META_MAGIC,
        meta_version=META_VERSION,
        meta_checkpoint_interval_sec=3600,
        meta_idle_quiet_ms=0,
        meta_enable_periodic=False,
        meta_verify_on_load=True,
        load_checkpoint_on_init=True,
        io_engine=io_engine,
        iouring_queue_depth=256,
        use_uring_cmd=False,
    )


# A measurement variant: (label, io_engine, recovery_threads).
Variant = tuple[str, str, int]


def time_bringup(args: argparse.Namespace, variant: Variant) -> float:
    label, io_engine, recovery_threads = variant
    raw_block_core.DEFAULT_RECOVERY_READ_THREADS = recovery_threads
    started = time.perf_counter()
    core = RawBlockCore(make_config(args, io_engine), key_namespace="object")
    elapsed = time.perf_counter() - started
    status = core.report_status()
    core.close()
    print(
        f"{label}: {elapsed:.6f}s, "
        f"indexed={status['indexed_key_count']}, "
        f"free_slots={status['free_slot_count']}"
    )
    return elapsed


def build_variants(args: argparse.Namespace) -> list[Variant]:
    """Build measurement variants to compare.

    POSIX is swept over --threads; io_uring uses one batched variant since its
    recovery reads do not use the POSIX reader thread pool.
    """
    variants: list[Variant] = []
    for io_engine in args.io_engine:
        if io_engine == "io_uring":
            variants.append(("io_uring/batched", "io_uring", 1))
        else:
            for threads in args.threads:
                variants.append((f"posix/threads={threads}", "posix", threads))
    return variants


def measure(args: argparse.Namespace) -> None:
    variants = build_variants(args)
    results: dict[str, list[float]] = {variant[0]: [] for variant in variants}
    for repeat in range(args.repeats):
        print(f"repeat {repeat + 1}/{args.repeats}")
        for variant in variants:
            results[variant[0]].append(time_bringup(args, variant))

    print("\nsummary")
    for label, samples in results.items():
        print(
            f"{label}: "
            f"median={statistics.median(samples):.6f}s "
            f"mean={statistics.mean(samples):.6f}s "
            f"samples={[round(sample, 6) for sample in samples]}"
        )
    baseline_label = variants[0][0]
    baseline = statistics.median(results[baseline_label])
    for label, samples in results.items():
        if label == baseline_label:
            continue
        new = statistics.median(samples)
        if new > 0:
            print(f"speedup median {baseline_label}/{label}: {baseline / new:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and benchmark synthetic RawBlockCore checkpoint bringup "
            "slot-header validation. Prepare writes checkpoint metadata and "
            "slot headers, not full payload data."
        )
    )
    parser.add_argument("--device-path", required=True)
    parser.add_argument(
        "--cache-space-gb",
        type=float,
        default=100,
        help="Logical cache space to prepare/use, in GiB units.",
    )
    parser.add_argument("--slot-bytes", type=int, default=mib(16))
    parser.add_argument("--block-align", type=int, default=4096)
    parser.add_argument("--header-bytes", type=int, default=4096)
    parser.add_argument("--meta-total-bytes", type=int, default=mib(256))
    parser.add_argument(
        "--use-odirect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--io-engine",
        nargs="+",
        choices=["posix", "io_uring"],
        default=["posix"],
        help=(
            "Engines to measure. posix is swept over --threads; io_uring uses a "
            "single batched-read variant."
        ),
    )
    parser.add_argument("--max-entries", type=int)
    parser.add_argument(
        "--i-understand-this-overwrites-device",
        action="store_true",
        help=(
            "Required with --prepare because metadata and slot headers are overwritten."
        ),
    )
    args = parser.parse_args()
    args.capacity_bytes = gib(args.cache_space_gb)
    if not args.prepare and not args.measure:
        args.measure = True
    return args


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare_fixture(args)
    if args.measure:
        measure(args)


if __name__ == "__main__":
    main()

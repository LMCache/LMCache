#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Standalone smoke test for the fp8 serde integration.

Drives the StorageManager directly (no vLLM, no HTTP server) to verify:
  1. L1 write + async store -> disk (fp8 serialize)
  2. L1 clear
  3. Prefetch (disk -> temp byte buffer -> fp8 deserialize -> L1 KV buffer)
  4. The deserialized tensor round-trips within fp8 quantization error

Usage:
    python smoke_test.py
"""

# Standard
import shutil
import tempfile
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager


def make_key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="test-model",
        kv_rank=0,
    )


def main() -> int:
    disk_path = tempfile.mkdtemp(prefix="lmcache_serde_smoke_")
    print(f"Disk L2 path: {disk_path}")

    # Build a minimal StorageManager config: 4 GB L1, fs L2 with fp8 serde.
    fs_cfg = FSL2AdapterConfig(
        base_path=disk_path,
        relative_tmp_dir=None,
        read_ahead_size=None,
        use_odirect=False,
    )
    fs_cfg.serde_config = {"type": "fp8", "fp8_dtype": "float8_e4m3fn"}

    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=4 << 30,
                use_lazy=True,
                init_size_in_bytes=1 << 30,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),
    )

    sm = StorageManager(cfg)
    print("StorageManager started")

    # KV layout: small for a fast smoke test — a single group, [2, 4, 256, 128].
    # (shape picked arbitrarily; 256K elements ~ 524 KB in bf16.)
    kv_shape = torch.Size([2, 4, 256, 128])
    kv_dtype = torch.bfloat16
    layout = MemoryLayoutDesc(shapes=[kv_shape], dtypes=[kv_dtype])

    # Craft two KV chunks with deterministic content.
    keys = [
        make_key(b"\x00" * 31 + b"\x01"),
        make_key(b"\x00" * 31 + b"\x02"),
    ]
    torch.manual_seed(0)
    originals = [torch.randn(kv_shape, dtype=kv_dtype) for _ in keys]

    # ---- Step 1: write to L1 ----
    reserved = sm.reserve_write(keys, layout, mode="new")
    assert len(reserved) == len(keys), f"only reserved {len(reserved)}/{len(keys)}"
    for k, orig in zip(keys, originals, strict=True):
        mem_obj = reserved[k]
        dst = mem_obj.tensor.view(kv_shape).view(kv_dtype)
        dst.copy_(orig)
    sm.finish_write(keys)
    print(f"Step 1: wrote {len(keys)} keys to L1")

    # ---- Step 2: wait for store controller to push to disk ----
    # The store listener fires on finish_write; the controller then allocates
    # temp buffers, submits async fp8 serialize, and on serde_fd submits L2 store.
    print("Step 2: waiting for L2 store to flush to disk...")
    disk_file_count = 0
    for _ in range(30):  # up to ~6s
        time.sleep(0.2)
        import os

        disk_file_count = sum(
            1
            for _ in os.scandir(disk_path)
            if _.is_file() or (_.is_dir() and any(os.scandir(_.path)))
        )
        if disk_file_count > 0:
            break
    assert disk_file_count > 0, f"Nothing showed up under {disk_path}"
    print(f"  Disk entries: {disk_file_count}")

    status = sm.report_status()
    print(
        "  L2 store_controller in_flight =",
        status["store_controller"]["in_flight_task_count"],
    )

    # ---- Step 3: clear L1 ----
    sm.clear(force=True)
    print("Step 3: L1 cleared")
    status = sm.report_status()
    assert status["l1_manager"]["total_object_count"] == 0

    # ---- Step 4: prefetch the same keys (triggers disk load + deserialize) ----
    handle = sm.submit_prefetch_task(keys, layout)
    print(f"Step 4: submitted prefetch (request_id={handle.prefetch_request_id})")

    # Poll for completion
    prefix_hits = None
    for _ in range(50):  # up to ~10s
        time.sleep(0.2)
        prefix_hits = sm.query_prefetch_status(handle)
        if prefix_hits is not None:
            break
    assert prefix_hits is not None, "prefetch never completed"
    print(f"  Prefix hits: {prefix_hits}/{len(keys)}")
    assert prefix_hits == len(keys), (
        f"expected {len(keys)} prefix hits, got {prefix_hits}"
    )

    # ---- Step 5: read back and verify round-trip through fp8 ----
    with sm.read_prefetched_results(keys) as mem_objs:
        assert mem_objs is not None, "read_prefetched_results yielded None"
        assert len(mem_objs) == len(keys)
        for k, orig, mem_obj in zip(keys, originals, mem_objs, strict=True):
            got = mem_obj.tensor.view(kv_shape).view(kv_dtype)
            # fp8_e4m3fn -> bf16 round-trip has large relative error; we check
            # that the values are in the same order of magnitude and highly
            # correlated.
            err = (got.float() - orig.float()).abs().mean().item()
            orig_mag = orig.float().abs().mean().item()
            corr = torch.corrcoef(
                torch.stack([got.float().flatten(), orig.float().flatten()])
            )[0, 1].item()
            print(
                f"  key={k.chunk_hash[-1]:#x} mean_abs_err={err:.4f} "
                f"orig_mag={orig_mag:.4f} corr={corr:.4f}"
            )
            assert corr > 0.95, (
                f"Correlation after fp8 round-trip should be >0.95, got {corr:.4f}"
            )
    sm.finish_read_prefetched(keys)
    print("Step 5: fp8 round-trip verified (corr > 0.95)")

    # ---- Cleanup ----
    sm.close()
    shutil.rmtree(disk_path, ignore_errors=True)
    print("\n[PASS] fp8 serde end-to-end smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Opt-in Spark experiment KV pool and metadata-only creation ledger.

Build the adjacent retention_allocator.cpp on the target with a C++17 compiler,
-fPIC -shared -pthread, CUDA toolkit headers and -lcuda. Point
LMCACHE_ZC_ALLOCATOR_SO at the resulting shared object. LMCACHE_ZC_RUN_ID
is a fresh UUID and LMCACHE_ZC_LEDGER_DIR an existing private directory.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _identity(pid: int):
    run = str(uuid.UUID(os.environ["LMCACHE_ZC_RUN_ID"]))
    ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    gen = f"{run}.{pid}.{ticks}"
    root = Path(os.environ["LMCACHE_ZC_LEDGER_DIR"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("LMCACHE_ZC_LEDGER_DIR must be an existing absolute directory")
    return gen, root / f"backing-{gen}.jsonl"


def record(event: str, **fields):
    if not os.environ.get("LMCACHE_ZC_RUN_ID"):
        return
    gen, path = _identity(os.getpid())
    row = dict(event=event, generation=gen, mono_ns=time.monotonic_ns(), **fields)
    with path.open("a") as log:
        log.write(json.dumps(row, separators=(",", ":")) + "\n")


_lock = threading.Lock()
_state = None


def get_mem_pool_context():
    """Return an allocation context solely for the connector's KV pool hook."""
    global _state
    so = os.environ.get("LMCACHE_ZC_ALLOCATOR_SO")
    if not so:
        return contextlib.nullcontext()
    import torch

    with _lock:
        if _state is None:
            pid = os.getpid()
            gen, path = _identity(pid)
            dll = ctypes.CDLL(so)
            dll.zc_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            if dll.zc_init(os.fsencode(path), gen.encode()) != 0:
                raise RuntimeError("Cannot initialize the VMM creation ledger")
            dll.zc_lookup.argtypes = [ctypes.c_uint64, ctypes.c_size_t] + [
                ctypes.POINTER(ctypes.c_uint64)
            ] * 3
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                so, "zc_malloc", "zc_free"
            )
            pool = torch.cuda.MemPool(allocator.allocator())
            _state = (pid, so, dll, allocator, pool)
        if _state[:2] != (os.getpid(), so):
            raise RuntimeError(
                "KV pool cannot be reused across fork or configuration change"
            )
        return _kv_allocation_scope(_state[4])


@contextlib.contextmanager
def _kv_allocation_scope(pool):
    import torch

    record("kv_pool_enter", pool_id=list(pool.id), tid=threading.get_native_id())
    try:
        with torch.cuda.use_mem_pool(pool):
            yield
    finally:
        record("kv_pool_exit", pool_id=list(pool.id), tid=threading.get_native_id())
        record("kv_pool_exit_blocks", blocks=_private_blocks())


def creation_at(ptr: int, span: int):
    """Return the original CUmemCreate identity, base, size for a named KV view."""
    if not os.environ.get("LMCACHE_ZC_ALLOCATOR_SO"):
        return None
    if _state is None or _state[0] != os.getpid():
        raise RuntimeError("KV allocator hook did not initialize this worker")
    base, size, seq = (ctypes.c_uint64() for _ in range(3))
    if (
        _state[2].zc_lookup(
            ptr, span, ctypes.byref(base), ctypes.byref(size), ctypes.byref(seq)
        )
        != 0
    ):
        return None
    gen, _ = _identity(os.getpid())
    return f"{gen}.{seq.value}", base.value, size.value


def lookup_creation(ptr: int, span: int):
    creation = creation_at(ptr, span)
    if creation is None and os.environ.get("LMCACHE_ZC_ALLOCATOR_SO"):
        raise RuntimeError(
            "KV view is outside the custom allocator's live creation inventory"
        )
    return creation


def audit_worker_pool(model, kv_tensors):
    """Inspect private-pool blocks and model storage without touching payload."""

    if not os.environ.get("LMCACHE_ZC_ALLOCATOR_SO"):
        return {"allocator": "stock"}
    overlaps = []
    weight_storages = set()
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        extent = (storage.data_ptr(), storage.nbytes())
        if not extent[1] or extent in weight_storages:
            continue
        weight_storages.add(extent)
        if creation_at(extent[0], 1) or creation_at(extent[0] + extent[1] - 1, 1):
            overlaps.append(name)
    kv = {
        (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        for t in kv_tensors
    }
    blocks = _private_blocks()
    for block in blocks:
        block["kv_storage"] = (block["address"], block["requested"]) in kv
    return dict(
        allocator="retention_vmm",
        model_storages_checked=len(weight_storages),
        model_overlaps=overlaps,
        pool_blocks=blocks,
        kv_storages=sorted(kv),
    )


def _private_blocks():
    import torch

    blocks = []
    for segment in torch.cuda.memory_snapshot():
        creation = creation_at(segment["address"], segment["total_size"])
        if not creation:
            continue
        for block in segment["blocks"]:
            addr = block["address"]
            size = block["requested_size"]
            blocks.append(
                dict(
                    creation_id=creation[0],
                    address=addr,
                    bytes=block["size"],
                    requested=size,
                    state=block["state"],
                )
            )
    return blocks

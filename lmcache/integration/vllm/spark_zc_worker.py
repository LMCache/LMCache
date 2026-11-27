# SPDX-License-Identifier: Apache-2.0
"""Private TP1 Spark diagnostics and target-only scheduler pins."""

from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import os
import re
import struct
import time

TARGET = "0fd976c40e819a2dad3e4cef45a9714d563cd649f1f9977caafab95f0ff1dcdd"


def identity():
    return {
        "pid": os.getpid(),
        "start_ticks": int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        "run_id": os.environ.get("ZC_RUN_ID", os.environ.get("LMCACHE_ZC_RUN_ID")),
    }


def emit(role, event, **fields):
    row = dict(**identity(), event=event, monotonic_ns=time.monotonic_ns(), **fields)
    path = Path(os.environ["LMCACHE_ZC_WORKER_DIR"]) / f"{role}-{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def pin_target(connector, meta):
    """Hold one extra scheduler ref per source block until worker death."""
    salt = os.environ["LMCACHE_ZC_TARGET_SALT"]
    if not salt or meta.cache_salt != salt:
        return
    if connector.request_trackers[meta.request_id].num_prompt_tokens != 4096:
        return
    op = meta.op
    if len(op.token_bytes) < 4096 * 4:
        return
    tokens = struct.unpack_from(">4096I", op.token_bytes)
    if sha256(struct.pack("<4096I", *tokens)).hexdigest() != TARGET:
        return
    if connector._group_tokens_per_block != [16] or connector.lazy_offload:
        raise RuntimeError(
            "retention PoC requires one 16-token group and eager store submission"
        )
    if not 0 <= op.start < op.end <= 4096 or op.start % 256 or op.end % 256:
        raise RuntimeError("invalid target store interval")
    slots = op.flat_block_ids
    if len(slots) != (op.end - op.start) // 16 or len(set(slots)) != len(slots):
        raise RuntimeError("invalid target store block table")
    pins = connector._spark_zc_pins
    positions = list(range(op.start // 16, op.end // 16))
    if any(pos in pins for pos in positions) or set(slots).intersection(pins.values()):
        raise RuntimeError("target pages may only be published once")
    pool = connector._gpu_block_pool
    blocks = [pool.blocks[slot] for slot in slots]
    if any(
        b.ref_cnt <= 0 or b.block_id != s for b, s in zip(blocks, slots, strict=True)
    ):
        raise RuntimeError("source block is not owned by active request")
    before = [b.ref_cnt for b in blocks]
    pool.touch(blocks)
    pins.update(zip(positions, slots, strict=True))
    emit(
        "scheduler",
        "pin",
        request_id=meta.request_id,
        cache_salt=salt,
        target_sha256=TARGET,
        start=op.start,
        end=op.end,
        block_ids=slots,
        refs_before=before,
        refs_after=[b.ref_cnt for b in blocks],
        pinned_blocks=len(pins),
        unavailable_bytes=len(pins) * 16 * 147456,
    )


def _layers():
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    adapter = get_kv_transfer_group().worker_adapter
    return list(adapter.kv_caches.items())


def _driver_probe(driver, ptr):
    row = {}
    status, base, size = driver.cuMemGetAddressRange(driver.CUdeviceptr(ptr))
    row.update(
        range_status=int(status),
        range_start=int(base) if not status else None,
        range_bytes=int(size) if not status else None,
    )
    status, handle = driver.cuMemRetainAllocationHandle(ptr)
    row["retain_status"] = int(status)
    if not status:
        try:
            status, prop = driver.cuMemGetAllocationPropertiesFromHandle(handle)
            row["properties_status"] = int(status)
            if not status:
                row.update(
                    requested_handle_types=int(prop.requestedHandleTypes),
                    allocation_type=int(prop.type),
                    location_type=int(prop.location.type),
                    location_id=int(prop.location.id),
                )
        finally:
            (status,) = driver.cuMemRelease(handle)
            row["release_status"] = int(status)
            if status:
                emit("worker", "diagnostic_release_failed", probe=row)
                raise RuntimeError("diagnostic VMM reference failed to release")
    return row


class SparkZCWorker:
    def zc_inventory(self):
        """Metadata-only inspection on already registered worker KV."""
        from dataclasses import asdict
        from cuda.bindings import driver
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        from lmcache.v1.platform.ipc_policy import get_ipc_policy
        import torch
        import vllm

        layers = _layers()
        probes = {}
        info = []
        from lmcache.v1.platform.devices.cuda.utils import _bind_primary_context

        _bind_primary_context(layers[0][1].device.index)
        for name, t in layers:
            ptr = t.data_ptr()
            probes.setdefault(ptr, None)
            if probes[ptr] is None:
                probes[ptr] = _driver_probe(driver, ptr)
            info.append(
                dict(
                    layer=name,
                    tensor_ptr=ptr,
                    shape=list(t.shape),
                    stride=list(t.stride()),
                    dtype=str(t.dtype),
                    storage_base=t.untyped_storage().data_ptr(),
                    storage_bytes=t.untyped_storage().nbytes(),
                    storage_offset=t.storage_offset(),
                    element_size=t.element_size(),
                    driver=probes[ptr],
                )
            )
        dist = metadata.distribution("vllm")
        status, driver_version = driver.cuDriverGetVersion()
        adapter = get_kv_transfer_group().worker_adapter
        return emit(
            "worker",
            "inventory",
            instance_id=int(adapter.instance_id),
            connector_module=type(get_kv_transfer_group()).__module__,
            transfer_context=type(adapter.transfer_ctx).__name__,
            ipc_policy=asdict(get_ipc_policy()),
            allocator_so=os.environ.get("LMCACHE_ZC_ALLOCATOR_SO"),
            vllm_version=dist.version,
            vllm_module=vllm.__file__,
            vllm_record_sha256=sha256(dist.read_text("RECORD").encode()).hexdigest(),
            lmcache_version=metadata.version("lmcache"),
            torch_version=str(torch.__version__),
            cuda_build=torch.version.cuda,
            driver_api_version=int(driver_version) if not status else None,
            device=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            allocator=torch.cuda.get_allocator_backend(),
            layers=info,
            diagnostic_refs=0,
            diagnostic_fds=0,
        )

    def zc_sync(self):
        """Fence worker GPU activity without enqueuing diagnostic payload work."""
        import torch

        with torch.cuda.nvtx.range("spark_zc/worker_final_sync"):
            torch.cuda.synchronize()
        return emit("worker", "synchronized", diagnostic_refs=0, diagnostic_fds=0)

    def zc_hash_target(self, block_ids):
        """Hash complete target in layer/plane/token/head/element order."""
        import torch

        if (
            len(block_ids) != 256
            or len(set(block_ids)) != 256
            or any(type(s) is not int or s < 0 for s in block_ids)
        ):
            raise ValueError("expected 256 distinct ordered physical page IDs")
        layers = _layers()
        indices = [
            re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name) for name, _ in layers
        ]
        if len(layers) != 36 or any(
            m is None or int(m[1]) != i for i, m in enumerate(indices)
        ):
            raise ValueError("registered order is not Qwen3 layer 0 through 35")
        digests = [sha256() for _ in range(16)]
        lengths = [0] * 16
        torch.cuda.synchronize()
        start = time.monotonic_ns()
        with (
            torch.inference_mode(),
            torch.cuda.nvtx.range("spark_zc/hash_worker_outside_store"),
        ):
            for name, t in layers:
                if tuple(t.shape[1:]) == (8, 16, 256):
                    t = t.transpose(1, 2)
                if (
                    tuple(t.shape[1:]) != (16, 8, 256)
                    or t.dtype != torch.bfloat16
                    or max(block_ids) >= t.shape[0]
                ):
                    raise ValueError(
                        f"unsupported registered layout {name}: {t.shape}/{t.stride()}"
                    )
                for i, digest in enumerate(digests):
                    slots = block_ids[16 * i : 16 * i + 16]
                    for plane in t.split(128, dim=-1):
                        data = plane[slots].contiguous().view(torch.uint8).cpu().numpy()
                        digest.update(memoryview(data).cast("B"))
                        lengths[i] += data.nbytes
        torch.cuda.synchronize()
        if lengths != [36 * 2**20] * 16:
            raise ValueError(f"incomplete target lengths {lengths}")
        chunks = [
            dict(
                chunk_index=i,
                start=i * 256,
                end=(i + 1) * 256,
                nbytes=n,
                sha256=h.hexdigest(),
                block_ids=block_ids[16 * i : 16 * i + 16],
            )
            for i, (h, n) in enumerate(zip(digests, lengths, strict=True))
        ]
        return emit(
            "worker",
            "target_hash",
            begin_ns=start,
            chunks=chunks,
            layers=[name for name, _ in layers],
            digest_order="layer,plane,token,head,element",
            diagnostic_refs=0,
            diagnostic_fds=0,
        )

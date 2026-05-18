#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture a small LMCache MP protocol golden trace.

By default this covers controller requests plus token-key lookup/query paths.
Use ``--include-pytorch-cuda-kv`` or ``--include-raw-cuda-kv`` to append
Python-captured CUDA IPC STORE/LOOKUP/RETRIEVE byte checksum cases that native
replay must match.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
import tempfile
import time

# Third Party
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    RawCudaIPCWrapper,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    # Third Party
    import torch

TraceResponse = bool | int | str | None
TracePayload = int | str | dict[str, object]

_KV_TRACE_CHUNK_SIZE = 32
_KV_TRACE_INSTANCE_ID = 4321
_KV_TRACE_MODEL = "facebook/opt-125m"
_KV_TRACE_STORE_BLOCKS = [1, 2]
_KV_TRACE_RETRIEVE_BLOCKS = [3, 4]
_KV_TRACE_PYTORCH_REQUEST_ID = "trace-kv-pytorch-cuda-1"
_KV_TRACE_RAW_REQUEST_ID = "trace-kv-raw-cuda-1"
_FS_L2_PARTIAL_HIT_REQUEST_ID = "trace-fs-l2-partial-hit-1"
_FS_L2_PARTIAL_HIT_PAYLOAD = b"lmcache-mp-fs-l2-partial-hit"


class TraceCase:
    def __init__(self, request_type: RequestType, payloads: list[object]) -> None:
        self.request_type = request_type
        self.payloads = payloads


@dataclass(frozen=True)
class CudaKvTraceLayout:
    name: str
    kv_layout: str
    shape: tuple[int, ...]
    extra_shapes: tuple[tuple[int, ...], ...] = ()

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return (self.shape, *self.extra_shapes)


_CUDA_KV_TRACE_LAYOUTS = {
    "NHD": CudaKvTraceLayout("NHD", "NHD", (2, 6, 16, 1, 8)),
    "COMPACT_NHD": CudaKvTraceLayout("COMPACT_NHD", "NHD", (2, 6, 16, 8)),
    "COMPRESSED_NHD": CudaKvTraceLayout("COMPRESSED_NHD", "NHD", (2, 6, 8, 1, 8)),
    "HETEROGENEOUS_NHD": CudaKvTraceLayout(
        "HETEROGENEOUS_NHD",
        "NHD",
        (2, 6, 16, 1, 8),
        ((2, 6, 16, 2, 8),),
    ),
    "MIXED_COMPRESSION_NHD": CudaKvTraceLayout(
        "MIXED_COMPRESSION_NHD",
        "NHD",
        (2, 6, 16, 1, 8),
        ((2, 6, 8, 1, 8),),
    ),
    "HND": CudaKvTraceLayout("HND", "HND", (2, 6, 1, 16, 8)),
    "LARGE_NHD": CudaKvTraceLayout("LARGE_NHD", "NHD", (2, 8, 16, 4, 16)),
    "MULTI_CHUNK_NHD": CudaKvTraceLayout(
        "MULTI_CHUNK_NHD",
        "NHD",
        (2, 10, 16, 1, 8),
    ),
    "WIDE_NHD": CudaKvTraceLayout("WIDE_NHD", "NHD", (2, 10, 16, 8, 16)),
    "WIDE_HND": CudaKvTraceLayout("WIDE_HND", "HND", (2, 10, 8, 16, 16)),
    "CROSS_LAYER_NHD": CudaKvTraceLayout(
        "CROSS_LAYER_NHD",
        "NHD",
        (6, 4, 2, 16, 1, 8),
    ),
    "CROSS_LAYER_HND": CudaKvTraceLayout(
        "CROSS_LAYER_HND",
        "HND",
        (6, 4, 2, 1, 16, 8),
    ),
    "TRTLLM_4D": CudaKvTraceLayout("TRTLLM_4D", "HND", (6, 4, 2, 1 * 16 * 8)),
    "MLA": CudaKvTraceLayout("MLA", "NHD", (6, 16, 8)),
}
_CUDA_KV_CAPTURE_LAYOUTS = {
    "CROSS_LAYER_HND",
    "CROSS_LAYER_NHD",
    "COMPACT_NHD",
    "COMPRESSED_NHD",
    "HETEROGENEOUS_NHD",
    "HND",
    "LARGE_NHD",
    "MULTI_CHUNK_NHD",
    "MIXED_COMPRESSION_NHD",
    "MLA",
    "NHD",
    "TRTLLM_4D",
    "WIDE_HND",
    "WIDE_NHD",
}


def _cuda_kv_trace_layout(name: str) -> CudaKvTraceLayout:
    return _CUDA_KV_TRACE_LAYOUTS[name.upper()]


def _cuda_kv_capture_layout_names(
    cuda_kv_layout: str,
    *,
    include_raw_cuda_kv: bool,
) -> list[str]:
    layout = cuda_kv_layout.upper()
    if layout != "ALL":
        return [layout]
    if include_raw_cuda_kv:
        raise RuntimeError("--cuda-kv-layout ALL is only supported for PyTorch CUDA KV")
    return sorted(_CUDA_KV_CAPTURE_LAYOUTS)


def _cuda_kv_block_slice(
    layout: CudaKvTraceLayout,
    blocks: list[int],
) -> tuple[slice, ...]:
    block_start = blocks[0]
    block_end = blocks[-1] + 1
    if layout.name in {
        "COMPACT_NHD",
        "COMPRESSED_NHD",
        "HETEROGENEOUS_NHD",
        "LARGE_NHD",
        "MULTI_CHUNK_NHD",
        "MIXED_COMPRESSION_NHD",
        "NHD",
        "HND",
        "WIDE_NHD",
        "WIDE_HND",
    }:
        return (slice(None), slice(block_start, block_end))
    if layout.name in {"CROSS_LAYER_HND", "CROSS_LAYER_NHD", "MLA", "TRTLLM_4D"}:
        return (slice(block_start, block_end),)
    raise ValueError(f"unsupported CUDA KV trace layout: {layout.name!r}")


def _cuda_kv_engine_type(layout: CudaKvTraceLayout) -> EngineType:
    if layout.name == "TRTLLM_4D":
        return EngineType.TRTLLM
    return EngineType.VLLM


def _cuda_kv_layout_hints(
    layout: CudaKvTraceLayout,
    *,
    include_layerwise_hint: bool = False,
) -> dict[str, object]:
    hints: dict[str, object] = {"kv_layout": layout.kv_layout}
    if layout.name == "TRTLLM_4D":
        hints.update(
            {
                "num_kv_heads": 1,
                "tokens_per_block": 16,
                "head_dim": 8,
            }
        )
    else:
        hints["inference_engine_logical_block_size"] = 16
    if include_layerwise_hint:
        hints["use_layerwise"] = True
    return hints


def _cuda_kv_token_count(layout: CudaKvTraceLayout) -> int:
    if layout.name == "MULTI_CHUNK_NHD":
        return _KV_TRACE_CHUNK_SIZE * 2
    return _KV_TRACE_CHUNK_SIZE


def _cuda_kv_store_blocks(layout: CudaKvTraceLayout) -> list[int]:
    if layout.name == "MULTI_CHUNK_NHD":
        return [1, 2, 3, 4]
    return _KV_TRACE_STORE_BLOCKS


def _cuda_kv_retrieve_blocks(layout: CudaKvTraceLayout) -> list[int]:
    if layout.name == "MULTI_CHUNK_NHD":
        return [5, 6, 7, 8]
    return _KV_TRACE_RETRIEVE_BLOCKS


def _cuda_kv_request_id(*, use_raw_cuda_ipc: bool, layout: CudaKvTraceLayout) -> str:
    if layout.name == "NHD":
        if use_raw_cuda_ipc:
            return _KV_TRACE_RAW_REQUEST_ID
        return _KV_TRACE_PYTORCH_REQUEST_ID
    mode = "raw" if use_raw_cuda_ipc else "pytorch"
    return f"trace-kv-{mode}-cuda-{layout.name.lower()}-1"


def _cuda_kv_instance_id(
    *,
    use_raw_cuda_ipc: bool,
    layout: CudaKvTraceLayout,
) -> int:
    if use_raw_cuda_ipc:
        return _KV_TRACE_INSTANCE_ID + 1
    if layout.name == "HND":
        return _KV_TRACE_INSTANCE_ID + 2
    if layout.name == "LARGE_NHD":
        return _KV_TRACE_INSTANCE_ID + 6
    if layout.name == "MULTI_CHUNK_NHD":
        return _KV_TRACE_INSTANCE_ID + 14
    if layout.name == "COMPACT_NHD":
        return _KV_TRACE_INSTANCE_ID + 9
    if layout.name == "COMPRESSED_NHD":
        return _KV_TRACE_INSTANCE_ID + 11
    if layout.name == "MIXED_COMPRESSION_NHD":
        return _KV_TRACE_INSTANCE_ID + 12
    if layout.name == "HETEROGENEOUS_NHD":
        return _KV_TRACE_INSTANCE_ID + 10
    if layout.name == "WIDE_NHD":
        return _KV_TRACE_INSTANCE_ID + 7
    if layout.name == "WIDE_HND":
        return _KV_TRACE_INSTANCE_ID + 8
    if layout.name == "MLA":
        return _KV_TRACE_INSTANCE_ID + 3
    if layout.name == "CROSS_LAYER_NHD":
        return _KV_TRACE_INSTANCE_ID + 4
    if layout.name == "CROSS_LAYER_HND":
        return _KV_TRACE_INSTANCE_ID + 5
    if layout.name == "TRTLLM_4D":
        return _KV_TRACE_INSTANCE_ID + 13
    return _KV_TRACE_INSTANCE_ID


def _cuda_kv_cache_salt(
    *,
    use_raw_cuda_ipc: bool,
    layout: CudaKvTraceLayout,
) -> str:
    mode = "raw" if use_raw_cuda_ipc else "pytorch"
    return f"trace-cuda-{mode}-{layout.name.lower()}"


def _lookup_key() -> IPCCacheEngineKey:
    return IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=1,
        worker_id=None,
        token_ids=list(range(128)),
        request_id="trace-lookup-1",
        cache_salt="trace-tenant",
    )


def _fs_l2_partial_hit_key() -> IPCCacheEngineKey:
    return IPCCacheEngineKey.from_token_ids(
        model_name=_KV_TRACE_MODEL,
        world_size=1,
        worker_id=None,
        token_ids=list(range(_KV_TRACE_CHUNK_SIZE * 2)),
        request_id=_FS_L2_PARTIAL_HIT_REQUEST_ID,
        cache_salt="trace-fs-l2-partial",
    )


def _fs_l2_filename(key: ObjectKey) -> str:
    safe_model = key.model_name.replace("/", "-SEP-")
    base = f"{safe_model}@{key.kv_rank:#010x}@{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}@{key.cache_salt}.data"
    return f"{base}.data"


def _fs_l2_seed_row() -> dict[str, object]:
    lookup_key = _fs_l2_partial_hit_key()
    hasher = TokenHasher(chunk_size=_KV_TRACE_CHUNK_SIZE, hash_algorithm="blake3")
    object_keys = ipc_key_to_object_keys(
        lookup_key,
        hasher.compute_chunk_hashes(list(lookup_key.token_ids)),
    )
    if len(object_keys) != 2:
        raise RuntimeError("FS-L2 partial-hit trace expected exactly two object keys")
    return {
        "kind": "fs_l2_seed",
        "chunk_size": _KV_TRACE_CHUNK_SIZE,
        "files": [
            {
                "filename": _fs_l2_filename(object_keys[0]),
                "payload_hex": _FS_L2_PARTIAL_HIT_PAYLOAD.hex(),
            }
        ],
    }


def _write_fs_l2_seed(base_path: Path, row: dict[str, object]) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    files = row["files"]
    if not isinstance(files, list):
        raise TypeError("fs_l2_seed files must be a list")
    for item in files:
        if not isinstance(item, dict):
            raise TypeError("fs_l2_seed file entries must be dicts")
        filename = item["filename"]
        payload_hex = item["payload_hex"]
        if not isinstance(filename, str):
            raise TypeError("fs_l2_seed filename must be a string")
        if not isinstance(payload_hex, str):
            raise TypeError("fs_l2_seed payload_hex must be a string")
        path = base_path / filename
        if path.parent != base_path:
            raise ValueError(f"invalid fs_l2_seed filename: {filename!r}")
        path.write_bytes(bytes.fromhex(payload_hex))


def _fs_l2_partial_hit_cases() -> list[TraceCase]:
    lookup_key = _fs_l2_partial_hit_key()
    return [
        TraceCase(RequestType.LOOKUP, [lookup_key, 1]),
        TraceCase(
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            [_FS_L2_PARTIAL_HIT_REQUEST_ID],
        ),
        TraceCase(RequestType.END_SESSION, [_FS_L2_PARTIAL_HIT_REQUEST_ID]),
    ]


def _trace_cases() -> list[TraceCase]:
    return [
        TraceCase(RequestType.PING, []),
        TraceCase(RequestType.GET_CHUNK_SIZE, []),
        TraceCase(RequestType.NOOP, []),
        TraceCase(RequestType.CLEAR, []),
        TraceCase(
            RequestType.REPORT_BLOCK_ALLOCATION,
            [
                7,
                "facebook/opt-125m",
                [
                    BlockAllocationRecord(
                        req_id="trace-report-1",
                        new_block_ids=[1, 2],
                        new_token_ids=[10, 11, 12],
                    )
                ],
            ],
        ),
        TraceCase(RequestType.LOOKUP, [_lookup_key(), 1]),
        TraceCase(RequestType.QUERY_PREFETCH_LOOKUP_HITS, ["trace-lookup-1"]),
        TraceCase(RequestType.QUERY_PREFETCH_STATUS, ["trace-lookup-1"]),
        TraceCase(RequestType.END_SESSION, ["trace-lookup-1"]),
    ]


def _native_status_expectation() -> dict[str, object]:
    return {
        "kind": "native_status_expectation",
        "last_block_allocation": {
            "instance_id": 7,
            "last_new_block_count": 2,
            "last_new_token_count": 3,
            "last_request_id": "trace-report-1",
            "model_name": "facebook/opt-125m",
            "record_count": 1,
        },
        "metrics": {
            "block_allocation_report_count": 1,
            "block_allocation_record_count": 1,
        },
    }


def _payload_to_json(payload: object) -> TracePayload:
    if isinstance(payload, IPCCacheEngineKey):
        return {
            "type": "IPCCacheEngineKey",
            "model_name": payload.model_name,
            "world_size": payload.world_size,
            "worker_id": payload.worker_id,
            "token_ids": list(payload.token_ids),
            "start": payload.start,
            "end": payload.end,
            "request_id": payload.request_id,
            "cache_salt": payload.cache_salt,
        }
    if isinstance(payload, list) and all(
        isinstance(record, BlockAllocationRecord) for record in payload
    ):
        return {
            "type": "BlockAllocationRecordList",
            "records": [
                {
                    "req_id": record.req_id,
                    "new_block_ids": record.new_block_ids,
                    "new_token_ids": record.new_token_ids,
                }
                for record in payload
            ],
        }
    if isinstance(payload, int | str):
        return payload
    raise TypeError(f"unsupported trace payload type: {type(payload)!r}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(
    server: str,
    port: int,
    *,
    chunk_size: int,
    enable_cuda: bool = False,
    l2_dir: Path | None = None,
) -> list[str]:
    if server == "native":
        cmd = [
            str(ensure_native_binary(enable_cuda=enable_cuda)),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(_free_port()),
            "--l1-size-gb",
            "0.001",
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            str(chunk_size),
            "--disable-http",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--l1-size-gb",
            "0.001",
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            str(chunk_size),
            "--disable-observability",
        ]
    if l2_dir is not None:
        cmd.extend(
            [
                "--l2-adapter",
                json.dumps({"type": "fs", "base_path": str(l2_dir)}),
            ]
        )
    return cmd


def _wait_for_ping(port: int) -> None:
    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
    deadline = time.time() + 10
    last_error: Exception | None = None
    try:
        while time.time() < deadline:
            try:
                if client.submit_request(RequestType.PING, []).result(timeout=1):
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.1)
    finally:
        client.close()
    raise RuntimeError(f"server did not respond to PING: {last_error}")


def _require_cuda_trace_dependencies(*, include_raw_cuda_kv: bool) -> None:
    # Third Party
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA KV trace capture requires CUDA")
    if not include_raw_cuda_kv:
        return
    try:
        # Third Party
        import cupy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("--include-raw-cuda-kv requires cupy") from exc


def _cuda_kv_tensors(kv_cache: object) -> list["torch.Tensor"]:
    # Third Party
    import torch

    if isinstance(kv_cache, list):
        tensors = kv_cache
    else:
        tensors = [kv_cache]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("expected CUDA KV cache to contain torch.Tensor values")
    return cast(list[torch.Tensor], tensors)


def _cuda_kv_slice_sha256(kv_cache: object, selection: tuple[slice, ...]) -> str:
    # Third Party

    digest = hashlib.sha256()
    for tensor in _cuda_kv_tensors(kv_cache):
        digest.update(tensor[selection].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _clone_cuda_kv_slice(kv_cache: object, selection: tuple[slice, ...]) -> object:
    tensors = _cuda_kv_tensors(kv_cache)
    clones = [tensor[selection].clone() for tensor in tensors]
    return clones if isinstance(kv_cache, list) else clones[0]


def _zero_cuda_kv_slice(kv_cache: object, selection: tuple[slice, ...]) -> None:
    for tensor in _cuda_kv_tensors(kv_cache):
        tensor[selection].zero_()


def _assert_cuda_kv_close(
    kv_cache: object,
    selection: tuple[slice, ...],
    expected: object,
) -> None:
    # Third Party
    import torch

    actual_tensors = _cuda_kv_tensors(kv_cache)
    expected_tensors = expected if isinstance(expected, list) else [expected]
    for actual, expected_tensor in zip(actual_tensors, expected_tensors, strict=True):
        torch.testing.assert_close(actual[selection], expected_tensor)


def _make_cuda_kv_cache(
    layout: CudaKvTraceLayout,
    *,
    use_raw_cuda_ipc: bool,
) -> object:
    # Third Party
    import torch

    torch.cuda.set_device(0)

    def make_one(shape: tuple[int, ...], offset: int) -> object:
        numel = math.prod(shape)
        if use_raw_cuda_ipc:
            # Third Party
            import cupy

            cupy_cache = (cupy.arange(numel, dtype=cupy.float16) + offset).reshape(
                shape
            )
            return torch.from_dlpack(cupy_cache)
        return (
            torch.arange(
                numel,
                device="cuda",
                dtype=torch.float16,
            )
            + offset
        ).reshape(shape)

    if layout.extra_shapes:
        return [
            make_one(shape, index * 4096) for index, shape in enumerate(layout.shapes)
        ]
    return make_one(layout.shape, 0)


def _capture_cuda_kv_case(
    client: MessageQueueClient,
    *,
    use_raw_cuda_ipc: bool,
    cuda_kv_layout: str,
    include_layerwise_hint: bool,
    cycle_index: int = 0,
    cycle_count: int = 1,
) -> dict[str, object]:
    # Third Party
    import torch

    layout = _cuda_kv_trace_layout(cuda_kv_layout)
    store_blocks = _cuda_kv_store_blocks(layout)
    retrieve_blocks = _cuda_kv_retrieve_blocks(layout)
    token_count = _cuda_kv_token_count(layout)
    kv_cache = cast(
        torch.Tensor,
        _make_cuda_kv_cache(layout, use_raw_cuda_ipc=use_raw_cuda_ipc),
    )
    source_slice = _cuda_kv_block_slice(layout, store_blocks)
    retrieve_slice = _cuda_kv_block_slice(layout, retrieve_blocks)
    expected = _clone_cuda_kv_slice(kv_cache, source_slice)
    _zero_cuda_kv_slice(kv_cache, retrieve_slice)
    event = torch.cuda.Event(interprocess=True)
    event.record()
    torch.cuda.synchronize()

    request_id = _cuda_kv_request_id(
        use_raw_cuda_ipc=use_raw_cuda_ipc,
        layout=layout,
    )
    cache_salt = _cuda_kv_cache_salt(
        use_raw_cuda_ipc=use_raw_cuda_ipc,
        layout=layout,
    )
    instance_id = _cuda_kv_instance_id(
        use_raw_cuda_ipc=use_raw_cuda_ipc,
        layout=layout,
    )
    if cycle_index:
        suffix = f"cycle-{cycle_index + 1}"
        request_id = f"{request_id}-{suffix}"
        cache_salt = f"{cache_salt}-{suffix}"
        instance_id += cycle_index * 1000

    worker_key = IPCCacheEngineKey.from_token_ids(
        model_name=_KV_TRACE_MODEL,
        world_size=1,
        worker_id=0,
        token_ids=list(range(token_count)),
        start=0,
        end=token_count,
        request_id=request_id,
        cache_salt=cache_salt,
    )
    lookup_key = worker_key.no_worker_id_version()
    wrapper_cls: type[CudaIPCWrapper] = (
        RawCudaIPCWrapper if use_raw_cuda_ipc else CudaIPCWrapper
    )

    assert (
        client.submit_request(
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                [wrapper_cls(tensor) for tensor in _cuda_kv_tensors(kv_cache)],
                _KV_TRACE_MODEL,
                1,
                _cuda_kv_engine_type(layout),
                _cuda_kv_layout_hints(
                    layout,
                    include_layerwise_hint=include_layerwise_hint,
                ),
            ],
        ).result(timeout=30)
        is None
    )
    store_response = cast(
        tuple[bytes, bool],
        client.submit_request(
            RequestType.STORE,
            [
                worker_key,
                instance_id,
                store_blocks,
                event.ipc_handle(),
            ],
        ).result(timeout=30),
    )
    store_event, store_ok = store_response
    assert store_ok
    assert store_event

    assert (
        client.submit_request(RequestType.LOOKUP, [lookup_key, 1]).result(timeout=10)
        is None
    )
    prefetch_status = None
    for _ in range(20):
        prefetch_status = client.submit_request(
            RequestType.QUERY_PREFETCH_STATUS,
            [worker_key.request_id],
        ).result(timeout=5)
        if prefetch_status is not None:
            break
        time.sleep(0.1)
    expected_prefetch_status = token_count // _KV_TRACE_CHUNK_SIZE
    if prefetch_status != expected_prefetch_status:
        raise AssertionError(
            f"expected {expected_prefetch_status} prefetched chunks, "
            f"got {prefetch_status!r}"
        )

    _zero_cuda_kv_slice(kv_cache, retrieve_slice)
    torch.cuda.synchronize()
    retrieve_response = cast(
        tuple[bytes, bool],
        client.submit_request(
            RequestType.RETRIEVE,
            [
                worker_key,
                instance_id,
                retrieve_blocks,
                event.ipc_handle(),
                0,
            ],
        ).result(timeout=30),
    )
    retrieve_event, retrieve_ok = retrieve_response
    assert retrieve_ok
    assert retrieve_event
    torch.cuda.synchronize()
    _assert_cuda_kv_close(kv_cache, retrieve_slice, expected)
    assert (
        client.submit_request(
            RequestType.FREE_LOOKUP_LOCKS,
            [lookup_key, expected_prefetch_status],
        ).result(timeout=10)
        is None
    )
    assert (
        client.submit_request(
            RequestType.END_SESSION,
            [lookup_key.request_id],
        ).result(timeout=10)
        is None
    )
    assert (
        client.submit_request(
            RequestType.UNREGISTER_KV_CACHE,
            [instance_id],
        ).result(timeout=10)
        is None
    )

    row: dict[str, object] = {
        "kind": (
            "raw_cuda_kv_roundtrip" if use_raw_cuda_ipc else "pytorch_cuda_kv_roundtrip"
        ),
        "chunk_size": _KV_TRACE_CHUNK_SIZE,
        "token_count": token_count,
        "instance_id": instance_id,
        "model_name": _KV_TRACE_MODEL,
        "layout": layout.name,
        "engine_type": _cuda_kv_engine_type(layout).value,
        "layout_hints": _cuda_kv_layout_hints(
            layout,
            include_layerwise_hint=include_layerwise_hint,
        ),
        "kv_layout": layout.kv_layout,
        "shape": list(layout.shape),
        "shapes": [list(shape) for shape in layout.shapes],
        "dtype": "torch.float16",
        "store_blocks": store_blocks,
        "retrieve_blocks": retrieve_blocks,
        "request_id": worker_key.request_id,
        "cache_salt": worker_key.cache_salt,
        "prefetch_status": prefetch_status,
        "expect_lookup_lock_after_retrieve": True,
        "free_lookup_locks": True,
        "end_session": True,
        "unregister_kv_cache": True,
        "sha256": _cuda_kv_slice_sha256(kv_cache, retrieve_slice),
    }
    if cycle_count > 1:
        row["lifecycle_cycle"] = cycle_index + 1
        row["lifecycle_cycles"] = cycle_count
    return row


def _capture_cuda_kv_layout_cases(
    server: str,
    *,
    use_raw_cuda_ipc: bool,
    cuda_kv_layout: str,
    include_layerwise_hint: bool,
    cuda_kv_lifecycle_cycles: int,
) -> list[dict[str, object]]:
    port = _free_port()
    proc = subprocess.Popen(
        _server_command(
            server,
            port,
            chunk_size=_KV_TRACE_CHUNK_SIZE,
            enable_cuda=True,
        ),
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_ping(port)
        context = zmq.Context.instance()
        client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
        try:
            return [
                _capture_cuda_kv_case(
                    client,
                    use_raw_cuda_ipc=use_raw_cuda_ipc,
                    cuda_kv_layout=cuda_kv_layout,
                    include_layerwise_hint=include_layerwise_hint,
                    cycle_index=cycle_index,
                    cycle_count=cuda_kv_lifecycle_cycles,
                )
                for cycle_index in range(cuda_kv_lifecycle_cycles)
            ]
        finally:
            client.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def capture_trace(
    server: str,
    output: Path,
    *,
    include_pytorch_cuda_kv: bool,
    include_raw_cuda_kv: bool,
    cuda_kv_layout: str,
    include_fs_l2_partial_hit: bool,
    include_layerwise_hint: bool,
    cuda_kv_lifecycle_cycles: int,
) -> None:
    if cuda_kv_lifecycle_cycles < 1:
        raise RuntimeError("--cuda-kv-lifecycle-cycles must be at least 1")
    include_cuda_kv = include_pytorch_cuda_kv or include_raw_cuda_kv
    cuda_kv_layouts = _cuda_kv_capture_layout_names(
        cuda_kv_layout,
        include_raw_cuda_kv=include_raw_cuda_kv,
    )
    if include_fs_l2_partial_hit and not include_cuda_kv:
        raise RuntimeError(
            "--include-fs-l2-partial-hit requires a CUDA KV trace so the "
            "Python server has registered KV layout metadata"
        )
    if include_layerwise_hint and not include_cuda_kv:
        raise RuntimeError(
            "--include-layerwise-hint requires a CUDA KV trace so the "
            "registration row can carry CUDA layout metadata"
        )
    if include_cuda_kv:
        _require_cuda_trace_dependencies(include_raw_cuda_kv=include_raw_cuda_kv)
    isolate_cuda_layouts = cuda_kv_lifecycle_cycles > 1 and len(cuda_kv_layouts) > 1
    port = _free_port()
    fs_l2_seed = _fs_l2_seed_row() if include_fs_l2_partial_hit else None
    l2_temp_dir = (
        tempfile.TemporaryDirectory(prefix="lmcache-mp-fs-l2-")
        if include_fs_l2_partial_hit
        else None
    )
    l2_dir = Path(l2_temp_dir.name) if l2_temp_dir is not None else None
    try:
        if fs_l2_seed is not None and l2_dir is not None:
            _write_fs_l2_seed(l2_dir, fs_l2_seed)
        proc = subprocess.Popen(
            _server_command(
                server,
                port,
                chunk_size=_KV_TRACE_CHUNK_SIZE if include_cuda_kv else 256,
                enable_cuda=include_cuda_kv,
                l2_dir=l2_dir,
            ),
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_ping(port)
            context = zmq.Context.instance()
            client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("w", encoding="utf-8") as out:
                    if fs_l2_seed is not None:
                        out.write(json.dumps(fs_l2_seed, sort_keys=True))
                        out.write("\n")
                    for case in _trace_cases():
                        response = cast(
                            TraceResponse,
                            client.submit_request(
                                case.request_type,
                                case.payloads,
                            ).result(timeout=5),
                        )
                        out.write(
                            json.dumps(
                                {
                                    "request_type": case.request_type.name,
                                    "payloads": [
                                        _payload_to_json(payload)
                                        for payload in case.payloads
                                    ],
                                    "response": response,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    out.write(json.dumps(_native_status_expectation(), sort_keys=True))
                    out.write("\n")
                    if include_raw_cuda_kv:
                        for layout in cuda_kv_layouts:
                            if isolate_cuda_layouts:
                                rows = _capture_cuda_kv_layout_cases(
                                    server,
                                    use_raw_cuda_ipc=True,
                                    cuda_kv_layout=layout,
                                    include_layerwise_hint=include_layerwise_hint,
                                    cuda_kv_lifecycle_cycles=(cuda_kv_lifecycle_cycles),
                                )
                            else:
                                rows = [
                                    _capture_cuda_kv_case(
                                        client,
                                        use_raw_cuda_ipc=True,
                                        cuda_kv_layout=layout,
                                        include_layerwise_hint=(include_layerwise_hint),
                                        cycle_index=cycle_index,
                                        cycle_count=cuda_kv_lifecycle_cycles,
                                    )
                                    for cycle_index in range(cuda_kv_lifecycle_cycles)
                                ]
                            for row in rows:
                                out.write(json.dumps(row, sort_keys=True) + "\n")
                    if include_pytorch_cuda_kv:
                        for layout in cuda_kv_layouts:
                            if isolate_cuda_layouts:
                                rows = _capture_cuda_kv_layout_cases(
                                    server,
                                    use_raw_cuda_ipc=False,
                                    cuda_kv_layout=layout,
                                    include_layerwise_hint=include_layerwise_hint,
                                    cuda_kv_lifecycle_cycles=(cuda_kv_lifecycle_cycles),
                                )
                            else:
                                rows = [
                                    _capture_cuda_kv_case(
                                        client,
                                        use_raw_cuda_ipc=False,
                                        cuda_kv_layout=layout,
                                        include_layerwise_hint=(include_layerwise_hint),
                                        cycle_index=cycle_index,
                                        cycle_count=cuda_kv_lifecycle_cycles,
                                    )
                                    for cycle_index in range(cuda_kv_lifecycle_cycles)
                                ]
                            for row in rows:
                                out.write(json.dumps(row, sort_keys=True) + "\n")
                    if include_fs_l2_partial_hit:
                        for case in _fs_l2_partial_hit_cases():
                            response = cast(
                                TraceResponse,
                                client.submit_request(
                                    case.request_type,
                                    case.payloads,
                                ).result(timeout=5),
                            )
                            if (
                                case.request_type
                                == RequestType.QUERY_PREFETCH_LOOKUP_HITS
                            ):
                                for _ in range(20):
                                    if response is not None:
                                        break
                                    time.sleep(0.1)
                                    response = cast(
                                        TraceResponse,
                                        client.submit_request(
                                            case.request_type,
                                            case.payloads,
                                        ).result(timeout=5),
                                    )
                            out.write(
                                json.dumps(
                                    {
                                        "request_type": case.request_type.name,
                                        "payloads": [
                                            _payload_to_json(payload)
                                            for payload in case.payloads
                                        ],
                                        "response": response,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
            finally:
                client.close()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        if l2_temp_dir is not None:
            l2_temp_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=["python", "native"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-pytorch-cuda-kv",
        action="store_true",
        help=("Append a PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE byte round-trip case."),
    )
    parser.add_argument(
        "--include-raw-cuda-kv",
        action="store_true",
        help="Append a raw CUDA IPC STORE/LOOKUP/RETRIEVE byte round-trip case.",
    )
    parser.add_argument(
        "--cuda-kv-layout",
        choices=[*sorted(_CUDA_KV_CAPTURE_LAYOUTS), "ALL"],
        default="NHD",
        help=(
            "CUDA KV tensor layout to use when appending CUDA KV trace cases. "
            "Use ALL to append every PyTorch CUDA layout."
        ),
    )
    parser.add_argument(
        "--include-fs-l2-partial-hit",
        action="store_true",
        help=(
            "Append an FS-L2 lookup with one seeded chunk hit and one miss. "
            "Requires a CUDA KV trace registration."
        ),
    )
    parser.add_argument(
        "--include-layerwise-hint",
        action="store_true",
        help="Include use_layerwise=True in CUDA KV registration layout hints.",
    )
    parser.add_argument(
        "--cuda-kv-lifecycle-cycles",
        type=int,
        default=1,
        help=(
            "Number of independent STORE/LOOKUP/RETRIEVE/FREE/END_SESSION "
            "CUDA KV cycles to append for each selected layout."
        ),
    )
    args = parser.parse_args()
    capture_trace(
        args.server,
        args.output,
        include_pytorch_cuda_kv=args.include_pytorch_cuda_kv,
        include_raw_cuda_kv=args.include_raw_cuda_kv,
        cuda_kv_layout=args.cuda_kv_layout,
        include_fs_l2_partial_hit=args.include_fs_l2_partial_hit,
        include_layerwise_hint=args.include_layerwise_hint,
        cuda_kv_lifecycle_cycles=args.cuda_kv_lifecycle_cycles,
    )


if __name__ == "__main__":
    main()

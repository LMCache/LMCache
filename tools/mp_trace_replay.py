#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay an LMCache MP golden trace against Python or native."""

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
import urllib.request

# Third Party
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CudaIPCWrapper,
    IPCCacheEngineKey,
    RawCudaIPCWrapper,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
from lmcache.v1.multiprocess.protocol import RequestType

if TYPE_CHECKING:
    # Third Party
    import torch

TraceResponse = bool | int | str | None
TracePayload = int | str | dict[str, object]

_KV_TRACE_INSTANCE_ID = 4321
_CUDA_KV_TRACE_KINDS = {"pytorch_cuda_kv_roundtrip", "raw_cuda_kv_roundtrip"}
_VLLM_KVCACHE_CHECKSUM_KIND = "vllm_kvcache_checksum_match"


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


def _json_int(payload: dict[str, object], name: str) -> int:
    value = payload[name]
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _json_str(payload: dict[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    return value


def _json_int_list(payload: dict[str, object], name: str) -> list[int]:
    value = payload[name]
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return [int(item) for item in value]


def _json_optional_str(
    payload: dict[str, object],
    name: str,
    default: str,
) -> str:
    value = payload.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    return value


def _json_optional_int(
    payload: dict[str, object],
    name: str,
    default: int,
) -> int:
    value = payload.get(name, default)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _json_dict(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    return cast(dict[str, object], value)


def _json_dict_list(payload: dict[str, object], name: str) -> list[dict[str, object]]:
    value = payload[name]
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{name} must be a list of dicts")
    return [cast(dict[str, object], item) for item in value]


def _cuda_kv_trace_layout(row: dict[str, object]) -> CudaKvTraceLayout:
    layout_name = _json_optional_str(row, "layout", "NHD").upper()
    if layout_name not in _CUDA_KV_TRACE_LAYOUTS:
        raise TypeError(f"unsupported CUDA trace layout: {layout_name!r}")
    layout = _CUDA_KV_TRACE_LAYOUTS[layout_name]
    shape = _json_int_list(row, "shape")
    if shape != list(layout.shape):
        raise TypeError(
            f"CUDA trace shape {shape!r} does not match "
            f"{layout.name} shape {list(layout.shape)!r}"
        )
    shapes = row.get("shapes")
    if shapes is not None:
        if not isinstance(shapes, list):
            raise TypeError("CUDA trace shapes must be a list")
        parsed_shapes = [[int(dim) for dim in item] for item in shapes]
        expected_shapes = [list(item) for item in layout.shapes]
        if parsed_shapes != expected_shapes:
            raise TypeError(
                f"CUDA trace shapes {parsed_shapes!r} do not match "
                f"{layout.name} shapes {expected_shapes!r}"
            )
    elif layout.extra_shapes:
        raise TypeError(f"CUDA trace layout {layout.name} requires shapes metadata")
    return layout


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


def _cuda_kv_engine_type(row: dict[str, object]) -> EngineType:
    value = _json_optional_str(row, "engine_type", EngineType.VLLM.value)
    return EngineType(value)


def _cuda_kv_layout_hints(
    row: dict[str, object],
    layout: CudaKvTraceLayout,
    kv_layout: str,
) -> dict[str, object]:
    hints = row.get("layout_hints")
    if hints is not None:
        if not isinstance(hints, dict):
            raise TypeError("CUDA trace layout_hints must be a dict")
        return cast(dict[str, object], hints)
    default_hints: dict[str, object] = {"kv_layout": kv_layout}
    if layout.name == "TRTLLM_4D":
        default_hints.update(
            {
                "num_kv_heads": 1,
                "tokens_per_block": 16,
                "head_dim": 8,
            }
        )
    else:
        default_hints["inference_engine_logical_block_size"] = 16
    return default_hints


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _trace_requires_cuda(rows: list[dict[str, object]]) -> bool:
    return any(row.get("kind") in _CUDA_KV_TRACE_KINDS for row in rows)


def _trace_requires_raw_cuda(rows: list[dict[str, object]]) -> bool:
    return any(row.get("kind") == "raw_cuda_kv_roundtrip" for row in rows)


def _trace_requires_native_http(rows: list[dict[str, object]]) -> bool:
    return any(
        row.get("kind") == "native_status_expectation"
        or bool(row.get("expect_lookup_lock_after_retrieve"))
        for row in rows
    )


def _trace_requires_fs_l2(rows: list[dict[str, object]]) -> bool:
    return any(row.get("kind") == "fs_l2_seed" for row in rows)


def _trace_chunk_size(rows: list[dict[str, object]]) -> int:
    chunk_size = 256
    for row in rows:
        if row.get("kind") in _CUDA_KV_TRACE_KINDS:
            chunk_size = _json_int(row, "chunk_size")
            break
        if row.get("kind") == "fs_l2_seed":
            chunk_size = _json_int(row, "chunk_size")
    return chunk_size


def _write_fs_l2_seed(base_path: Path, rows: list[dict[str, object]]) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("kind") != "fs_l2_seed":
            continue
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


def _checksum_response(probe: dict[str, object]) -> dict[str, object]:
    return _json_dict(probe, "response")


def _assert_vllm_kvcache_checksum_row(row: dict[str, object]) -> None:
    writer_probe = _json_dict(row, "writer")
    readers = _json_dict_list(row, "readers")
    if writer_probe.get("request_type") != "STORE":
        raise AssertionError(f"writer checksum probe is not STORE: {writer_probe!r}")
    writer_response = _checksum_response(writer_probe)
    if not readers:
        raise AssertionError("checksum trace row does not include reader probes")
    for reader_index, reader_probe in enumerate(readers):
        if reader_probe.get("request_type") != "RETRIEVE":
            raise AssertionError(
                f"reader {reader_index} checksum probe is not RETRIEVE: "
                f"{reader_probe!r}"
            )
        reader_response = _checksum_response(reader_probe)
        for key in ("chunk_size", "num_chunks", "block_id_ranges"):
            if writer_response.get(key) != reader_response.get(key):
                raise AssertionError(
                    f"reader {reader_index} checksum metadata differs for {key}: "
                    f"writer={writer_response!r}, reader={reader_response!r}"
                )
        if writer_response.get("chunk_checksums") != reader_response.get(
            "chunk_checksums"
        ):
            raise AssertionError(
                f"reader {reader_index} checksum mismatch: "
                f"writer={writer_response!r}, reader={reader_response!r}"
            )


def _server_command(
    server: str,
    port: int,
    *,
    chunk_size: int,
    enable_cuda: bool = False,
    http_port: int | None = None,
    l2_dir: Path | None = None,
) -> list[str]:
    if server == "native":
        cmd = [
            str(ensure_native_binary(enable_cuda=enable_cuda)),
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
        ]
        if http_port is None:
            cmd.append("--disable-http")
        else:
            cmd.extend(["--http-host", "127.0.0.1", "--http-port", str(http_port)])
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
    try:
        while time.time() < deadline:
            try:
                if client.submit_request(RequestType.PING, []).result(timeout=1):
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
    finally:
        client.close()
    raise RuntimeError("server did not respond to PING")


def _wait_for_http(url: str) -> None:
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"server did not expose HTTP status: {last_error}")


def _status(http_port: int) -> dict[str, object]:
    return json.loads(
        urllib.request.urlopen(
            f"http://127.0.0.1:{http_port}/status",
            timeout=5,
        ).read()
    )


def _assert_native_status(status: dict[str, object], row: dict[str, object]) -> None:
    expected_last = row["last_block_allocation"]
    if status.get("last_block_allocation") != expected_last:
        raise AssertionError(
            "native status last_block_allocation mismatch: "
            f"expected {expected_last!r}, got {status.get('last_block_allocation')!r}"
        )
    metrics = status.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("native status metrics must be a dict")
    expected_metrics = row["metrics"]
    if not isinstance(expected_metrics, dict):
        raise TypeError("native status expectation metrics must be a dict")
    for key, expected_value in expected_metrics.items():
        if metrics.get(key) != expected_value:
            raise AssertionError(
                f"native status metric {key} mismatch: "
                f"expected {expected_value!r}, got {metrics.get(key)!r}"
            )


def _payload_from_json(payload: TracePayload) -> object:
    if isinstance(payload, int | str):
        return payload
    if payload.get("type") == "IPCCacheEngineKey":
        token_ids = payload["token_ids"]
        if not isinstance(token_ids, list):
            raise TypeError("IPCCacheEngineKey token_ids must be a list")
        worker_id = payload["worker_id"]
        if worker_id is not None and not isinstance(worker_id, int):
            raise TypeError("IPCCacheEngineKey worker_id must be null or an int")
        return IPCCacheEngineKey.from_token_ids(
            model_name=_json_str(payload, "model_name"),
            world_size=_json_int(payload, "world_size"),
            worker_id=None if worker_id is None else int(worker_id),
            token_ids=[int(token_id) for token_id in token_ids],
            start=_json_int(payload, "start"),
            end=_json_int(payload, "end"),
            request_id=_json_str(payload, "request_id"),
            cache_salt=_json_str(payload, "cache_salt"),
        )
    if payload.get("type") == "BlockAllocationRecordList":
        records = payload["records"]
        if not isinstance(records, list):
            raise TypeError("BlockAllocationRecordList records must be a list")
        out = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("BlockAllocationRecord must be a dict")
            out.append(
                BlockAllocationRecord(
                    req_id=_json_str(record, "req_id"),
                    new_block_ids=_json_int_list(record, "new_block_ids"),
                    new_token_ids=_json_int_list(record, "new_token_ids"),
                )
            )
        return out
    raise TypeError(f"unsupported trace payload: {payload!r}")


def _require_cuda_trace_dependencies(*, include_raw_cuda_kv: bool) -> None:
    # Third Party
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA KV trace replay requires CUDA")
    if not include_raw_cuda_kv:
        return
    try:
        # Third Party
        import cupy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("raw CUDA KV trace replay requires cupy") from exc


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
    digest = hashlib.sha256()
    for tensor in _cuda_kv_tensors(kv_cache):
        digest.update(tensor[selection].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _zero_cuda_kv_slice(kv_cache: object, selection: tuple[slice, ...]) -> None:
    for tensor in _cuda_kv_tensors(kv_cache):
        tensor[selection].zero_()


def _cuda_kv_key(row: dict[str, object]) -> IPCCacheEngineKey:
    chunk_size = _json_int(row, "chunk_size")
    token_count = _json_optional_int(row, "token_count", chunk_size)
    return IPCCacheEngineKey.from_token_ids(
        model_name=_json_str(row, "model_name"),
        world_size=1,
        worker_id=0,
        token_ids=list(range(token_count)),
        start=0,
        end=token_count,
        request_id=_json_str(row, "request_id"),
        cache_salt=_json_optional_str(row, "cache_salt", ""),
    )


def _cuda_kv_instance_id(row: dict[str, object]) -> int:
    value = row.get("instance_id", _KV_TRACE_INSTANCE_ID)
    if not isinstance(value, int):
        raise TypeError("instance_id must be an int")
    return value


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


def _replay_cuda_kv_case(
    client: MessageQueueClient,
    row: dict[str, object],
    *,
    http_port: int | None = None,
) -> None:
    # Third Party
    import torch

    row_kind = _json_str(row, "kind")
    if row_kind not in _CUDA_KV_TRACE_KINDS:
        raise TypeError(f"unsupported CUDA KV trace kind: {row_kind!r}")
    use_raw_cuda_ipc = row_kind == "raw_cuda_kv_roundtrip"
    if _json_str(row, "dtype") != "torch.float16":
        raise TypeError("CUDA trace replay currently supports torch.float16")
    layout = _cuda_kv_trace_layout(row)
    kv_layout = _json_optional_str(row, "kv_layout", layout.kv_layout)
    store_blocks = _json_int_list(row, "store_blocks")
    retrieve_blocks = _json_int_list(row, "retrieve_blocks")
    retrieve_slice = _cuda_kv_block_slice(layout, retrieve_blocks)
    instance_id = _cuda_kv_instance_id(row)

    kv_cache = cast(
        torch.Tensor,
        _make_cuda_kv_cache(layout, use_raw_cuda_ipc=use_raw_cuda_ipc),
    )
    _zero_cuda_kv_slice(kv_cache, retrieve_slice)
    event = torch.cuda.Event(interprocess=True)
    event.record()
    torch.cuda.synchronize()

    worker_key = _cuda_kv_key(row)
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
                worker_key.model_name,
                1,
                _cuda_kv_engine_type(row),
                _cuda_kv_layout_hints(row, layout, kv_layout),
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
    if not store_ok or not store_event:
        raise AssertionError(f"{row_kind} STORE replay failed")

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
    if prefetch_status != row["prefetch_status"]:
        raise AssertionError(
            f"{row_kind} prefetch status mismatch: "
            f"expected {row['prefetch_status']!r}, got {prefetch_status!r}"
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
    if not retrieve_ok or not retrieve_event:
        raise AssertionError(f"{row_kind} RETRIEVE replay failed")
    torch.cuda.synchronize()

    actual_sha256 = _cuda_kv_slice_sha256(kv_cache, retrieve_slice)
    if actual_sha256 != row["sha256"]:
        raise AssertionError(
            f"{row_kind} byte checksum mismatch: "
            f"expected {row['sha256']!r}, got {actual_sha256!r}"
        )
    if row.get("expect_lookup_lock_after_retrieve") and http_port is not None:
        status = _status(http_port)
        cache_status = status["cache"]
        if not isinstance(cache_status, dict):
            raise TypeError("native status cache must be a dict")
        expected_lock_count = row["prefetch_status"]
        if cache_status.get("lock_count") != expected_lock_count:
            raise AssertionError(
                "native lookup lock count after RETRIEVE mismatch: "
                f"expected {expected_lock_count!r}, "
                f"got {cache_status.get('lock_count')!r}"
            )
        if cache_status.get("locked_entries") != expected_lock_count:
            raise AssertionError(
                "native locked entry count after RETRIEVE mismatch: "
                f"expected {expected_lock_count!r}, "
                f"got {cache_status.get('locked_entries')!r}"
            )
    if row.get("free_lookup_locks"):
        assert (
            client.submit_request(
                RequestType.FREE_LOOKUP_LOCKS,
                [lookup_key, row["prefetch_status"]],
            ).result(timeout=10)
            is None
        )
    if row.get("end_session"):
        assert (
            client.submit_request(
                RequestType.END_SESSION,
                [lookup_key.request_id],
            ).result(timeout=10)
            is None
        )
    if row.get("unregister_kv_cache"):
        assert (
            client.submit_request(
                RequestType.UNREGISTER_KV_CACHE,
                [instance_id],
            ).result(timeout=10)
            is None
        )


def replay_trace(server: str, input_path: Path) -> None:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    cuda_trace = _trace_requires_cuda(rows)
    raw_cuda_trace = _trace_requires_raw_cuda(rows)
    native_http_trace = server == "native" and _trace_requires_native_http(rows)
    fs_l2_trace = _trace_requires_fs_l2(rows)
    if cuda_trace:
        _require_cuda_trace_dependencies(include_raw_cuda_kv=raw_cuda_trace)
    chunk_size = _trace_chunk_size(rows)
    port = _free_port()
    http_port = _free_port() if native_http_trace else None
    l2_temp_dir = (
        tempfile.TemporaryDirectory(prefix="lmcache-mp-fs-l2-") if fs_l2_trace else None
    )
    try:
        l2_dir = Path(l2_temp_dir.name) if l2_temp_dir is not None else None
        if l2_dir is not None:
            _write_fs_l2_seed(l2_dir, rows)
        proc = subprocess.Popen(
            _server_command(
                server,
                port,
                chunk_size=chunk_size,
                enable_cuda=cuda_trace,
                http_port=http_port,
                l2_dir=l2_dir,
            ),
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_ping(port)
            if http_port is not None:
                _wait_for_http(f"http://127.0.0.1:{http_port}/healthcheck")
            context = zmq.Context.instance()
            client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
            try:
                for row in rows:
                    if row.get("kind") == "fs_l2_seed":
                        continue
                    if row.get("kind") == _VLLM_KVCACHE_CHECKSUM_KIND:
                        _assert_vllm_kvcache_checksum_row(row)
                        continue
                    if row.get("kind") in _CUDA_KV_TRACE_KINDS:
                        _replay_cuda_kv_case(client, row, http_port=http_port)
                        continue
                    if row.get("kind") == "native_status_expectation":
                        if http_port is not None:
                            _assert_native_status(_status(http_port), row)
                        continue
                    request_type = RequestType[row["request_type"]]
                    payloads = [
                        _payload_from_json(cast(TracePayload, payload))
                        for payload in row["payloads"]
                    ]
                    response = cast(
                        TraceResponse,
                        client.submit_request(request_type, payloads).result(timeout=5),
                    )
                    if response is None and row["response"] is not None:
                        for _ in range(20):
                            time.sleep(0.1)
                            response = cast(
                                TraceResponse,
                                client.submit_request(
                                    request_type,
                                    payloads,
                                ).result(timeout=5),
                            )
                            if response is not None:
                                break
                    if response != row["response"]:
                        raise AssertionError(
                            f"{request_type.name} response mismatch: "
                            f"expected {row['response']!r}, got {response!r}"
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
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    replay_trace(args.server, args.input)


if __name__ == "__main__":
    main()

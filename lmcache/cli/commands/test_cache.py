# SPDX-License-Identifier: Apache-2.0
"""``lmcache test-cache`` — end-to-end test for LMCache MP cache server.

Supports both **CPU** and **GPU** modes (``--mode cpu|gpu``).

This command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup, get prefetch job ID
      2. QUERY_PREFETCH_STATUS — poll until prefetch completes
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

Usage examples::

    # CPU mode (default): 3 requests, 512 tokens each
    lmcache test-cache --port 5555 --num-tokens 512 \\
        --start 0 --end 3

    # GPU mode: real CUDA tensors + IPC
    lmcache test-cache --port 5555 --mode gpu \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape
    lmcache test-cache --port 5555 --num-tokens 512 \\
        --num-layers 8 --num-heads 4 --head-size 64 \\
        --dtype float32

    # Run forever starting from sequence 0
    lmcache test-cache --port 5555
"""

# Future
from __future__ import annotations

# Standard
import argparse
import hashlib
import itertools
import json
import sys
import time
import urllib.error
import urllib.request

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.utils import compress_slot_mapping
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.protocols.base import RequestType

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

_HELLO_TOKEN_ID = 9906
_MODEL_NAME = "test-model"
_WORLD_SIZE = 1
_INSTANCE_ID = 0

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #

_REQUEST_UID_COUNTER = 0


def _next_uid() -> int:
    global _REQUEST_UID_COUNTER
    uid = _REQUEST_UID_COUNTER
    _REQUEST_UID_COUNTER += 1
    return uid


def _send_request(
    sock: zmq.Socket,
    request_type: RequestType,
    payloads: list[bytes] | None = None,
    timeout_ms: int = 10000,
) -> list[bytes] | None:
    """Send a request and wait for the response.

    Returns the raw response frames (excluding uid and type),
    or *None* on timeout.
    """
    uid = _next_uid()
    b_uid = msgspec.msgpack.encode(uid)
    b_type = msgspec.msgpack.encode(request_type)

    frames: list[bytes] = [b_uid, b_type]
    if payloads:
        frames.extend(payloads)

    sock.send_multipart(frames)

    if sock.poll(timeout_ms, zmq.POLLIN):
        resp = sock.recv_multipart()
        return resp[2:] if len(resp) > 2 else []
    return None


# ------------------------------------------------------------------ #
#  Token / key helpers                                                 #
# ------------------------------------------------------------------ #


def _build_token_ids(
    seq_no: int,
    num_tokens: int,
) -> tuple[int, ...]:
    """Build token sequence: ``(seq_no, hello, hello, ...)``."""
    return (seq_no,) + (_HELLO_TOKEN_ID,) * num_tokens


def _make_key(
    token_ids: tuple[int, ...],
    request_id: str,
    start: int = 0,
    end: int = 0,
    worker_id: int | None = None,
) -> IPCCacheEngineKey:
    """Build an IPCCacheEngineKey."""
    return IPCCacheEngineKey(
        model_name=_MODEL_NAME,
        world_size=_WORLD_SIZE,
        worker_id=worker_id,
        token_ids=token_ids,
        start=start,
        end=end if end > 0 else len(token_ids),
        request_id=request_id,
    )


# ------------------------------------------------------------------ #
#  Protocol operations                                                 #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  GPU KV cache allocation                                             #
# ------------------------------------------------------------------ #


def _allocate_gpu_kv_cache(
    num_layers: int = 32,
    num_heads: int = 8,
    head_size: int = 128,
    num_blocks: int = 1024,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
) -> list[torch.Tensor]:
    """Allocate paged GPU KV cache tensors.

    Each layer is a tensor of shape
    ``(2, num_blocks, block_size, num_heads, head_size)``
    matching the vLLM NHD layout.
    """
    torch.random.manual_seed(42)
    dev = torch.device(device)
    return [
        torch.randn(
            (2, num_blocks, block_size, num_heads, head_size),
            dtype=dtype,
            device=dev,
        )
        for _ in range(num_layers)
    ]


def _send_register_kv_cache(
    sock: zmq.Socket,
    instance_id: int = 0,
    model_name: str = _MODEL_NAME,
    world_size: int = _WORLD_SIZE,
    layout_hints: dict | None = None,
    gpu_tensors: list[torch.Tensor] | None = None,
) -> bool:
    """REGISTER_KV_CACHE — register a KV cache context.

    In CPU mode the server creates a ``CpuCacheContext``;
    in GPU mode real CUDA tensors are wrapped via
    ``CudaIPCWrapper`` and sent over IPC.
    """
    hints: dict = {"kv_layout": "NHD"}
    if layout_hints:
        hints.update(layout_hints)

    if gpu_tensors is not None:
        kv_caches = [CudaIPCWrapper(t) for t in gpu_tensors]
        enc = get_customized_encoder(
            type=list[CudaIPCWrapper],
        )
        b_kv = enc.encode(kv_caches)
    else:
        b_kv = msgspec.msgpack.encode([])  # CPU mode

    payloads = [
        msgspec.msgpack.encode(instance_id),
        b_kv,
        msgspec.msgpack.encode(model_name),
        msgspec.msgpack.encode(world_size),
        msgspec.msgpack.encode(hints),
    ]
    resp = _send_request(
        sock,
        RequestType.REGISTER_KV_CACHE,
        payloads,
    )
    return resp is not None


def _send_lookup(
    sock: zmq.Socket,
    key: IPCCacheEngineKey,
) -> int | None:
    """LOOKUP — returns prefetch job ID, or None on timeout."""
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(1),  # tp_size
    ]
    resp = _send_request(sock, RequestType.LOOKUP, payloads)
    if resp is None:
        return None
    return msgspec.msgpack.decode(resp[0], type=int)


def _poll_prefetch_status(
    sock: zmq.Socket,
    job_id: int,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or None if timed out.
    """
    for _ in range(max_polls):
        payloads = [msgspec.msgpack.encode(job_id)]
        resp = _send_request(
            sock,
            RequestType.QUERY_PREFETCH_STATUS,
            payloads,
        )
        if resp is None:
            return None
        result = msgspec.msgpack.decode(
            resp[0],
            type=int | None,  # type: ignore[arg-type]
        )
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _make_event_handle(use_gpu: bool) -> bytes:
    """Create a CUDA event IPC handle, or empty bytes."""
    if use_gpu:
        event = torch.cuda.Event(interprocess=True)
        event.record()
        return event.ipc_handle()
    return b""


def _send_store(
    sock: zmq.Socket,
    key: IPCCacheEngineKey,
    chunk_size: int,
    block_offset: int = 0,
    block_size: int = 16,
    use_gpu: bool = False,
) -> str:
    """STORE — store KV cache blocks. Returns status string."""
    num_tokens = key.end - key.start
    num_blocks = num_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(_INSTANCE_ID),
        msgspec.msgpack.encode(block_ids),
        msgspec.msgpack.encode(_make_event_handle(use_gpu)),
    ]
    resp = _send_request(sock, RequestType.STORE, payloads)
    if resp is None:
        return "timeout"
    result = msgspec.msgpack.decode(
        resp[0],
        type=tuple[bytes, bool],
    )
    return "stored" if result[1] else "store_failed"


def _send_retrieve(
    sock: zmq.Socket,
    key: IPCCacheEngineKey,
    chunk_size: int,
    hit_chunks: int,
    block_offset: int = 0,
    block_size: int = 16,
    use_gpu: bool = False,
) -> str:
    """RETRIEVE — retrieve KV cache blocks. Returns status."""
    hit_tokens = hit_chunks * chunk_size
    num_blocks = hit_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(_INSTANCE_ID),
        msgspec.msgpack.encode(block_ids),
        msgspec.msgpack.encode(_make_event_handle(use_gpu)),
        msgspec.msgpack.encode(0),  # skip_first_n_tokens
    ]
    resp = _send_request(
        sock,
        RequestType.RETRIEVE,
        payloads,
    )
    if resp is None:
        return "timeout"
    result = msgspec.msgpack.decode(
        resp[0],
        type=tuple[bytes, bool],
    )
    return "retrieved" if result[1] else "retrieve_failed"


def _send_end_session(
    sock: zmq.Socket,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    payloads = [msgspec.msgpack.encode(request_id)]
    _send_request(sock, RequestType.END_SESSION, payloads)


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    slot_start: int,
    slot_end: int,
    chunk_size: int,
) -> list[str] | None:
    """Query KV cache checksums via the HTTP API."""
    slots = list(range(slot_start, slot_end))
    compressed = compress_slot_mapping(slots)
    parts: list[str] = []
    for item in compressed:
        if isinstance(item, list):
            parts.append("[%d,%d]" % (item[0], item[1]))
        else:
            parts.append(str(item))
    slot_mapping = ",".join(parts)
    url = "%s/api/kvcache/check?slot_mapping=%s&chunk_size=%d" % (
        http_base,
        slot_mapping,
        chunk_size,
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return data.get("chunk_checksums", [])
    except (urllib.error.URLError, OSError):
        pass
    return None


# ------------------------------------------------------------------ #
#  Per-request flow                                                    #
# ------------------------------------------------------------------ #


def _process_request(
    sock: zmq.Socket,
    seq_no: int,
    num_tokens: int,
    chunk_size: int,
    pass_label: str,
    http_base: str = "",
    block_size: int = 16,
    use_gpu: bool = False,
) -> list[str] | None:
    """Run the full lookup -> retrieve/store flow."""
    token_ids = _build_token_ids(seq_no, num_tokens)
    request_id = "req-%d-%s" % (seq_no, pass_label)

    # Align end to chunk_size (only full chunks)
    num_full_tokens = (len(token_ids) // chunk_size) * chunk_size
    if num_full_tokens == 0:
        print(
            "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
            % (seq_no, pass_label, len(token_ids), chunk_size)
        )
        return None

    # Key for lookup (worker_id=None)
    lookup_key = _make_key(
        token_ids,
        request_id,
        start=0,
        end=num_full_tokens,
    )

    # 1. LOOKUP
    t0 = time.monotonic()
    job_id = _send_lookup(sock, lookup_key)
    if job_id is None:
        print("  [seq %d/%s] LOOKUP timeout" % (seq_no, pass_label))
        return None

    # 2. QUERY_PREFETCH_STATUS (poll)
    hit_chunks = _poll_prefetch_status(sock, job_id)
    if hit_chunks is None:
        hit_chunks = 0

    total_chunks = num_full_tokens // chunk_size
    miss_chunks = total_chunks - hit_chunks
    hit_tokens = hit_chunks * chunk_size
    lookup_ms = (time.monotonic() - t0) * 1000

    print(
        "  [seq %d/%s] LOOKUP: %d/%d chunks hit "
        "(%.1f ms)"
        % (
            seq_no,
            pass_label,
            hit_chunks,
            total_chunks,
            lookup_ms,
        )
    )

    # Block offset: each request uses a different block
    # range so that different requests touch different data.
    num_blocks = num_full_tokens // block_size
    block_offset = seq_no * num_blocks

    # 3. RETRIEVE hit portion
    if hit_chunks > 0:
        retrieve_key = _make_key(
            token_ids,
            request_id,
            start=0,
            end=hit_tokens,
            worker_id=0,
        )
        t1 = time.monotonic()
        status = _send_retrieve(
            sock,
            retrieve_key,
            chunk_size,
            hit_chunks,
            block_offset=block_offset,
            block_size=block_size,
            use_gpu=use_gpu,
        )
        retrieve_ms = (time.monotonic() - t1) * 1000
        print(
            "  [seq %d/%s] RETRIEVE: %s "
            "(%d tokens, %.1f ms)"
            % (
                seq_no,
                pass_label,
                status,
                hit_tokens,
                retrieve_ms,
            )
        )

    # 4. STORE miss portion
    if miss_chunks > 0:
        store_start = hit_tokens
        store_end = num_full_tokens
        store_key = _make_key(
            token_ids,
            request_id,
            start=store_start,
            end=store_end,
            worker_id=0,
        )
        t2 = time.monotonic()
        store_block_off = block_offset + (hit_tokens // block_size)
        status = _send_store(
            sock,
            store_key,
            chunk_size,
            block_offset=store_block_off,
            block_size=block_size,
            use_gpu=use_gpu,
        )
        store_ms = (time.monotonic() - t2) * 1000
        print(
            "  [seq %d/%s] STORE: %s "
            "(%d tokens, %.1f ms)"
            % (
                seq_no,
                pass_label,
                status,
                store_end - store_start,
                store_ms,
            )
        )

    # 5. Query checksums via HTTP API
    checksums = None
    if http_base and num_full_tokens > 0:
        slot_start = block_offset * block_size
        slot_end = slot_start + num_full_tokens
        checksums = _query_checksum(
            http_base,
            slot_start,
            slot_end,
            chunk_size,
        )
        if checksums:
            digest = hashlib.md5("".join(checksums).encode()).hexdigest()[:16]
            print(
                "  [seq %d/%s] CHECKSUM: %s (%d chunks)"
                % (
                    seq_no,
                    pass_label,
                    digest,
                    len(checksums),
                )
            )

    # 6. END_SESSION
    _send_end_session(sock, request_id)
    return checksums


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(sock: zmq.Socket) -> int:
    """Query the server's chunk size."""
    resp = _send_request(sock, RequestType.GET_CHUNK_SIZE)
    if resp and resp:
        return msgspec.msgpack.decode(resp[0], type=int)
    return 256  # fallback


# ------------------------------------------------------------------ #
#  Command                                                             #
# ------------------------------------------------------------------ #


class TestCacheCommand(BaseCommand):
    """End-to-end test for LMCache MP cache server."""

    def name(self) -> str:
        return "test-cache"

    def help(self) -> str:
        return "End-to-end test for LMCache MP cache server (CPU / GPU)."

    def add_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        parser.add_argument(
            "--port",
            type=int,
            default=5555,
            help="MP server port (default: 5555)",
        )
        parser.add_argument(
            "--host",
            default="localhost",
            help="MP server host (default: localhost)",
        )
        parser.add_argument(
            "--mode",
            choices=["cpu", "gpu"],
            default="cpu",
            help="Run mode: cpu or gpu (default: cpu)",
        )
        parser.add_argument(
            "--num-tokens",
            type=int,
            default=512,
            help="Tokens per request (default: 512)",
        )

        # -- KV cache shape --
        kv = parser.add_argument_group(
            "KV cache shape",
        )
        kv.add_argument(
            "--num-layers",
            type=int,
            default=32,
            help="Number of KV layers (default: 32)",
        )
        kv.add_argument(
            "--num-heads",
            type=int,
            default=8,
            help="Attention heads (default: 8)",
        )
        kv.add_argument(
            "--head-size",
            type=int,
            default=128,
            help="Head size (default: 128)",
        )
        kv.add_argument(
            "--num-blocks",
            type=int,
            default=1024,
            help="Paged blocks (default: 1024)",
        )
        kv.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Tokens per block (default: 16)",
        )
        kv.add_argument(
            "--dtype",
            type=str,
            default="float16",
            choices=["float16", "bfloat16", "float32"],
            help="KV cache dtype (default: float16)",
        )

        parser.add_argument(
            "--start",
            type=int,
            default=0,
            help="Starting sequence number (default: 0)",
        )
        parser.add_argument(
            "--end",
            type=int,
            default=None,
            help=("Ending sequence number (exclusive). If not set, runs forever."),
        )
        parser.add_argument(
            "--interval",
            type=float,
            default=0.5,
            help="Seconds between requests (default: 0.5)",
        )
        parser.add_argument(
            "--http-port",
            type=int,
            default=8080,
            help=("HTTP server port for checksum API (default: 8080)"),
        )

    def execute(self, args: argparse.Namespace) -> None:
        use_gpu = args.mode == "gpu"
        if use_gpu and not torch.cuda.is_available():
            print("ERROR: --mode gpu requires CUDA")
            sys.exit(1)

        url = "tcp://%s:%d" % (args.host, args.port)
        print("Connecting to LMCache MP Server at %s (mode=%s) ..." % (url, args.mode))

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 1000)
        sock.connect(url)

        # Query chunk size from server
        chunk_size = _get_chunk_size(sock)
        print("Server chunk_size = %d" % chunk_size)

        # Build layout_hints.
        # dtype is sent as a string ("float16") because
        # torch.dtype is not msgpack-serializable.
        layout_hints = {
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "head_size": args.head_size,
            "num_blocks": args.num_blocks,
            "block_size": args.block_size,
            "dtype": args.dtype,
        }

        num_tokens = args.num_tokens
        print(
            "Each request: %d tokens (%d full chunks)"
            % (
                num_tokens + 1,
                (num_tokens + 1) // chunk_size,
            )
        )
        print(
            "KV shape: %d layers, %d heads x %d, "
            "dtype=%s, blocks=%dx%d"
            % (
                args.num_layers,
                args.num_heads,
                args.head_size,
                args.dtype,
                args.num_blocks,
                args.block_size,
            )
        )

        # Allocate GPU tensors if in GPU mode
        gpu_tensors: list[torch.Tensor] | None = None
        if use_gpu:
            gpu_tensors = _allocate_gpu_kv_cache(
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                head_size=args.head_size,
                num_blocks=args.num_blocks,
                block_size=args.block_size,
                dtype=_DTYPE_MAP[args.dtype],
            )
            print(
                "Allocated %d GPU tensors on %s"
                % (
                    len(gpu_tensors),
                    gpu_tensors[0].device,
                )
            )

        # Register KV cache before any store/retrieve
        ok = _send_register_kv_cache(
            sock,
            layout_hints=layout_hints,
            gpu_tensors=gpu_tensors,
        )
        print("REGISTER_KV_CACHE: %s" % ("OK" if ok else "FAIL"))
        print()

        if args.end is not None:
            seq_iter: itertools.count | range = range(
                args.start,
                args.end,
            )
        else:
            seq_iter = itertools.count(args.start)

        http_base = "http://%s:%d" % (args.host, args.http_port)

        try:
            for seq_no in seq_iter:
                print("=== Request seq=%d ===" % seq_no)

                # Pass 1: cold (miss -> store)
                cold_checksums = _process_request(
                    sock,
                    seq_no,
                    num_tokens,
                    chunk_size,
                    "cold",
                    http_base=http_base,
                    block_size=args.block_size,
                    use_gpu=use_gpu,
                )

                time.sleep(args.interval)

                # Pass 2: warm (hit -> retrieve)
                warm_checksums = _process_request(
                    sock,
                    seq_no,
                    num_tokens,
                    chunk_size,
                    "warm",
                    http_base=http_base,
                    block_size=args.block_size,
                    use_gpu=use_gpu,
                )

                # Compare checksums
                if cold_checksums and warm_checksums:
                    if cold_checksums == warm_checksums:
                        print("  [seq %d] CHECKSUM MATCH OK" % seq_no)
                    else:
                        print("  [seq %d] CHECKSUM MISMATCH!" % seq_no)
                        for i, (c, w) in enumerate(
                            zip(
                                cold_checksums,
                                warm_checksums,
                                strict=False,
                            )
                        ):
                            print(
                                "    chunk %d: cold=%s "
                                "warm=%s %s"
                                % (
                                    i,
                                    c[:12],
                                    w[:12],
                                    "OK" if c == w else "FAIL",
                                )
                            )

                print()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping...")

        sock.close()
        ctx.term()
        print("Done.")

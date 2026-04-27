# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench kvcache`` — end-to-end test for LMCache MP cache server.

Supports **GPU** mode (``--mode gpu``).

.. note::
    CPU mode is planned but not yet implemented.

This command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup, get prefetch job ID
      2. QUERY_PREFETCH_STATUS — poll until prefetch completes
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

Usage examples::

    # GPU mode: real CUDA tensors + IPC
    lmcache bench kvcache --rpc-url tcp://localhost:5555 \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape (multi-group spec)
    lmcache bench kvcache --rpc-url tcp://localhost:5555 \\
        --kvcache-shape-spec '(2,32,1024,8,128):float16:32'

    # Run forever starting from sequence 0
    lmcache bench kvcache --rpc-url tcp://localhost:5555
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
from lmcache.v1.kv_layer_groups import (
    DTYPE_MAP,
    format_kvcache_shape_spec,
    parse_kvcache_shape_spec,
)
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

# Default KV shape spec matching the original defaults:
# 32 layers, (2, num_blocks=1024, block_size=16, 8 heads, 128 head_size)
_DEFAULT_SHAPE_SPEC = "(2,1024,16,8,128):float16:32"

# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #


class ZmqClient:
    """Thin wrapper around a DEALER ``zmq.Socket``.

    Encapsulates the per-connection monotonic request UID counter
    (previously a module-level global) and adds UID-matching on
    ``recv_multipart`` so late replies from prior timed-out requests
    are discarded instead of being mis-interpreted as the next
    request's response.
    """

    def __init__(self, sock: zmq.Socket) -> None:
        self._sock = sock
        self._uid_counter = 0

    @property
    def sock(self) -> zmq.Socket:
        """Expose the underlying socket (read-only)."""
        return self._sock

    def _next_uid(self) -> int:
        uid = self._uid_counter
        self._uid_counter += 1
        return uid

    def send_request(
        self,
        request_type: RequestType,
        payloads: list[bytes] | None = None,
        timeout_ms: int = 10000,
    ) -> list[bytes] | None:
        """Send a request and wait for the matching response.

        Returns the raw response payload frames (excluding uid and
        type) on success -- an empty list means a successful void
        reply -- or *None* on timeout.

        Malformed frames (``< 2`` frames or undecodable UID) and
        late replies from previously timed-out calls are silently
        dropped; polling continues until the matching UID arrives
        or the overall budget expires.
        """
        uid = self._next_uid()
        b_uid = msgspec.msgpack.encode(uid)
        b_type = msgspec.msgpack.encode(request_type)

        frames: list[bytes] = [b_uid, b_type]
        if payloads:
            frames.extend(payloads)

        self._sock.send_multipart(frames)

        # Deadline-based loop: poll & recv until we see a reply
        # whose UID matches ours, or the overall budget expires.
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return None
            if not self._sock.poll(remaining_ms, zmq.POLLIN):
                return None

            resp = self._sock.recv_multipart()
            if len(resp) < 2:
                # Malformed reply -- drop and keep polling so a
                # stray frame from a prior timed-out request does
                # not fail the current one.
                continue

            try:
                resp_uid = msgspec.msgpack.decode(resp[0], type=int)
            except msgspec.DecodeError:
                # Unparsable UID -- skip and keep waiting.
                continue

            if resp_uid != uid:
                # Stale response from a previously timed-out
                # request. Drop and keep polling.
                continue

            # resp[0]=uid, resp[1]=type; payload starts at [2].
            return resp[2:]


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
    device: str | torch.device | None = None,
    kv_size: int = 2,
) -> list[torch.Tensor]:
    """Allocate paged GPU KV cache tensors.

    Each layer is a tensor of shape
    ``(kv_size, num_blocks, block_size, num_heads, head_size)``
    matching the vLLM NHD layout. ``kv_size`` is 2 for standard
    K/V attention; override via the ``--kvcache-shape-spec``
    first dimension for architectures that need a different
    leading dimension (e.g. MLA).
    """
    torch.random.manual_seed(42)
    dev = (
        torch.device(device)
        if device
        else torch.device("cuda", torch.cuda.current_device())
    )
    shape = (kv_size, num_blocks, block_size, num_heads, head_size)

    def _alloc() -> torch.Tensor:
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype, device=dev)
        # ``torch.randn`` only supports floating-point dtypes; fall
        # back to ``randint`` for integer dtypes (e.g. ``uint8``
        # used by FP8 quantized KV cache layouts).
        iinfo = torch.iinfo(dtype)
        return torch.randint(iinfo.min, iinfo.max + 1, shape, dtype=dtype, device=dev)

    return [_alloc() for _ in range(num_layers)]


def _send_register_kv_cache(
    client: ZmqClient,
    instance_id: int = 0,
    model_name: str = _MODEL_NAME,
    world_size: int = _WORLD_SIZE,
    layout_hints: dict | None = None,
    gpu_tensors: list[torch.Tensor] | None = None,
) -> bool:
    """REGISTER_KV_CACHE — register a KV cache context.

    In GPU mode real CUDA tensors are wrapped via
    ``CudaIPCWrapper`` and sent over IPC.

    .. note::
        CPU mode (``gpu_tensors is None``) is not yet
        supported.
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
        # TODO(maobaolong): support CPU mode registration
        raise NotImplementedError(
            "CPU mode is not yet supported. Please use --mode gpu."
        )

    payloads = [
        msgspec.msgpack.encode(instance_id),
        b_kv,
        msgspec.msgpack.encode(model_name),
        msgspec.msgpack.encode(world_size),
        msgspec.msgpack.encode(hints),
    ]
    resp = client.send_request(
        RequestType.REGISTER_KV_CACHE,
        payloads,
    )
    return resp is not None


def _send_lookup(
    client: ZmqClient,
    key: IPCCacheEngineKey,
) -> int | None:
    """LOOKUP — returns prefetch job ID, or None on timeout."""
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(1),  # tp_size
    ]
    resp = client.send_request(RequestType.LOOKUP, payloads)
    if not resp:
        return None
    return msgspec.msgpack.decode(resp[0], type=int)


def _poll_prefetch_status(
    client: ZmqClient,
    job_id: int,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or None if timed out.
    """
    for _ in range(max_polls):
        payloads = [msgspec.msgpack.encode(job_id)]
        resp = client.send_request(
            RequestType.QUERY_PREFETCH_STATUS,
            payloads,
        )
        if not resp:
            return None
        result = msgspec.msgpack.decode(
            resp[0],
            type=int | None,  # type: ignore[arg-type]
        )
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _make_event_handle() -> bytes:
    """Create a CUDA event IPC handle for GPU mode."""
    event = torch.cuda.Event(interprocess=True)
    event.record()
    return event.ipc_handle()


def _send_store(
    client: ZmqClient,
    key: IPCCacheEngineKey,
    block_offset: int = 0,
    block_size: int = 16,
) -> str:
    """STORE — store KV cache blocks. Returns status string."""
    num_tokens = key.end - key.start
    num_blocks = num_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(_INSTANCE_ID),
        msgspec.msgpack.encode(block_ids),
        msgspec.msgpack.encode(_make_event_handle()),
    ]
    resp = client.send_request(RequestType.STORE, payloads)
    if not resp:
        return "timeout"
    result = msgspec.msgpack.decode(
        resp[0],
        type=tuple[bytes, bool],
    )
    return "stored" if result[1] else "store_failed"


def _send_retrieve(
    client: ZmqClient,
    key: IPCCacheEngineKey,
    chunk_size: int,
    hit_chunks: int,
    block_offset: int = 0,
    block_size: int = 16,
) -> str:
    """RETRIEVE — retrieve KV cache blocks. Returns status."""
    hit_tokens = hit_chunks * chunk_size
    num_blocks = hit_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [
        msgspec.msgpack.encode(key),
        msgspec.msgpack.encode(_INSTANCE_ID),
        msgspec.msgpack.encode(block_ids),
        msgspec.msgpack.encode(_make_event_handle()),
        msgspec.msgpack.encode(0),  # skip_first_n_tokens
    ]
    resp = client.send_request(
        RequestType.RETRIEVE,
        payloads,
    )
    if not resp:
        return "timeout"
    result = msgspec.msgpack.decode(
        resp[0],
        type=tuple[bytes, bool],
    )
    return "retrieved" if result[1] else "retrieve_failed"


def _send_end_session(
    client: ZmqClient,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    payloads = [msgspec.msgpack.encode(request_id)]
    client.send_request(RequestType.END_SESSION, payloads)


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    slot_start: int,
    slot_end: int,
    chunk_size: int,
) -> list[str] | None:
    """Query KV cache checksums via the HTTP API.

    This CLI pins ``layerwise=false`` so the server always
    returns ``chunk_checksums`` as a flat ``list[str]``. We
    still defensively validate the response type — if a future
    endpoint variant returns a per-layer ``dict`` we log and
    skip the comparison rather than letting ``str.join`` crash.
    """
    slots = list(range(slot_start, slot_end))
    compressed = compress_slot_mapping(slots)
    parts: list[str] = []
    for item in compressed:
        if isinstance(item, list):
            parts.append("[%d,%d]" % (item[0], item[1]))
        else:
            parts.append(str(item))
    slot_mapping = ",".join(parts)
    url = ("%s/api/kvcache/check?slot_mapping=%s&chunk_size=%d&layerwise=false") % (
        http_base,
        slot_mapping,
        chunk_size,
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") != "success":
                return None
            checksums = data.get("chunk_checksums", [])
            if not isinstance(checksums, list) or not all(
                isinstance(c, str) for c in checksums
            ):
                print(
                    "  [WARNING] unexpected chunk_checksums "
                    "type=%s; expected list[str]" % type(checksums).__name__
                )
                return None
            return checksums
    except (urllib.error.URLError, OSError) as exc:
        print("  [WARNING] Checksum query failed: %s" % exc)
    return None


# ------------------------------------------------------------------ #
#  Per-request flow                                                    #
# ------------------------------------------------------------------ #


def _process_request(
    client: ZmqClient,
    seq_no: int,
    num_tokens: int,
    chunk_size: int,
    pass_label: str,
    http_base: str = "",
    block_size: int = 16,
    total_blocks: int = 1024,
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
    job_id = _send_lookup(client, lookup_key)
    if job_id is None:
        print("  [seq %d/%s] LOOKUP timeout" % (seq_no, pass_label))
        return None

    # 2. QUERY_PREFETCH_STATUS (poll)
    hit_chunks = _poll_prefetch_status(client, job_id)
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
    # Wrap with modulo and clamp so the entire range
    # [block_offset, block_offset + num_blocks) stays
    # within [0, total_blocks).
    num_blocks = num_full_tokens // block_size
    usable = max(total_blocks - num_blocks, 1)
    block_offset = (seq_no * num_blocks) % usable

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
            client,
            retrieve_key,
            chunk_size,
            hit_chunks,
            block_offset=block_offset,
            block_size=block_size,
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
            client,
            store_key,
            block_offset=store_block_off,
            block_size=block_size,
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
    _send_end_session(client, request_id)
    return checksums


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: ZmqClient) -> int:
    """Query the server's chunk size."""
    resp = client.send_request(RequestType.GET_CHUNK_SIZE)
    if resp:
        return msgspec.msgpack.decode(resp[0], type=int)
    return 256  # fallback


# ------------------------------------------------------------------ #
#  Command                                                             #
# ------------------------------------------------------------------ #


class TestCacheCommand(BaseCommand):
    """End-to-end test for the LMCache MP cache server.

    Connects to a running LMCache multiprocess (MP) server via
    ZMQ DEALER and exercises the full KV-cache data path
    (REGISTER → LOOKUP → QUERY_PREFETCH_STATUS → RETRIEVE →
    STORE → optional HTTP checksum) for a sequence of synthetic
    requests. Each sequence is replayed twice — a "cold" pass
    (expected cache miss → STORE) followed by a "warm" pass
    (expected hit → RETRIEVE) — and the per-chunk checksums are
    compared to verify round-trip integrity.

    The command is registered under ``lmcache bench kvcache``.

    CLI arguments (see :meth:`add_arguments` for full details):
        --rpc-url: ZMQ endpoint of the MP server.
        --mode: Currently only ``gpu`` is supported; CPU mode is
            a planned follow-up.
        --num-tokens: Number of tokens per synthetic request.
        --kvcache-shape-spec: Multi-group KV cache shape spec in
            the form ``(shape):dtype:layers[;...]``.
        --num-blocks / --block-size: Paged-KV allocation sizing.
        --start / --end: Sequence number range (exclusive end).
            When ``--end`` is omitted the loop runs forever.
        --interval: Delay (seconds) between sub-passes.
        --url: HTTP base URL of the cache server's checksum API.

    Exit behaviour:
        * Exits with status 1 if CUDA is unavailable in GPU mode.
        * Ctrl-C triggers a graceful shutdown of the ZMQ socket
          and context before returning.
    """

    def name(self) -> str:
        """Return the CLI sub-command name."""
        return "test-cache"

    def help(self) -> str:
        """Return a short help string for ``--help`` output."""
        return "End-to-end test for LMCache MP cache server (GPU mode)."

    def add_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Register CLI arguments for the test-cache command."""
        parser.add_argument(
            "--rpc-url",
            default="tcp://localhost:5555",
            help=("ZMQ endpoint of the MP server (default: tcp://localhost:5555)"),
        )
        # TODO(maobaolong): add "cpu" choice once CPU mode is
        # implemented.
        parser.add_argument(
            "--mode",
            choices=["gpu"],
            default="gpu",
            help="Run mode (default: gpu)",
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
            "--kvcache-shape-spec",
            type=str,
            default=_DEFAULT_SHAPE_SPEC,
            help=(
                "KV shape spec. One or more groups separated by ';'. "
                "Each group is '(kv_size,NB,BS,NH,HS):dtype:layers' "
                "where NB=num_blocks, BS=block_size, NH=num_heads, "
                "HS=head_size. Supported dtypes: %s. "
                "See lmcache.v1.kv_layer_groups.parse_kvcache_shape_spec "
                "for full docs. Default: '%s'"
                % (", ".join(DTYPE_MAP.keys()), _DEFAULT_SHAPE_SPEC)
            ),
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
            help=("Seconds between requests (default: 0.5)"),
        )
        parser.add_argument(
            "--url",
            default="http://localhost:8080",
            help=("HTTP base URL for checksum API (default: http://localhost:8080)"),
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Run the end-to-end cache test loop."""
        if not torch.cuda.is_available():
            print("ERROR: --mode gpu requires CUDA")
            sys.exit(1)

        url = args.rpc_url
        print("Connecting to LMCache MP Server at %s (mode=%s) ..." % (url, args.mode))

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 1000)
        sock.connect(url)
        client = ZmqClient(sock)

        try:
            # Query chunk size from server
            chunk_size = _get_chunk_size(client)
            print("Server chunk_size = %d" % chunk_size)

            # Parse KV shape spec
            layer_groups = parse_kvcache_shape_spec(
                args.kvcache_shape_spec,
            )
            # Echo the resolved spec so operators can verify that
            # their input was interpreted as intended. The echoed
            # string is a valid ``--kvcache-shape-spec`` itself.
            print(
                "Resolved KV shape spec: %s" % format_kvcache_shape_spec(layer_groups)
            )
            # Use the first group to derive shape parameters.
            # ``nb``/``bs``/``kv_size`` from the spec take
            # precedence when set (>0); otherwise fall back
            # to the CLI flags so existing specs that only
            # declare ``nh``/``hs`` keep working.
            first = layer_groups[0]
            num_layers = sum(g.num_layers for g in layer_groups)
            num_heads = first.shape_desc.nh
            head_size = first.shape_desc.hs
            spec_nb = getattr(first.shape_desc, "nb", 0) or 0
            spec_bs = getattr(first.shape_desc, "bs", 0) or 0
            spec_kv = getattr(first.shape_desc, "kv_size", 0) or 0
            num_blocks = spec_nb if spec_nb > 0 else args.num_blocks
            block_size = spec_bs if spec_bs > 0 else args.block_size
            kv_size = spec_kv if spec_kv > 0 else 2
            if spec_nb and spec_nb != args.num_blocks:
                print(
                    "  [info] spec nb=%d overrides --num-blocks=%d"
                    % (spec_nb, args.num_blocks)
                )
            if spec_bs and spec_bs != args.block_size:
                print(
                    "  [info] spec bs=%d overrides --block-size=%d"
                    % (spec_bs, args.block_size)
                )
            dtype = first.dtype
            dtype_str = next(
                (k for k, v in DTYPE_MAP.items() if v == dtype),
                "float16",
            )

            # Build layout_hints.
            # dtype is sent as a string ("float16") because
            # torch.dtype is not msgpack-serializable.
            layout_hints = {
                "num_layers": num_layers,
                "num_heads": num_heads,
                "head_size": head_size,
                "num_blocks": num_blocks,
                "block_size": block_size,
                "dtype": dtype_str,
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
                "dtype=%s, blocks=%dx%d, kv=%d"
                % (
                    num_layers,
                    num_heads,
                    head_size,
                    dtype_str,
                    num_blocks,
                    block_size,
                    kv_size,
                )
            )

            # Allocate GPU tensors
            gpu_tensors = _allocate_gpu_kv_cache(
                num_layers=num_layers,
                num_heads=num_heads,
                head_size=head_size,
                num_blocks=num_blocks,
                block_size=block_size,
                dtype=dtype,
                kv_size=kv_size,
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
                client,
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

            http_base = args.url.rstrip("/")

            for seq_no in seq_iter:
                print("=== Request seq=%d ===" % seq_no)

                # Pass 1: cold (miss -> store)
                cold_checksums = _process_request(
                    client,
                    seq_no,
                    num_tokens,
                    chunk_size,
                    "cold",
                    http_base=http_base,
                    block_size=block_size,
                    total_blocks=num_blocks,
                )

                time.sleep(args.interval)

                # Pass 2: warm (hit -> retrieve)
                warm_checksums = _process_request(
                    client,
                    seq_no,
                    num_tokens,
                    chunk_size,
                    "warm",
                    http_base=http_base,
                    block_size=block_size,
                    total_blocks=num_blocks,
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
                                    ("OK" if c == w else "FAIL"),
                                )
                            )

                print()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            sock.close()
            ctx.term()
        print("Done.")

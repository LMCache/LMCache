# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench kvcache`` — end-to-end test for LMCache MP cache server.

Supports **GPU** mode (``--mode gpu``).

.. note::
    CPU mode is planned but not yet implemented.

This command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
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
from typing import Any
import argparse
import hashlib
import itertools
import json
import sys
import time
import urllib.error
import urllib.request

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.cli.commands.base import BaseCommand

# ``lmcache bench kvcache`` allocates real CUDA tensors and talks to
# the MP server via ZMQ, both of which are absent from the thin
# ``lmcache-cli`` distribution (no torch, no zmq, no lmcache.v1.*).
# Importing them unconditionally would kill the *entire* ``lmcache``
# CLI at registry load time with an opaque ImportError. Wrap the
# heavy imports and remember the error so ``add_arguments`` /
# ``execute`` can bail out with an actionable install hint.
_IMPORT_ERROR: ImportError | None = None
try:
    # Third Party
    import torch
    import zmq

    # First Party
    from lmcache.utils import (
        EngineType,
        check_interprocess_event_support,
        compress_slot_mapping,
    )
    from lmcache.v1.kv_layer_groups import (
        DTYPE_MAP,
        KVLayerGroupInfo,
        format_kvcache_shape_spec,
        parse_kvcache_shape_spec,
    )
    from lmcache.v1.multiprocess.custom_types import (
        CudaIPCWrapper,
        IPCCacheEngineKey,
    )
    from lmcache.v1.multiprocess.futures import MessagingFuture
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.protocols.base import RequestType
except ImportError as _exc:
    _IMPORT_ERROR = _exc
    # Fallback placeholder so ``add_arguments`` can still build its
    # help text without crashing on a CLI-only install.
    DTYPE_MAP = {}  # type: ignore[assignment]


def _require_full_install() -> None:
    """Exit with an install hint if the full LMCache runtime is missing.

    ``lmcache bench kvcache`` needs torch, zmq and ``lmcache.v1.*``
    (MP client, KV layer-group parser). When those imports failed at
    module load — almost always because the user installed
    ``lmcache-cli`` instead of the full package — print the shortest
    actionable message to stderr and exit with status ``2`` so
    scripts can detect the install gap programmatically.
    """
    if _IMPORT_ERROR is None:
        return
    print(
        "ERROR: `lmcache bench kvcache` needs the full LMCache package "
        "(torch, zmq, MP runtime), but only the `lmcache-cli` shell "
        "appears to be installed.\n"
        "  Install the full package with `pip install lmcache` and try "
        "again.\n"
        f"  Original import error: {_IMPORT_ERROR}",
        file=sys.stderr,
    )
    sys.exit(2)


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

# Default RPC call timeout (seconds) for blocking request/reply
# round-trips.
_DEFAULT_RPC_TIMEOUT_S = 10.0

# Unique sentinel returned by :func:`_call` on RPC timeout so callers
# can disambiguate it from a legitimate ``None`` (void) reply.
_TIMEOUT = object()


def _call(
    client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
) -> Any:
    """Submit a request through ``MessageQueueClient`` and block.

    Returns the decoded response (possibly ``None`` for void replies)
    on success, or the sentinel ``_TIMEOUT`` on RPC timeout.
    """
    future: MessagingFuture[Any] = client.submit_request(request_type, payloads)
    try:
        return future.result(timeout=timeout_s)
    except TimeoutError:
        return _TIMEOUT


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
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    kv_size: int = 2,
    groups: list[KVLayerGroupInfo] | None = None,
) -> list[torch.Tensor]:
    """Allocate paged GPU KV cache tensors.

    Each layer is a tensor of shape
    ``(kv_size, num_blocks, block_size, num_heads, head_size)``
    matching the vLLM NHD layout. ``kv_size`` is 2 for standard
    K/V attention; override via the ``--kvcache-shape-spec``
    first dimension for architectures that need a different
    leading dimension (e.g. MLA).

    When ``groups`` is provided, tensors are allocated per-group
    using each group's own ``(kv_size, NB, BS, NH, HS)`` / ``dtype``
    (for heterogeneous multi-group specs). In that mode the flat
    ``num_heads`` / ``head_size`` / ``dtype`` / ``kv_size`` kwargs
    are ignored, and ``num_layers`` is derived from the groups.
    """
    # ``torch.float16`` cannot be used as a default value because the
    # module must load on ``lmcache-cli`` (no torch) installs.
    if dtype is None:
        dtype = torch.float16
    torch.random.manual_seed(42)
    dev = (
        torch.device(device)
        if device
        else torch.device(torch_device_type, torch_dev.current_device())
    )

    def _alloc(
        shape: tuple[int, ...],
        a_dtype: torch.dtype,
    ) -> torch.Tensor:
        if a_dtype.is_floating_point:
            return torch.randn(shape, dtype=a_dtype, device=dev)
        # ``torch.randn`` only supports floating-point dtypes; fall
        # back to ``randint`` for integer dtypes (e.g. ``uint8``
        # used by FP8 quantized KV cache layouts).
        iinfo = torch.iinfo(a_dtype)
        return torch.randint(iinfo.min, iinfo.max + 1, shape, dtype=a_dtype, device=dev)

    if groups:
        tensors: list[torch.Tensor] = []
        for g in groups:
            sd = g.shape_desc
            g_shape = (sd.kv_size, sd.nb, sd.bs, sd.nh, sd.hs)
            tensors.extend(_alloc(g_shape, g.dtype) for _ in range(sd.nl))
        return tensors

    shape = (kv_size, num_blocks, block_size, num_heads, head_size)
    return [_alloc(shape, dtype) for _ in range(num_layers)]


def _send_register_kv_cache(
    client: MessageQueueClient,
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

    if gpu_tensors is None:
        # TODO(maobaolong): support CPU mode registration
        raise NotImplementedError(
            "CPU mode is not yet supported. Please use --mode gpu."
        )

    kv_caches = [CudaIPCWrapper(t) for t in gpu_tensors]
    # TODO(maobaolong): Make the engine type configurable
    payloads = [
        instance_id,
        kv_caches,
        model_name,
        world_size,
        EngineType.VLLM,
        hints,
    ]
    result = _call(client, RequestType.REGISTER_KV_CACHE, payloads)
    return result is not _TIMEOUT


def _send_lookup(
    client: MessageQueueClient,
    key: IPCCacheEngineKey,
) -> bool:
    """LOOKUP — submit a prefix lookup.

    The server-side handler returns ``None`` (void) on success, so
    we only distinguish RPC timeout from a completed call.
    """
    result = _call(client, RequestType.LOOKUP, [key, 1])
    return result is not _TIMEOUT


def _poll_prefetch_status(
    client: MessageQueueClient,
    request_id: str,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or ``None`` if the polling budget
    is exhausted. The server keys prefetch jobs by ``request_id``
    (str), not an integer job handle.
    """
    for _ in range(max_polls):
        result = _call(
            client,
            RequestType.QUERY_PREFETCH_STATUS,
            [request_id],
        )
        if result is _TIMEOUT:
            # RPC timeout — treat as giving up on this poll cycle.
            return None
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _make_event_handle() -> bytes:
    """Create a CUDA event IPC handle for GPU mode."""
    check_interprocess_event_support()
    event = torch_dev.Event(interprocess=True)
    event.record()
    return event.ipc_handle()


def _send_store(
    client: MessageQueueClient,
    key: IPCCacheEngineKey,
    block_offset: int = 0,
    block_size: int = 16,
) -> str:
    """STORE — store KV cache blocks. Returns status string."""
    num_tokens = key.end - key.start
    num_blocks = num_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [key, _INSTANCE_ID, block_ids, _make_event_handle()]
    result = _call(client, RequestType.STORE, payloads)
    if result is _TIMEOUT:
        return "timeout"
    return "stored" if result[1] else "store_failed"


def _send_retrieve(
    client: MessageQueueClient,
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
        key,
        _INSTANCE_ID,
        block_ids,
        _make_event_handle(),
        0,  # skip_first_n_tokens
    ]
    result = _call(client, RequestType.RETRIEVE, payloads)
    if result is _TIMEOUT:
        return "timeout"
    return "retrieved" if result[1] else "retrieve_failed"


def _send_end_session(
    client: MessageQueueClient,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    _call(client, RequestType.END_SESSION, [request_id])


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
) -> list[str] | None:
    """Query KV cache checksums via the HTTP API.

    Uses the MP-native ``block_ids`` + ``block_size`` addressing
    scheme so the query matches the same block-level semantics
    as ``STORE`` / ``RETRIEVE``. This CLI pins ``layerwise=false``
    so the server always returns ``chunk_checksums`` as a flat
    ``list[str]``. We still defensively validate the response
    type — if a future endpoint variant returns a per-layer
    ``dict`` we log and skip the comparison rather than letting
    ``str.join`` crash.
    """
    blocks = list(range(block_offset, block_offset + num_blocks))
    compressed = compress_slot_mapping(blocks)
    parts: list[str] = []
    for item in compressed:
        if isinstance(item, list):
            parts.append("[%d,%d]" % (item[0], item[1]))
        else:
            parts.append(str(item))
    block_ids = ",".join(parts)
    # The MP /api/kvcache/check endpoint is block-native: its
    # chunk_size counts blocks per chunk, while our caller passes
    # in the server-side token-level chunk_size. Convert here.
    if chunk_size % block_size != 0:
        print(
            "  [WARNING] chunk_size %d not a multiple of block_size %d; "
            "skipping checksum query" % (chunk_size, block_size)
        )
        return None
    chunk_size_blocks = chunk_size // block_size
    url = (
        "%s/api/kvcache/check?block_ids=%s&block_size=%d&chunk_size=%d&layerwise=false"
    ) % (
        http_base,
        block_ids,
        block_size,
        chunk_size_blocks,
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
    client: MessageQueueClient,
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
    if not _send_lookup(client, lookup_key):
        print("  [seq %d/%s] LOOKUP timeout" % (seq_no, pass_label))
        return None

    # 2. QUERY_PREFETCH_STATUS (poll by request_id)
    hit_chunks = _poll_prefetch_status(client, lookup_key.request_id)
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
        checksums = _query_checksum(
            http_base,
            block_offset,
            num_blocks,
            block_size,
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


def _get_chunk_size(client: MessageQueueClient) -> int:
    """Query the server's chunk size."""
    result = _call(client, RequestType.GET_CHUNK_SIZE, [])
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)


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
        # When running on a ``lmcache-cli``-only install ``DTYPE_MAP``
        # is empty; bail out early so users see an actionable error
        # instead of an empty "Supported dtypes:" string.
        _require_full_install()
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
                "KV shape spec. Describes one or more KV layer groups "
                "separated by ';'. "
                "Grammar: "
                "'(kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]'. "
                "Fields: kv_size=2 for classical K/V or 1 for MLA, "
                "NB=num_blocks, BS=block_size (tokens/block), "
                "NH=num_heads, HS=head_size (elements). "
                "dtype is the element dtype (supported: %s); 'uint8' "
                "is used for FP8-quantized KV. 'layers' is the number "
                "of consecutive layers sharing this group's geometry. "
                "Multi-group example (MLA + classical attention): "
                "'(1,1024,16,1,128):float16:4;"
                "(2,1024,16,8,128):float16:28'. "
                "All groups must share the same NB and BS. "
                "See lmcache.v1.kv_layer_groups.parse_kvcache_shape_spec "
                "for the authoritative parser. Default: '%s'"
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
        _require_full_install()
        if not torch_dev.is_available():
            print("ERROR: --mode gpu requires CUDA")
            sys.exit(1)

        url = args.rpc_url
        print("Connecting to LMCache MP Server at %s (mode=%s) ..." % (url, args.mode))

        ctx = zmq.Context()
        client = MessageQueueClient(url, ctx)

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
            # Paged KV demands identical ``NB`` / ``BS`` across all
            # groups (block_id -> slot maths is shared), but
            # ``kv_size`` / ``NH`` / ``HS`` / ``dtype`` may vary per
            # group. ``_allocate_gpu_kv_cache(groups=...)`` honours
            # each group's own shape; ``_process_request`` only needs
            # a single ``block_size`` / ``total_blocks``.
            first = layer_groups[0]
            nb_vals = {g.shape_desc.nb for g in layer_groups}
            bs_vals = {g.shape_desc.bs for g in layer_groups}
            if len(nb_vals) > 1 or len(bs_vals) > 1:
                raise ValueError(
                    "All groups must share NB and BS (paged KV "
                    "requires uniform block geometry). Got NB=%s BS=%s"
                    % (sorted(nb_vals), sorted(bs_vals))
                )
            num_layers = sum(g.num_layers for g in layer_groups)
            spec_nb = getattr(first.shape_desc, "nb", 0) or 0
            spec_bs = getattr(first.shape_desc, "bs", 0) or 0
            num_blocks = spec_nb if spec_nb > 0 else args.num_blocks
            block_size = spec_bs if spec_bs > 0 else args.block_size
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
            # For display / legacy hint fields only: collapse to the
            # first group when homogeneous, otherwise report "mixed".
            heads_set = {g.shape_desc.nh for g in layer_groups}
            hs_set = {g.shape_desc.hs for g in layer_groups}
            kv_size_set = {g.shape_desc.kv_size for g in layer_groups}
            dtype_set = {g.dtype for g in layer_groups}
            num_heads_disp: int | str = (
                first.shape_desc.nh if len(heads_set) == 1 else "mixed"
            )
            head_size_disp: int | str = (
                first.shape_desc.hs if len(hs_set) == 1 else "mixed"
            )
            kv_size_disp: int | str = (
                first.shape_desc.kv_size if len(kv_size_set) == 1 else "mixed"
            )
            if len(dtype_set) == 1:
                dtype_str = next(
                    (k for k, v in DTYPE_MAP.items() if v == first.dtype),
                    "float16",
                )
            else:
                dtype_str = "mixed"

            # Build layout_hints.
            # dtype is sent as a string ("float16") because
            # torch.dtype is not msgpack-serializable. For
            # heterogeneous multi-group specs, per-layer fields
            # (heads / head_size / dtype / kv_size) are reported as
            # ``"mixed"`` — ``layout_hints`` is only consumed by the
            # server to pick a ``kv_layout``, real per-layer shape is
            # discovered from the tensors themselves.
            layout_hints = {
                "num_layers": num_layers,
                "num_heads": num_heads_disp,
                "head_size": head_size_disp,
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
                "KV shape: %d layers, %s heads x %s, "
                "dtype=%s, blocks=%dx%d, kv=%s"
                % (
                    num_layers,
                    num_heads_disp,
                    head_size_disp,
                    dtype_str,
                    num_blocks,
                    block_size,
                    kv_size_disp,
                )
            )

            # Allocate GPU tensors — one tensor per layer, shaped
            # according to that layer's group in the spec (so
            # heterogeneous ``nh`` / ``hs`` / ``dtype`` / ``kv_size``
            # are honoured).
            gpu_tensors = _allocate_gpu_kv_cache(
                groups=layer_groups,
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
            client.close()
            ctx.term()
        print("Done.")

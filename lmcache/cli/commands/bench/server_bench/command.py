# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subcommand implementation.

This module owns the full registration + execution flow for the
end-to-end LMCache MP cache-server sanity test. ``BenchCommand`` only
forwards CLI dispatch to :func:`run_server_bench` and parser
registration to :func:`register_server_parser`.

The command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

Usage examples::

    # GPU mode: real CUDA tensors + IPC
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape (multi-group spec)
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --kvcache-shape-spec '(2,32,1024,8,128):float16:32'

    # Run forever starting from sequence 0
    lmcache bench server --rpc-url tcp://localhost:5555
"""

# Future
from __future__ import annotations

# Standard
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
from typing import TYPE_CHECKING
import argparse
import itertools
import sys
import time

# First Party
from lmcache import torch_dev

# Heavy imports reused by the orchestrator. ``DTYPE_MAP`` is required
# for the ``--kvcache-shape-spec`` help string at parser-registration
# time. On a slim install these symbols are placeholders; the
# ``_require_full_install`` guard inside the helpers module keeps
# orchestration safe.
from lmcache.cli.commands.bench.server_bench.helpers import (
    _DEFAULT_SHAPE_SPEC,
    _IMPORT_ERROR,
    DTYPE_MAP,
    _allocate_kv_cache,
    _get_chunk_size,
    _process_request,
    _require_full_install,
    _send_register_kv_cache,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


# Stash the original (full-install) ImportError so the parser-stub
# branch and the orchestrator branch can both surface it verbatim.
__all__ = (
    "register_server_parser",
    "run_server_bench",
)


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def register_server_parser(
    subparsers: argparse._SubParsersAction,
    dispatch_func,
) -> argparse.ArgumentParser:
    """Register the ``lmcache bench server`` subcommand parser.

    On a slim ``lmcache-cli`` install (where torch / zmq / the MP
    runtime are absent) this still registers a *stub* parser so
    ``lmcache bench --help`` keeps working; the stub defers to
    :func:`run_server_bench`, which prints an actionable install
    hint and exits with status ``1``.

    Args:
        subparsers: The ``bench`` subparsers action.
        dispatch_func: Function to bind via ``set_defaults(func=...)``.
            Typically ``BenchCommand.execute`` so that the outer
            dispatcher can route the call back into
            :func:`run_server_bench`.

    Returns:
        The created ``ArgumentParser`` (mostly for testing).
    """
    if _IMPORT_ERROR is not None:
        # Slim install — register a stub parser only.
        stub = subparsers.add_parser(
            "server",
            help="(requires full lmcache install)",
            description=(
                "End-to-end sanity test for the LMCache MP cache server. "
                "Requires the full `lmcache` package; not available in "
                "the `lmcache-cli` install."
            ),
        )
        stub.set_defaults(func=dispatch_func)
        return stub

    parser = subparsers.add_parser(
        "server",
        help="End-to-end test for LMCache MP cache server (GPU mode).",
        description=(
            "End-to-end sanity test for the LMCache MP cache server: "
            "runs LOOKUP / STORE / RETRIEVE against a live MP server "
            "and verifies KV cache checksums."
        ),
    )

    parser.add_argument(
        "--rpc-url",
        default="tcp://localhost:5555",
        help=("ZMQ endpoint of the MP server (default: tcp://localhost:5555)"),
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help=(
            "Run mode (default: gpu). In cpu mode the bench drives "
            "the data-transfer path: the server allocates a SHM pool "
            "and the client gathers/scatters chunks via slot "
            "descriptors. CPU handle mode is not yet supported."
        ),
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "handle", "data"],
        default="auto",
        help=(
            "Transport routing for STORE/RETRIEVE (default: auto). "
            "`handle` forces the GPU-style single-shot path "
            "(REGISTER_KV_CACHE + STORE/RETRIEVE). "
            "`data` forces the worker-side gather/scatter path "
            "(REGISTER_KV_CACHE_NON_GPU_CONTEXT + PREPARE/COMMIT). "
            "`auto` keeps the historical mapping: gpu->handle, cpu->data."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=512,
        help="Tokens per request (default: 512)",
    )

    # -- KV cache shape --
    kv = parser.add_argument_group("KV cache shape")
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

    parser.set_defaults(func=dispatch_func)
    return parser


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_server_bench(  # noqa: ARG001  (command kept for symmetry with siblings)
    command: "BaseCommand",
    args: argparse.Namespace,
) -> None:
    """Centralized orchestrator: run the server bench loop.

    Args:
        command: The outer ``BenchCommand`` instance. Currently unused
            (server prints directly), but kept for signature
            symmetry with :func:`run_engine_bench` /
            :func:`run_l2_adapter_bench` and to allow future migration
            to ``command.create_metrics``.
        args: Parsed CLI arguments for ``lmcache bench server``.
    """
    _require_full_install()

    # Heavy imports — safe now that _require_full_install passed.
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.kv_layer_groups import (
        format_kvcache_shape_spec,
        parse_kvcache_shape_spec,
    )
    from lmcache.v1.multiprocess.mq import MessageQueueClient

    use_gpu = args.mode == "gpu"
    if use_gpu and not torch_dev.is_available():
        print("ERROR: --mode gpu requires CUDA")
        sys.exit(1)

    # Resolve transfer mode. ``auto`` reproduces the historical
    # behaviour: gpu -> handle path, cpu -> data path.
    transfer_mode = args.transfer_mode
    if transfer_mode == "auto":
        use_handle = use_gpu
    elif transfer_mode == "handle":
        use_handle = True
    else:
        use_handle = False

    url = args.rpc_url
    print(
        "Connecting to LMCache MP Server at %s (mode=%s, transfer=%s) ..."
        % (url, args.mode, transfer_mode),
    )

    ctx = zmq.Context()
    client = MessageQueueClient(url, ctx)
    server_shm: "shared_memory.SharedMemory | None" = None
    server_pool: "memoryview | None" = None

    try:
        # Query chunk size from server
        chunk_size = _get_chunk_size(client)
        print("Server chunk_size = %d" % chunk_size)

        # Parse KV shape spec
        layer_groups = parse_kvcache_shape_spec(args.kvcache_shape_spec)
        # One block-id list is sent per LMCache KV group; each shape-spec
        # group becomes its own group server-side.
        num_engine_group_infos = len(layer_groups) or 1
        # Echo the resolved spec so operators can verify that their
        # input was interpreted as intended. The echoed string is a
        # valid ``--kvcache-shape-spec`` itself.
        print(
            "Resolved KV shape spec: %s" % format_kvcache_shape_spec(layer_groups),
        )
        # Paged KV demands identical ``NB`` / ``BS`` across all groups
        # (block_id -> slot maths is shared), but ``kv_size`` / ``NH`` /
        # ``HS`` / ``dtype`` may vary per group. ``_allocate_kv_cache(
        # groups=...)`` honours each group's own shape; ``_process_request``
        # only needs a single ``block_size`` / ``total_blocks``.
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
        # For display / legacy hint fields only: collapse to the first
        # group when homogeneous, otherwise report "mixed".
        heads_set = {g.shape_desc.nh for g in layer_groups}
        hs_set = {g.shape_desc.hs for g in layer_groups}
        kv_size_set = {g.shape_desc.kv_size for g in layer_groups}
        dtype_set = {g.dtype for g in layer_groups}
        num_heads_disp: int | str = (
            first.shape_desc.nh if len(heads_set) == 1 else "mixed"
        )
        head_size_disp: int | str = first.shape_desc.hs if len(hs_set) == 1 else "mixed"
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

        # Build layout_hints. dtype is sent as a string ("float16")
        # because torch.dtype is not msgpack-serializable. For
        # heterogeneous multi-group specs, per-layer fields (heads /
        # head_size / dtype / kv_size) are reported as "mixed" —
        # ``layout_hints`` is only consumed by the server to pick a
        # ``kv_layout``; the real per-layer shape is discovered from
        # the tensors themselves.
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

        # Allocate KV tensors. GPU mode wraps real CUDA tensors
        # via CUDA IPC; CPU mode (data transfer mode) allocates
        # plain CPU tensors used for client-side checksum self-check.
        # TODO(baoloongmao): CPU handle mode (zero-copy SHM IPC with
        # the server) will be implemented in a separate PR.
        if use_gpu:
            # First Party
            from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper

            allocated = _allocate_kv_cache(groups=layer_groups, use_gpu=True)
            print(
                "Allocated %d GPU tensors on %s" % (len(allocated), allocated[0].device)
            )
            kv_wrappers: list = [CudaIPCWrapper(t) for t in allocated]
            client_kv_tensors = allocated
        else:
            if use_handle:
                print(
                    "ERROR: --mode cpu --transfer-mode handle is not yet "
                    "supported in this PR (TODO: separate PR)."
                )
                sys.exit(1)
            cpu_tensors = _allocate_kv_cache(groups=layer_groups, use_gpu=False)
            print("Allocated %d CPU tensors" % len(cpu_tensors))
            kv_wrappers = []
            client_kv_tensors = cpu_tensors

        # Register KV cache before any store/retrieve.
        register_ok, register_response = _send_register_kv_cache(
            client,
            layout_hints=layout_hints,
            kv_caches=kv_wrappers if use_handle else None,
            use_gpu=use_gpu,
            use_handle=use_handle,
        )
        print("REGISTER_KV_CACHE: %s" % ("OK" if register_ok else "FAIL"))
        print()

        # In data mode the server reply carries the SHM pool name
        # and size; the bench attaches to the same pool so
        # STORE/RETRIEVE can exchange tensor data via slot
        # descriptors. We open via :class:`SharedMemory` (matching
        # the server-side allocator in ``transfer_context/shm.py``)
        # and unregister from the worker's resource tracker so the
        # segment is not unlinked when the bench exits -- the server
        # owns its lifetime.
        if not use_handle and register_ok and register_response is not None:
            shm_name = register_response.shm_name
            pool_size = register_response.pool_size
            if shm_name and pool_size > 0:
                server_shm = shared_memory.SharedMemory(
                    name=shm_name.lstrip("/"), create=False
                )
                try:
                    unregister("/%s" % server_shm.name, "shared_memory")
                except KeyError:
                    pass
                server_pool = server_shm.buf

        if args.end is not None:
            seq_iter: itertools.count | range = range(args.start, args.end)
        else:
            seq_iter = itertools.count(args.start)

        http_base = args.url.rstrip("/")

        # In data mode the server has no paged kv_tensors view to
        # hash, so we self-check on the client. Handle mode keeps
        # the legacy server-side /kvcache/check path.
        client_tensors = None if use_handle else client_kv_tensors

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
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                server_pool=server_pool,
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
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                server_pool=server_pool,
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
                            "    chunk %d: cold=%s warm=%s %s"
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
        # Release the bench-side view of the server SHM pool first
        # (data mode only; server_shm stays None otherwise). The
        # ``memoryview`` returned by ``SharedMemory.buf`` must be
        # released before ``SharedMemory.close``, otherwise CPython
        # raises ``BufferError`` on shutdown.
        if server_pool is not None:
            try:
                server_pool.release()
            except (BufferError, ValueError):
                pass
        if server_shm is not None:
            try:
                server_shm.close()
            except (BufferError, ValueError):
                pass
        client.close()
        ctx.term()
    print("Done.")

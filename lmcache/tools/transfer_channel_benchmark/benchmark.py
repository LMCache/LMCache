# SPDX-License-Identifier: Apache-2.0
"""Core logic for the transfer channel throughput benchmark.

A *server* process uses an :class:`L1MemoryManager` to initialize a registered
L1 region, allocates a pool of source memory objects in it, registers the
region with a transfer channel, and publishes the source object catalog
(offset/size per object) over a small ZMQ side-channel.

A *client* process fetches the catalog, allocates its own objects via an
:class:`L1MemoryManager`, connects the transfer channel, and issues batched
transfers of a random subset of the source objects, reporting aggregate
throughput in GB/s. With ``--op read`` the client pulls the server's objects
into its own buffer; with ``--op write`` it pushes its objects into the
server's buffer. ``--verify`` checks reads in the client's buffer and writes in
the server's buffer (through the same side-channel).
"""

# Standard
import argparse
import json
import random
import time

# Third Party
import zmq

# torch is an optional dependency for this benchmark tool. It (and the
# distributed runtime imported below, which also needs torch) is imported here
# rather than in config.py so the `lmcache` CLI can register this tool without
# torch installed. Raise a clear, actionable error if it is missing.
try:
    # Third Party
    import torch
except ImportError as err:
    raise ImportError(
        "PyTorch is required to run the transfer channel benchmark but is not "
        "installed. Install it (e.g. `pip install torch`) and retry."
    ) from err

# First Party
from lmcache.logging import init_logger
from lmcache.tools.transfer_channel_benchmark.config import (
    BenchmarkConfig,
    build_config,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.distributed.transfer_channel import (
    TransferChannelAddress,
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_CATALOG_REQUEST = b"catalog"
_VERIFY_KEY = "verify"
_LARGE_LAZY_INIT = 20 * 1024**3
_CLIENT_BUFFER_SLACK = 64 * 1024**2


############################################################
# Helpers
############################################################
def _zmq_endpoint(url: str) -> str:
    """Return a ZMQ tcp endpoint for a ``host:port`` (or pass-through url)."""
    return url if "://" in url else f"tcp://{url}"


def _gbps(num_bytes: int, seconds: float) -> float:
    """Return throughput in GB/s (decimal GB) for ``num_bytes`` over ``seconds``."""
    return (num_bytes / 1e9) / seconds if seconds > 0 else float("inf")


def _create_l1_manager(cfg: BenchmarkConfig, size_bytes: int) -> L1MemoryManager:
    """Create an L1 memory manager sized to ``size_bytes`` for this benchmark."""
    manager_config = L1MemoryManagerConfig(
        size_in_bytes=size_bytes,
        use_lazy=cfg.use_lazy,
        init_size_in_bytes=min(_LARGE_LAZY_INIT, size_bytes),
        align_bytes=cfg.page_size,
    )
    return L1MemoryManager(manager_config)


def _mismatched_objects(
    source_objs: list[MemoryObj], expected: dict[int, int]
) -> list[int]:
    """Return the source object indices whose bytes differ from ``expected``.

    Args:
        source_objs: The server's source objects, indexed by catalog index.
        expected: Catalog index -> byte value every byte of that object must
            hold.

    Returns:
        The mismatching indices in ascending order (out-of-range indices count
        as mismatches).
    """
    mismatched: list[int] = []
    for index in sorted(expected):
        value = expected[index]
        if index < 0 or index >= len(source_objs):
            mismatched.append(index)
            continue
        tensor = source_objs[index].tensor
        if tensor is None or not bool((tensor == value).all()):
            mismatched.append(index)
    return mismatched


def _expected_write_pattern(index: int, iteration: int) -> int:
    """Return the byte a write iteration stores into source object ``index``.

    The pattern differs from the server's initial ``index % 256`` on every
    iteration, so a verified write proves the bytes actually landed.
    """
    return (index + iteration + 1) % 256


def _object_layout(cfg: BenchmarkConfig) -> MemoryLayoutDesc:
    """Return the layout for a single ``object_size``-byte uint8 object."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([cfg.object_size])], dtypes=[torch.uint8]
    )


############################################################
# Server
############################################################
def server_main(cfg: BenchmarkConfig) -> None:
    """Run the benchmark server: register a buffer and serve its catalog.

    Args:
        cfg: The benchmark configuration (``role == "server"``).

    Raises:
        RuntimeError: If the source pool cannot be allocated in the buffer.
    """
    pool_bytes = cfg.num_source_objects * cfg.object_size
    if pool_bytes > cfg.buffer_size:
        raise RuntimeError(
            f"source pool ({pool_bytes / 1e9:.2f} GB = {cfg.num_source_objects} "
            f"x {cfg.object_size / 1024**2:.1f} MiB) does not fit in the "
            f"{cfg.buffer_size / 1e9:.2f} GB buffer; increase --buffer-size or "
            f"reduce --num-source-objects / --object-size."
        )

    logger.debug(
        "Allocating L1 buffer (%.2f GB) and %d source objects",
        cfg.buffer_size / 1e9,
        cfg.num_source_objects,
    )
    manager = _create_l1_manager(cfg, cfg.buffer_size)
    source_objs: list[MemoryObj] = []
    try:
        error, source_objs = manager.allocate(
            _object_layout(cfg), cfg.num_source_objects
        )
        if error != L1Error.SUCCESS:
            raise RuntimeError(
                f"Failed to allocate {cfg.num_source_objects} source objects: "
                f"{error}. Increase --buffer-size or reduce "
                f"--num-source-objects / --object-size."
            )

        # Always write a deterministic per-object byte pattern (object index
        # mod 256) so that a client run with --verify can validate the
        # transferred bytes without any extra coordination. This is a one-time
        # cost at startup and does not affect the measured read throughput.
        for index, obj in enumerate(source_objs):
            assert obj.tensor is not None, "Failed to allocate tensor!"
            obj.tensor.fill_(index % 256)

        initialize_transfer_channel_context(
            cfg.transfer_channel_type,
            l1_memory_desc=manager.get_l1_memory_desc(),
            listen_url=cfg.url,
            advertise_url=cfg.url,
            backends=[cfg.nixl_backend],
        )
        try:
            _serve_catalog(cfg, source_objs)
        finally:
            delete_transfer_channel_context()
    finally:
        if source_objs:
            manager.free(source_objs)
        manager.close()


def _serve_catalog(cfg: BenchmarkConfig, source_objs: list[MemoryObj]) -> None:
    """Serve the source object catalog and verify requests until timeout.

    Two request kinds arrive on the ZMQ REP socket: the raw ``catalog`` word,
    answered with the catalog JSON, and a JSON ``{"verify": {index: byte}}``
    object, answered with ``{"ok": bool, "mismatched": [index, ...]}`` after
    checking the named source objects in this process's buffer.
    """
    catalog = [
        (obj.shm_offset, obj.shm_byte_length, index)
        for index, obj in enumerate(source_objs)
    ]
    catalog_payload = json.dumps(
        {
            "page_size": cfg.page_size,
            "object_size": cfg.object_size,
            "objects": catalog,
        }
    ).encode()

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(_zmq_endpoint(cfg.control_url))
    print(
        f"[server] ready: transfer channel on {cfg.url}, "
        f"catalog on {cfg.control_url} ({len(catalog)} source objects)",
        flush=True,
    )

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline = time.monotonic() + cfg.server_timeout
    try:
        while time.monotonic() < deadline:
            events = dict(poller.poll(timeout=1000))
            if socket not in events:
                continue
            request = socket.recv()
            if request == _CATALOG_REQUEST:
                socket.send(catalog_payload)
                continue
            expected = {
                int(index): int(value)
                for index, value in json.loads(request)[_VERIFY_KEY].items()
            }
            mismatched = _mismatched_objects(source_objs, expected)
            print(
                f"[server] verify request: {len(expected)} objects, "
                f"{len(mismatched)} mismatched",
                flush=True,
            )
            socket.send(
                json.dumps({"ok": not mismatched, "mismatched": mismatched}).encode()
            )
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        print("[server] shut down", flush=True)


############################################################
# Client
############################################################
def client_main(cfg: BenchmarkConfig) -> bool:
    """Run the benchmark client: transfer a subset of source objects and report.

    Args:
        cfg: The benchmark configuration (``role == "client"``).

    Returns:
        ``True`` on success.

    Raises:
        ValueError: If the server's page/object size or pool size is
            incompatible with this client's configuration.
        RuntimeError: If allocation, a transfer, or verification fails.
    """
    catalog = _fetch_catalog(cfg)
    if catalog["page_size"] != cfg.page_size:
        raise ValueError(
            f"page_size mismatch: client {cfg.page_size} vs server "
            f"{catalog['page_size']}; they must match."
        )
    if catalog["object_size"] != cfg.object_size:
        raise ValueError(
            f"object_size mismatch: client {cfg.object_size} vs server "
            f"{catalog['object_size']}; they must match."
        )
    source_objs = catalog["objects"]
    if len(source_objs) < cfg.num_objects:
        raise ValueError(
            f"server has only {len(source_objs)} source objects, fewer than "
            f"--num-objects ({cfg.num_objects})."
        )

    client_buffer = cfg.num_objects * cfg.object_size + _CLIENT_BUFFER_SLACK
    manager = _create_l1_manager(cfg, client_buffer)
    local_objs: list[MemoryObj] = []
    try:
        error, local_objs = manager.allocate(_object_layout(cfg), cfg.num_objects)
        if error != L1Error.SUCCESS:
            raise RuntimeError(
                f"Failed to allocate {cfg.num_objects} local objects: {error}."
            )

        ctx = initialize_transfer_channel_context(
            cfg.transfer_channel_type,
            l1_memory_desc=manager.get_l1_memory_desc(),
            listen_url=cfg.listen_url,
            advertise_url=cfg.listen_url,
            backends=[cfg.nixl_backend],
        )
        try:
            local_addrs = ctx.get_transfer_channel_address(
                [(obj.shm_offset, obj.shm_byte_length) for obj in local_objs]
            )
            rng = random.Random(cfg.seed)
            chosen = rng.sample(source_objs, cfg.num_objects)
            remote_addrs = [
                TransferChannelAddress(offset=offset, size=size)
                for offset, size, _ in chosen
            ]
            chosen_indices = [index for _, _, index in chosen]

            connect_start = time.perf_counter()
            client = ctx.get_transfer_channel_client(cfg.url)
            verb = "writing" if cfg.op == "write" else "reading"
            print(
                f"[client] connected in {time.perf_counter() - connect_start:.1f}s; "
                f"{verb} {cfg.num_objects} objects x "
                f"{cfg.object_size / 1024**2:.1f} MiB",
                flush=True,
            )

            def one_transfer(iteration: int) -> float:
                if cfg.op == "write":
                    # Filling the local objects is setup, not transfer time.
                    for obj, index in zip(local_objs, chosen_indices, strict=True):
                        assert obj.tensor is not None, "Failed to allocate tensor!"
                        obj.tensor.fill_(_expected_write_pattern(index, iteration))
                    start = time.perf_counter()
                    task_id = client.submit_write(local_addrs, remote_addrs)
                    result = client.query_write_status(task_id)
                    while not result.is_finished():
                        result = client.query_write_status(task_id)
                else:
                    start = time.perf_counter()
                    task_id = client.submit_read(local_addrs, remote_addrs)
                    result = client.query_read_status(task_id)
                    while not result.is_finished():
                        result = client.query_read_status(task_id)
                elapsed = time.perf_counter() - start
                succeeded = sum(result.succeeded_mask)
                if succeeded != cfg.num_objects:
                    raise RuntimeError(
                        f"{cfg.op} failed: {succeeded}/{cfg.num_objects} objects "
                        "succeeded."
                    )
                return elapsed

            total_iters = cfg.warmup + cfg.iters
            times = [one_transfer(iteration) for iteration in range(total_iters)]
            times = times[cfg.warmup :]

            if cfg.verify and cfg.op == "write":
                _verify_remote(
                    cfg,
                    {
                        index: _expected_write_pattern(index, total_iters - 1)
                        for index in chosen_indices
                    },
                )
            elif cfg.verify:
                _verify(local_objs, chosen_indices)

            _report(cfg, times)
            return True
        finally:
            delete_transfer_channel_context()
            del ctx
    finally:
        if local_objs:
            manager.free(local_objs)
        manager.close()


def _control_request(cfg: BenchmarkConfig, request: bytes, what: str) -> dict:
    """Send one request to the server's control socket and decode the reply.

    Args:
        cfg: The client configuration (``control_url``, ``server_timeout``).
        request: The raw request bytes.
        what: Short description of the request for the timeout error.

    Raises:
        RuntimeError: No reply arrived within ``server_timeout``.
    """
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(cfg.server_timeout * 1000))
    socket.connect(_zmq_endpoint(cfg.control_url))
    try:
        socket.send(request)
        try:
            reply = socket.recv()
        except zmq.Again as exc:
            raise RuntimeError(
                f"timed out {what} from {cfg.control_url}; is the server running?"
            ) from exc
    finally:
        socket.close(linger=0)
    return json.loads(reply)


def _fetch_catalog(cfg: BenchmarkConfig) -> dict:
    """Fetch the source object catalog from the server's control socket."""
    return _control_request(cfg, _CATALOG_REQUEST, "fetching catalog")


def _verify_remote(cfg: BenchmarkConfig, expected: dict[int, int]) -> None:
    """Ask the server to check the written source objects against ``expected``.

    Args:
        cfg: The client configuration.
        expected: Catalog index -> byte value the whole object must hold.

    Raises:
        RuntimeError: The server reports mismatching objects.
    """
    reply = _control_request(
        cfg,
        json.dumps({_VERIFY_KEY: {str(k): v for k, v in expected.items()}}).encode(),
        "verifying written objects",
    )
    if not reply["ok"]:
        raise RuntimeError(
            f"verify FAILED on the server for source objects {reply['mismatched']}."
        )
    print(
        f"[client] verify OK: server confirms {len(expected)} written objects "
        "match expected pattern",
        flush=True,
    )


def _verify(local_objs: list[MemoryObj], chosen_indices: list[int]) -> None:
    """Check each local object holds the expected per-object byte pattern.

    Raises:
        RuntimeError: If any destination object does not match.
    """
    for obj, index in zip(local_objs, chosen_indices, strict=True):
        expected = index % 256
        tensor = obj.tensor
        if tensor is None or not bool((tensor == expected).all()):
            raise RuntimeError(
                f"verify FAILED for source object {index}: expected all bytes "
                f"== {expected}."
            )
    print(
        f"[client] verify OK: {len(local_objs)} objects match expected pattern",
        flush=True,
    )


def _report(cfg: BenchmarkConfig, times: list[float]) -> None:
    """Print best/median/mean latency and throughput for the measured transfers."""
    total_bytes = cfg.num_objects * cfg.object_size
    ordered = sorted(times)
    best = ordered[0]
    median = ordered[len(ordered) // 2]
    mean = sum(ordered) / len(ordered)
    print(f"\n==== Transfer channel {cfg.op} throughput ====", flush=True)
    print(f"  channel type     : {cfg.transfer_channel_type}", flush=True)
    print(
        f"  payload per {cfg.op:<5}: {total_bytes / 1e9:.3f} GB "
        f"({cfg.num_objects} objs x {cfg.object_size / 1024**2:.1f} MiB)",
        flush=True,
    )
    print(f"  iterations       : {cfg.iters} (warmup {cfg.warmup})", flush=True)
    print(
        f"  best   : {best * 1e3:8.2f} ms   {_gbps(total_bytes, best):7.2f} GB/s",
        flush=True,
    )
    print(
        f"  median : {median * 1e3:8.2f} ms   {_gbps(total_bytes, median):7.2f} GB/s",
        flush=True,
    )
    print(
        f"  mean   : {mean * 1e3:8.2f} ms   {_gbps(total_bytes, mean):7.2f} GB/s",
        flush=True,
    )


############################################################
# Entry
############################################################
def run_benchmark(args: argparse.Namespace) -> bool:
    """Dispatch to the server or client role.

    Args:
        args: Parsed CLI arguments produced by ``add_benchmark_arguments``.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    cfg = build_config(args)
    if cfg.role == "server":
        server_main(cfg)
        return True
    return client_main(cfg)

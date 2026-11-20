# SPDX-License-Identifier: Apache-2.0
"""Real MP servers, CPU-only, against a real coordinator, for end-to-end tests.

- :func:`running_coordinator` serves a coordinator on a thread.
- :func:`running_server` starts ``lmcache server`` as a subprocess from the
  test's own interpreter, waits until the coordinator lists it, and records
  its cache events in an ``events``-level trace.
- :class:`KVClient` is one inference worker: it registers a small paged KV
  cache, which on CPU sits in POSIX shared memory, and stores token sequences
  over the server's request port, as ``lmcache bench server --mode cpu`` does.

A test can then compare what a server sent, read from its trace, with what
the coordinator made of it, read over HTTP.
"""

# Standard
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TypeVar
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

# Third Party
import httpx
import torch
import uvicorn
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import EncodedObjectKey, ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import EVENTS_TRACE_BATCH
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.events_replay import EventsTrace
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transfer_context import (
    TransferContext,
    create_transfer_context,
)
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

CHUNK_SIZE = 4
"""Tokens per chunk, on the coordinator and every server alike."""

L1_SIZE_GB = 0.0625
"""L1 per server. A CPU server allocates all of it at startup."""

L1_BACKEND = "dram"
"""The backend name a server's L1 events carry."""

_READY_TIMEOUT_S = 60.0
_STOP_TIMEOUT_S = 15.0
_SETTLE_TIMEOUT_S = 15.0
_POLL_INTERVAL_S = 0.1
_HTTP_TIMEOUT_S = 5.0
_RPC_TIMEOUT_S = 20.0
_LOG_TAIL_LINES = 40
_DIRECTORY_PAGE = 10000

T = TypeVar("T")


def _free_port() -> int:
    """Return a loopback port the kernel just had free."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _answers(url: str) -> bool:
    """Whether ``url`` answers 200."""
    try:
        return httpx.get(url, timeout=1.0).status_code == httpx.codes.OK
    except httpx.HTTPError:
        return False


def encoded(keys: Sequence[ObjectKey]) -> list[dict[str, str | int]]:
    """Render keys as the JSON the delete endpoints take."""
    return [asdict(key.to_encoded_object_key()) for key in keys]


def settle(read: Callable[[], T], expected: T) -> T:
    """Poll ``read`` until it returns ``expected``, or time out.

    A server reports its events on a flush interval, so what the coordinator
    holds catches up with what a test did a moment later.

    Args:
        read: Returns the current value.
        expected: The value to wait for.

    Returns:
        The last value read: ``expected`` unless the timeout passed first, in
        which case the caller's own assertion shows the difference.
    """
    deadline = time.monotonic() + _SETTLE_TIMEOUT_S
    while True:
        current = read()
        if current == expected or time.monotonic() >= deadline:
            return current
        time.sleep(_POLL_INTERVAL_S)


def steady(read: Callable[[], T], interval_s: float) -> T:
    """Read until two reads ``interval_s`` apart agree, or time out.

    For a value a server keeps changing on its own tick, such as L1 contents
    while eviction runs, where :func:`settle` has nothing fixed to wait for.

    Args:
        read: Returns the current value.
        interval_s: Time between reads; longer than the tick being waited out.

    Returns:
        The agreed value, or the last read if the timeout passed first.
    """
    deadline = time.monotonic() + _SETTLE_TIMEOUT_S
    current = read()
    while time.monotonic() < deadline:
        time.sleep(interval_s)
        previous, current = current, read()
        if current == previous:
            break
    return current


def coordinator_config() -> MPCoordinatorConfig:
    """A coordinator config with no timers, chunked like the servers."""
    return MPCoordinatorConfig(
        host="127.0.0.1",
        port=_free_port(),
        chunk_size=CHUNK_SIZE,
        health_check_interval=0.0,
        eviction_check_interval=0.0,
    )


@contextmanager
def running_coordinator(
    config: MPCoordinatorConfig | None = None,
) -> Iterator[str]:
    """Serve a coordinator on a thread for the duration of the block.

    Args:
        config: Its configuration; :func:`coordinator_config` when ``None``.

    Yields:
        Its base URL.

    Raises:
        RuntimeError: If it does not answer ``/healthz`` in time.
    """
    config = config or coordinator_config()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config), host=config.host, port=config.port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://{config.host}:{config.port}"
    try:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while not _answers(f"{url}/healthz"):
            if time.monotonic() >= deadline:
                raise RuntimeError("the coordinator did not come up")
            time.sleep(_POLL_INTERVAL_S)
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=_STOP_TIMEOUT_S)


@dataclass(frozen=True)
class Placement:
    """One copy of a key, as the coordinator's directory lists it.

    Attributes:
        instance_id: The server holding it.
        tier: Its tier.
        backend: Its backend within the tier.
        shared: Whether the backend is fleet-shared.
        key: The object key.
        size_bytes: Its size.
    """

    instance_id: str
    tier: Tier
    backend: str
    shared: bool
    key: ObjectKey
    size_bytes: int


def directory(coordinator_url: str) -> set[Placement]:
    """Every placement the coordinator's directory holds.

    Args:
        coordinator_url: The coordinator's base URL.

    Returns:
        The placements, read over ``GET /directory/keys``.
    """
    held: set[Placement] = set()
    offset = 0
    while True:
        response = httpx.get(
            f"{coordinator_url}/directory/keys",
            params={"offset": offset, "limit": _DIRECTORY_PAGE},
            timeout=_HTTP_TIMEOUT_S,
        )
        response.raise_for_status()
        page = response.json()
        for row in page["keys"]:
            key = EncodedObjectKey(**row["key"]).to_object_key()
            for placement in row["placements"]:
                held.add(
                    Placement(
                        instance_id=placement["instance_id"],
                        tier=Tier(placement["tier"]),
                        backend=placement["backend"],
                        shared=bool(placement.get("shared", False)),
                        key=key,
                        size_bytes=int(placement.get("size_bytes", 0)),
                    )
                )
        offset += len(page["keys"])
        if not page["keys"] or offset >= page["total"]:
            return held


def keys_of(
    model_name: str, token_ids: Sequence[int], cache_salt: str = ""
) -> list[ObjectKey]:
    """The keys a world-size-1 server stores a sequence's whole chunks under.

    Args:
        model_name: The model.
        token_ids: The sequence.
        cache_salt: The tenant.

    Returns:
        One key per whole chunk, in order.
    """
    hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")
    keys, _ = resolve_object_keys(hasher, model_name, 1, list(token_ids), cache_salt)
    return keys


@dataclass(frozen=True)
class L2Backend:
    """One ``--l2-adapter`` entry of a server.

    Deletes that name no adapter reach a server's first L2 backend, so list
    the node-local one first.

    Attributes:
        type_name: The adapter type; also the ``backend`` its events carry.
        shared: Whether the server reports the backend as fleet-shared.
        options: The adapter's own settings.
    """

    type_name: str
    shared: bool = False
    options: Mapping[str, object] = field(default_factory=dict)

    def to_flag(self) -> str:
        """Return the JSON ``--l2-adapter`` takes."""
        fields = {"type": self.type_name, "shared": self.shared, **self.options}
        return json.dumps(fields)


def local_l2() -> L2Backend:
    """A node-local L2 backend: in memory, private to its server."""
    return L2Backend("mock", options={"max_size_gb": 0.0625, "mock_bandwidth_gb": 10})


def shared_l2(path: pathlib.Path) -> L2Backend:
    """A fleet-shared L2 backend: a directory the server reports as shared.

    Args:
        path: The pool's directory; created if missing. Servers sharing the
            pool are given the same path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return L2Backend("fs", shared=True, options={"base_path": str(path)})


class MPServer:
    """One ``lmcache server`` subprocess, reporting to a coordinator.

    Args:
        instance_id: Its identity on the coordinator.
        workdir: Where its log and trace go.
        coordinator_url: The coordinator it registers and reports to.
        l2: Its L2 backends, in ``--l2-adapter`` order.
        l1_size_gb: Its L1 size.
    """

    def __init__(
        self,
        instance_id: str,
        workdir: pathlib.Path,
        coordinator_url: str,
        l2: Sequence[L2Backend],
        l1_size_gb: float = L1_SIZE_GB,
    ) -> None:
        self.instance_id = instance_id
        self._l1_size_gb = l1_size_gb
        self._coordinator_url = coordinator_url
        self._l2 = tuple(l2)
        self._rpc_port = _free_port()
        self._http_port = _free_port()
        self._workdir = workdir
        self._runs = 0
        self._trace_paths: list[pathlib.Path] = []
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def rpc_url(self) -> str:
        """The request port a :class:`KVClient` connects to."""
        return f"tcp://127.0.0.1:{self._rpc_port}"

    @property
    def http_url(self) -> str:
        """The HTTP API the coordinator calls."""
        return f"http://127.0.0.1:{self._http_port}"

    @property
    def log_path(self) -> pathlib.Path:
        """The current run's log."""
        return self._workdir / f"{self.instance_id}-{self._runs}.log"

    def start(self) -> None:
        """Start the process and wait until the coordinator lists it.

        Each start is a new run, with its own log and trace.

        Raises:
            RuntimeError: If it exits or is not registered in time; the
                message carries the end of its log.
        """
        env = dict(
            os.environ,
            LMCACHE_DISABLE_BANNER="1",
            LMCACHE_TRACK_USAGE="false",
            DO_NOT_TRACK="1",
        )
        self._runs += 1
        self._trace_paths.append(self._workdir / f"{self.instance_id}-{self._runs}.lct")
        with self.log_path.open("wb") as log:
            self._process = subprocess.Popen(
                self._command(), env=env, stdout=log, stderr=subprocess.STDOUT
            )
        # Serving HTTP means the engine is up; registering is a background
        # task after that, so wait for the coordinator too.
        self._wait_until(
            lambda: _answers(f"{self.http_url}/healthcheck"), "serve its HTTP API"
        )
        self._wait_until(self._registered, "register with the coordinator")

    def stop(self) -> None:
        """Stop the process: SIGINT, then SIGKILL if it hangs. Idempotent."""
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def restart(self) -> None:
        """Stop the server and start it again as a new incarnation.

        An incarnation is the start time in whole seconds, so a restart
        within the same second would reuse it and have its batches dropped
        as duplicates; the restart waits that second out.
        """
        self.stop()
        time.sleep(1.0)
        self.start()

    def batches(self) -> list[CacheEventBatch]:
        """The batches every run of the server recorded, in order.

        A run's trace is complete once that run has stopped.
        """
        trace = EventsTrace.load([str(path) for path in self._trace_paths])
        wire = [r.args for r in trace.records if r.qualname == EVENTS_TRACE_BATCH]
        return CacheEventsRequest.model_validate({"batches": wire}).batches

    def log_tail(self) -> str:
        """Return the last lines of the server's log."""
        if not self.log_path.exists():
            return ""
        lines = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    def _command(self) -> list[str]:
        """The server's command line."""
        command = [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.http_server",
            "--instance-id",
            self.instance_id,
            "--port",
            str(self._rpc_port),
            "--http-port",
            str(self._http_port),
            "--chunk-size",
            str(CHUNK_SIZE),
            "--l1-size-gb",
            str(self._l1_size_gb),
            "--eviction-policy",
            "LRU",
            "--coordinator-url",
            self._coordinator_url,
            # The default is the host's LAN address, which the loopback-bound
            # HTTP server does not answer on.
            "--coordinator-advertise-ip",
            "127.0.0.1",
            "--coordinator-event-reporting",
            "--coordinator-heartbeat-interval",
            "1",
            "--coordinator-event-flush-interval",
            "0.2",
            "--trace-level",
            "events",
            "--trace-output",
            str(self._trace_paths[-1]),
        ]
        for backend in self._l2:
            command += ["--l2-adapter", backend.to_flag()]
        return command

    def _registered(self) -> bool:
        """Whether the coordinator lists the server."""
        response = httpx.get(
            f"{self._coordinator_url}/instances", timeout=_HTTP_TIMEOUT_S
        )
        listed = response.json()["instances"]
        return any(row["instance_id"] == self.instance_id for row in listed)

    def _wait_until(self, condition: Callable[[], bool], what: str) -> None:
        """Poll ``condition`` until it holds.

        Args:
            condition: The predicate.
            what: What the server is waiting to do, for the error.

        Raises:
            RuntimeError: If the process exits or the timeout passes first.
        """
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.instance_id} exited before it could {what}:\n"
                    f"{self.log_tail()}"
                )
            if condition():
                return
            time.sleep(_POLL_INTERVAL_S)
        raise RuntimeError(
            f"{self.instance_id} did not {what} within {_READY_TIMEOUT_S:.0f}s:\n"
            f"{self.log_tail()}"
        )


@contextmanager
def running_server(
    coordinator_url: str,
    workdir: pathlib.Path,
    instance_id: str,
    l2: Sequence[L2Backend],
    l1_size_gb: float = L1_SIZE_GB,
) -> Iterator[MPServer]:
    """Run one registered server for the duration of the block.

    Args:
        coordinator_url: The coordinator it reports to.
        workdir: Where its log and trace go.
        instance_id: Its identity on the coordinator.
        l2: Its L2 backends; the node-local one first.
        l1_size_gb: Its L1 size.

    Yields:
        The running server. Its trace is complete once the block exits.
    """
    server = MPServer(instance_id, workdir, coordinator_url, l2, l1_size_gb)
    try:
        server.start()
        yield server
    finally:
        server.stop()


class KVClient:
    """One rank-0 worker of a world-size-1 engine, storing into one server.

    One block per chunk, so a sequence of N chunks uses blocks 1 to N: block
    0 is ``--null-block-id``, which the server reads as "no KV here".

    Args:
        rpc_url: The server's request port.
        model_name: The model the cache is registered for.
        heads: KV heads per layer; with ``head_size``, sets a chunk's size.
        head_size: Elements per head.
    """

    _LAYERS = 2
    _BLOCKS = 16
    _DTYPE = torch.float16
    _WORKER_ID = 1000

    def __init__(
        self, rpc_url: str, model_name: str, heads: int = 1, head_size: int = 8
    ) -> None:
        self.model_name = model_name
        self.chunk_bytes = (
            self._LAYERS * 2 * CHUNK_SIZE * heads * head_size * self._DTYPE.itemsize
        )
        """Bytes one stored chunk takes: every layer's K and V for its tokens."""
        self._kv_caches = {
            f"layer.{i}": torch.randn(
                (2, self._BLOCKS, CHUNK_SIZE, heads, head_size), dtype=self._DTYPE
            )
            for i in range(self._LAYERS)
        }
        self._zmq = zmq.Context()
        self._zmq.setsockopt(zmq.LINGER, 0)
        self._client: RequestClient = RequestClientFactory.create(
            rpc_url, context=self._zmq
        )
        self._transfer: TransferContext = create_transfer_context(
            self._kv_caches,
            mode="lmcache_driven",
            instance_id=self._WORKER_ID,
            req_client=self._client,
        )
        self._transfer.register(
            self._kv_caches,
            model_name,
            1,
            1,
            _RPC_TIMEOUT_S,
            layout_hints={"kv_layout": "NHD"},
            engine_group_infos=[
                EngineGroupInfo(
                    engine_group_id=0,
                    layer_indices=tuple(range(self._LAYERS)),
                    tokens_per_block=CHUNK_SIZE,
                )
            ],
            engine_type=EngineType.VLLM,
        )

    def store(self, token_ids: Sequence[int], cache_salt: str = "") -> None:
        """Store a sequence's whole chunks, then end its session.

        A session left open holds read locks that make a later L1 delete
        report the keys as skipped.

        Args:
            token_ids: The sequence; a partial last chunk is not stored.
            cache_salt: The tenant.

        Raises:
            ValueError: If the sequence needs more blocks than the cache has.
            RuntimeError: If the server refused the store.
        """
        chunks = len(token_ids) // CHUNK_SIZE
        if chunks >= self._BLOCKS:
            raise ValueError(
                f"{chunks} chunks need more than the cache's {self._BLOCKS - 1} blocks"
            )
        request_id = uuid.uuid4().hex
        key = IPCCacheServerKey(
            model_name=self.model_name,
            world_size=1,
            worker_id=0,
            token_ids=tuple(token_ids),
            start=0,
            end=chunks * CHUNK_SIZE,
            request_id=request_id,
            cache_salt=cache_salt,
            num_kv_readers=1,
        )
        event = self._transfer.create_recorded_event()
        stored = self._transfer.submit_store(
            request_id, key, self._kv_caches, [list(range(1, chunks + 1))], event, 1
        ).result(timeout=_RPC_TIMEOUT_S)
        self._client.end_session(request_id).result(timeout=_RPC_TIMEOUT_S)
        if not stored:
            raise RuntimeError(f"the server refused to store {chunks} chunks")

    def close(self) -> None:
        """Unregister the cache and disconnect."""
        # ``None`` would mean the cache was never registered; the
        # constructor always registers it.
        unregistered = self._transfer.unregister()
        if unregistered is not None:
            unregistered.result(timeout=_RPC_TIMEOUT_S)
        self._transfer.close()
        self._client.close()
        self._zmq.term()


@contextmanager
def kv_client(
    rpc_url: str, model_name: str, heads: int = 1, head_size: int = 8
) -> Iterator[KVClient]:
    """Open a :class:`KVClient` for the duration of the block.

    Args:
        rpc_url: The server's request port.
        model_name: The model the cache is registered for.
        heads: KV heads per layer.
        head_size: Elements per head.

    Yields:
        The connected client.
    """
    client = KVClient(rpc_url, model_name, heads, head_size)
    try:
        yield client
    finally:
        client.close()

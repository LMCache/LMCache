# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for fs_native restart recovery (#5371).

Each test drives the real MP server CLI (``lmcache.v1.multiprocess.http_server``,
what ``lmcache server`` runs) as a subprocess with an ``fs_native`` L2 adapter:
real KV cache blocks are stored through the request client, written through to
chunk files, the server is restarted over the same ``base_path``, and the result
is read back through the server's own ``/status`` endpoint and by retrieving the
KV data into fresh GPU blocks and comparing it with the original.

Set ``LMCACHE_TEST_ODIRECT_DIR`` to a directory on a filesystem that supports
``O_DIRECT`` (tmpfs and overlayfs usually do not) to also run the O_DIRECT
variant.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from tests.v1.multiprocess.test_cache_server import (
    BLOCKS_PER_KEY,
    CHUNK_SIZE,
    ClientContext,
    _recorded_event_handle,
    create_cache_key,
    lookup_all,
    retrieve_keys,
    store_keys,
)

pytestmark = pytest.mark.cuda

PROJECT_ROOT = Path(__file__).parents[3]
STARTUP_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 60.0
SETTLE_TIMEOUT_S = 30.0
TIMEOUT_S = 20.0
NUM_LAYERS = 4
MODEL = "testmodel"
# Keys are stored in two batches written >1 s apart, so mtime orders them.
OLD_KEYS = list(range(0, 6))
NEW_KEYS = list(range(6, 12))
ALL_KEYS = OLD_KEYS + NEW_KEYS


# =============================================================================
# Server process
# =============================================================================


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Server:
    """A running MP server subprocess and how to reach it."""

    process: subprocess.Popen
    port: int
    http_port: int
    log_path: Path

    def status(self) -> dict:
        url = f"http://127.0.0.1:{self.http_port}/status"
        return json.loads(urllib.request.urlopen(url, timeout=10).read())

    def l2_bytes_tracked(self) -> int:
        adapters = self.status()["storage_manager"]["l2_eviction_controller"][
            "adapters"
        ]
        return adapters[0]["total_bytes_used"]

    def l2_idle(self) -> bool:
        store = self.status()["storage_manager"]["store_controller"]
        return store["pending_keys_count"] == 0 and store["in_flight_task_count"] == 0

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")

    def client(self) -> RequestClient:
        return RequestClientFactory.create(f"tcp://127.0.0.1:{self.port}")


def start_server(base_path: Path, log_dir: Path, **adapter: object) -> Server:
    """Start the MP HTTP server CLI with one fs_native L2 adapter."""
    spec = {
        "type": "fs_native",
        "base_path": str(base_path),
        "num_workers": 4,
        "max_capacity_gb": 1,
        "eviction": {
            "eviction_policy": "LRU",
            "trigger_watermark": 0.9,
            "eviction_ratio": 0.5,
        },
    }
    spec.update(adapter)
    port, http_port = _free_port(), _free_port()
    log_path = log_dir / f"server-{port}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PROJECT_ROOT), env.get("PYTHONPATH")) if p
    )
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--l1-size-gb",
        "0.5",
        "--eviction-policy",
        "LRU",
        "--l2-adapter",
        json.dumps(spec),
    ]
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    server = Server(process, port, http_port, log_path)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"server exited during startup:\n{server.log()}")
        try:
            server.status()
            return server
        except OSError:
            time.sleep(0.3)
    stop_server(server)
    pytest.fail(f"server did not become ready:\n{server.log()}")


def stop_server(server: Server) -> None:
    if server.process.poll() is None:
        server.process.send_signal(signal.SIGTERM)
        try:
            server.process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=10)


# =============================================================================
# Client side: real KV blocks through the request client
# =============================================================================


def key(index: int) -> IPCCacheServerKey:
    return create_cache_key(index, model=MODEL)


def blocks(indices: list[int], offset_keys: int = 0) -> list[int]:
    """GPU block ids for keys stored at positions ``indices`` (+ offset)."""
    return [
        (i + offset_keys) * BLOCKS_PER_KEY + b
        for i in indices
        for b in range(BLOCKS_PER_KEY)
    ]


@dataclass
class Session:
    """A client registered with one server."""

    server: Server
    client: RequestClient
    instance_id: int

    def store(self, indices: list[int]) -> None:
        store_keys(
            self.client,
            [key(i) for i in indices],
            self.instance_id,
            blocks(indices),
            _recorded_event_handle(),
        )

    def found(self, indices: list[int]) -> int:
        return lookup_all(self.client, [key(i) for i in indices])

    def retrieve_to(self, indices: list[int], offset_keys: int) -> list[bool]:
        return retrieve_keys(
            self.client,
            [key(i) for i in indices],
            self.instance_id,
            blocks(indices, offset_keys),
            _recorded_event_handle(),
        )

    def close(self) -> None:
        try:
            self.client.unregister_kv_cache(self.instance_id).result(timeout=TIMEOUT_S)
        finally:
            self.client.close()


def connect(server: Server, ctx: ClientContext) -> Session:
    client = server.client()
    instance_id = os.getpid()
    client.register_kv_cache(
        instance_id, ctx.get_kv_cache(), MODEL, 1, EngineType.VLLM, {}, []
    ).result(timeout=TIMEOUT_S)
    return Session(server, client, instance_id)


def assert_retrieved_exactly(ctx: ClientContext, indices: list[int], offset: int):
    """The KV retrieved into ``offset``-shifted blocks equals the original."""
    for i in indices:
        for layer in range(ctx.num_layers):
            src = ctx.gpu_kv_caches[layer][:, blocks([i])]
            dst = ctx.gpu_kv_caches[layer][:, blocks([i], offset)]
            assert torch.equal(src, dst), f"key {i}, layer {layer}: data differs"


def wait_until(condition, timeout: float = SETTLE_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.2)
    return True


def chunk_files(base: Path) -> dict[str, float]:
    """Canonical chunk files on disk: name -> mtime."""
    return {p.name: p.stat().st_mtime for p in base.glob("*.data") if p.is_file()}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ctx() -> Generator[ClientContext, None, None]:
    """GPU KV cache with distinct content per key position (kept across restarts)."""
    if not (torch_dev.is_available() and torch_device_type == "cuda"):
        pytest.skip("needs a CUDA device")
    context = ClientContext(torch.device(torch_device_type), num_layers=NUM_LAYERS)
    yield context
    del context.gpu_kv_caches
    torch_dev.empty_cache()


def _odirect_supported(path: Path) -> bool:
    probe = path / ".odirect-probe"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
        os.close(fd)
        return True
    except OSError:
        return False
    finally:
        probe.unlink(missing_ok=True)


@pytest.fixture(params=[False, True], ids=["buffered", "odirect"])
def base_path(request, tmp_path) -> Generator[tuple[Path, bool], None, None]:
    """An empty L2 directory, and whether the server should use O_DIRECT."""
    pytest.importorskip("lmcache.lmcache_fs")
    use_odirect = request.param
    if not use_odirect:
        yield tmp_path / "l2", False
        return
    root = os.environ.get("LMCACHE_TEST_ODIRECT_DIR")
    if not root or not _odirect_supported(Path(root)):
        pytest.skip("set LMCACHE_TEST_ODIRECT_DIR to an O_DIRECT-capable directory")
    path = Path(root) / f"lmcache-e2e-{os.getpid()}-{time.monotonic_ns()}"
    yield path, True
    shutil.rmtree(path, ignore_errors=True)


@dataclass
class Seeded:
    """A cache directory written by a first server run, then shut down."""

    path: Path
    use_odirect: bool
    bytes_stored: int
    bytes_per_key: int
    files: dict[str, float]
    log_dir: Path

    def start(self, **adapter: object) -> Server:
        return start_server(
            self.path, self.log_dir, use_odirect=self.use_odirect, **adapter
        )


@pytest.fixture
def seeded(base_path, tmp_path, ctx) -> Seeded:
    """First server run: store OLD_KEYS, wait >1 s, store NEW_KEYS; shut down
    once every chunk is on disk and accounted."""
    path, use_odirect = base_path
    server = start_server(path, tmp_path, use_odirect=use_odirect)
    try:
        session = connect(server, ctx)
        try:
            session.store(OLD_KEYS)
            assert wait_until(lambda: len(chunk_files(path)) == len(OLD_KEYS))
            time.sleep(1.1)  # distinct mtimes for the two batches
            session.store(NEW_KEYS)
            assert wait_until(
                lambda: len(chunk_files(path)) == len(ALL_KEYS) and server.l2_idle()
            ), f"L2 writes did not complete:\n{server.log()}"
            stored = server.l2_bytes_tracked()
        finally:
            session.close()
    finally:
        stop_server(server)
    files = chunk_files(path)
    assert stored == sum((path / f).stat().st_size for f in files)
    return Seeded(path, use_odirect, stored, stored // len(ALL_KEYS), files, tmp_path)


def remaining(seeded: Seeded) -> set[str]:
    """Which of the first run's chunk files are still regular files on disk."""
    return {n for n in seeded.files if (seeded.path / n).is_file()}


def sibling_name(name: str, n: int) -> str:
    """Another canonical chunk file name: the same key fields with a different
    (still lowercase hex) chunk hash."""
    stem, ext = name.rsplit(".", 1)
    head, chunk_hash = stem.rsplit("@", 1)
    flipped = format((int(chunk_hash[-1], 16) + n) % 16, "x")
    return f"{head}@{chunk_hash[:-1]}{flipped}.{ext}"


def _files_of(seeded: Seeded, indices: list[int]) -> set[str]:
    """Chunk files of the given key positions, by write order (mtime)."""
    ordered = sorted(seeded.files, key=seeded.files.__getitem__)
    by_index = dict(zip(ALL_KEYS, ordered, strict=True))
    return {by_index[i] for i in indices}


# =============================================================================
# Happy path
# =============================================================================


class TestRestartHappyPath:
    def test_restart_accounts_and_serves_previous_run(self, seeded, ctx):
        """After a restart the server counts every file from the previous run
        and serves the KV data from them bit-exactly (L1 is empty after the
        restart, so it can only come from L2)."""
        server = seeded.start()
        try:
            assert server.l2_bytes_tracked() == seeded.bytes_stored
            assert "registered 12 objects recovered" in server.log()
            session = connect(server, ctx)
            try:
                assert session.found(ALL_KEYS) == len(ALL_KEYS)
                assert all(session.retrieve_to(ALL_KEYS, offset_keys=20))
            finally:
                session.close()
        finally:
            stop_server(server)
        assert_retrieved_exactly(ctx, ALL_KEYS, offset=20)
        assert chunk_files(seeded.path) == seeded.files

    def test_repeated_restarts_and_restores_do_not_double_count(self, seeded, ctx):
        """Restarting twice, and storing the recovered keys again, keeps usage
        equal to the bytes on disk."""
        for _ in range(2):
            server = seeded.start()
            try:
                assert server.l2_bytes_tracked() == seeded.bytes_stored
                session = connect(server, ctx)
                try:
                    session.store(ALL_KEYS)
                    assert wait_until(server.l2_idle)
                    assert server.l2_bytes_tracked() == seeded.bytes_stored
                finally:
                    session.close()
            finally:
                stop_server(server)
        assert chunk_files(seeded.path).keys() == seeded.files.keys()

    def test_new_writes_after_restart_add_on_top(self, seeded, ctx):
        """New chunks written after a restart are counted on top of the
        recovered ones."""
        new = [30, 31, 32]
        server = seeded.start()
        try:
            session = connect(server, ctx)
            try:
                session.store(new)
                expected = seeded.bytes_stored + len(new) * seeded.bytes_per_key
                assert wait_until(
                    lambda: server.l2_idle() and server.l2_bytes_tracked() == expected
                )
            finally:
                session.close()
        finally:
            stop_server(server)
        assert len(chunk_files(seeded.path)) == len(ALL_KEYS) + len(new)


# =============================================================================
# Eviction of recovered files
# =============================================================================


class TestRecoveredEviction:
    def test_restart_over_cap_evicts_the_older_batch_only(self, seeded, ctx):
        """Restarted with a cap just under what is on disk (usage 0.95 > the
        0.9 watermark, eviction ratio 0.5), the server evicts exactly the six
        oldest files; the newer batch stays and is served bit-exactly."""
        cap_gib = seeded.bytes_stored / 0.95 / 1024**3
        server = seeded.start(max_capacity_gb=cap_gib)
        try:
            assert wait_until(
                lambda: remaining(seeded) == _files_of(seeded, NEW_KEYS)
            ), f"older batch not evicted:\n{server.log()}"
            assert wait_until(
                lambda: server.l2_bytes_tracked()
                == len(NEW_KEYS) * seeded.bytes_per_key
            )
            session = connect(server, ctx)
            try:
                assert session.found(OLD_KEYS) == 0
                assert session.found(NEW_KEYS) == len(NEW_KEYS)
                assert all(session.retrieve_to(NEW_KEYS, offset_keys=20))
            finally:
                session.close()
        finally:
            stop_server(server)
        assert_retrieved_exactly(ctx, NEW_KEYS, offset=20)


# =============================================================================
# Opt-outs: nothing counted, nothing deleted, still served
# =============================================================================


@pytest.mark.parametrize(
    "adapter",
    [{"recover_on_start": False}, {"shared": True}],
    ids=["recover_on_start_false", "shared"],
)
def test_opt_out_leaves_previous_run_untouched(seeded, ctx, adapter):
    """With recovery off, or a shared directory, a cap far below what is on
    disk evicts nothing: files from the previous run are not counted, not
    deleted, and still served (lookup reads the file)."""
    server = seeded.start(max_capacity_gb=1 / 1024**2, **adapter)
    try:
        time.sleep(3)  # three eviction-loop passes
        assert server.l2_bytes_tracked() == 0
        assert chunk_files(seeded.path) == seeded.files
        session = connect(server, ctx)
        try:
            assert session.found(ALL_KEYS) == len(ALL_KEYS)
            assert all(session.retrieve_to(ALL_KEYS, offset_keys=20))
        finally:
            session.close()
    finally:
        stop_server(server)
    assert_retrieved_exactly(ctx, ALL_KEYS, offset=20)


# =============================================================================
# Adversarial directory contents
# =============================================================================


class TestAdversarialDirectory:
    def test_junk_entries_are_ignored_and_never_deleted(self, seeded, ctx):
        """Foreign files, an in-flight .tmp file, a non-canonical name that
        still parses, and a directory and a symlink carrying valid chunk file
        names are neither counted nor evicted, and do not stop the server from
        starting or from evicting the real files."""
        victim = sorted(seeded.files)[0]
        junk = {
            "notes.data": b"not a chunk",
            victim.replace(".data", ".tmp"): b"half-written",
            "model@1@0@aa.data": b"parses, but is not a canonical name",
        }
        for name, payload in junk.items():
            (seeded.path / name).write_bytes(payload)
        fake_dir = seeded.path / sibling_name(victim, 1)
        fake_dir.mkdir()
        outside = seeded.log_dir / "outside.bin"
        outside.write_bytes(b"\0" * seeded.bytes_per_key)
        link = seeded.path / sibling_name(victim, 2)
        link.symlink_to(outside)
        assert {fake_dir.name, link.name}.isdisjoint(seeded.files)

        # A cap just under the real files alone forces one eviction pass.
        cap_gib = seeded.bytes_stored / 0.95 / 1024**3
        server = seeded.start(max_capacity_gb=cap_gib)
        try:
            assert wait_until(
                lambda: remaining(seeded) == _files_of(seeded, NEW_KEYS)
            ), server.log()
            assert wait_until(
                lambda: server.l2_bytes_tracked()
                == len(NEW_KEYS) * seeded.bytes_per_key
            )
            assert server.process.poll() is None
        finally:
            stop_server(server)
        for name, payload in junk.items():
            assert (seeded.path / name).read_bytes() == payload
        assert fake_dir.is_dir() and link.is_symlink() and outside.exists()

    def test_truncated_chunk_file_is_not_served_as_data(self, seeded, ctx):
        """A chunk file truncated by a crash mid-write must not be returned as
        KV data or take the server down; the intact keys are still served."""
        victim_index = NEW_KEYS[0]
        victim = seeded.path / next(iter(_files_of(seeded, [victim_index])))
        with victim.open("r+b") as f:
            f.truncate(victim.stat().st_size // 2)
        intact = [i for i in ALL_KEYS if i != victim_index]

        server = seeded.start()
        try:
            session = connect(server, ctx)
            try:
                # Poison the destination so a partial copy would be visible.
                for layer in range(ctx.num_layers):
                    ctx.gpu_kv_caches[layer][:, blocks([victim_index], 40)] = -1.0
                session.found([victim_index])
                retrieved = session.retrieve_to([victim_index], offset_keys=40)
                assert retrieved == [False] or all(
                    torch.equal(
                        ctx.gpu_kv_caches[layer][:, blocks([victim_index])],
                        ctx.gpu_kv_caches[layer][:, blocks([victim_index], 40)],
                    )
                    for layer in range(ctx.num_layers)
                ), "a truncated chunk was served as (wrong) KV data"
                assert server.process.poll() is None, server.log()
                assert session.found(intact) == len(intact)
                assert all(session.retrieve_to(intact, offset_keys=20))
            finally:
                session.close()
        finally:
            stop_server(server)
        assert_retrieved_exactly(ctx, intact, offset=20)

    @pytest.mark.xfail(
        strict=True,
        reason="Known limitation (PR #5372): a recovered file deleted by "
        "another process stays accounted, because the native delete reports "
        "False for a missing file and its bytes are never released.",
    )
    def test_externally_deleted_recovered_file_releases_its_bytes(self, seeded, ctx):
        """A recovered file removed by someone else should stop counting once
        eviction tries to delete it."""
        extra = [40, 41, 42]
        # 12 recovered keys sit at 0.8 of this cap; 15 keys reach 1.0 > 0.9.
        cap_gib = len(ALL_KEYS) * seeded.bytes_per_key / 0.8 / 1024**3
        server = seeded.start(max_capacity_gb=cap_gib)
        try:
            assert server.l2_bytes_tracked() == seeded.bytes_stored
            for name in _files_of(seeded, OLD_KEYS):
                (seeded.path / name).unlink()  # deleted behind the server's back
            session = connect(server, ctx)
            try:
                session.store(extra)  # pushes usage over the watermark
            finally:
                session.close()
            time.sleep(4)  # a few eviction passes
            on_disk = sum(p.stat().st_size for p in seeded.path.glob("*.data"))
            assert server.l2_bytes_tracked() == on_disk
        finally:
            stop_server(server)

# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the offload server's socket rank in ``VllmServiceFactory``.

The offload server binds ``engine_<id>_service_offload_lmcache_rpc_port_<100 + rank>``.
It used the tensor-parallel rank, which is 0 on every worker when the engine
runs context parallelism with TP=1, so all workers bound the same ipc path. When
two bound at the same moment one got ``ZMQError: Address already in use``, which
failed ``LMCacheManager`` init for that worker (degraded mode: every lookup then
waited for the lookup timeout).

At most one real ``ZMQOffloadServer`` is started per test: closing several live
servers in one process aborts libzmq (``Assertion failed: pfd.revents & POLLIN``),
independently of this change. Tests that need one socket per worker record the
path a server would bind, or bind those paths with plain zmq sockets.
"""

# Standard
from types import SimpleNamespace
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import zmq

pytest.importorskip("vllm", reason="VllmServiceFactory imports vLLM")

# Third Party
import vllm.distributed.parallel_state as ps  # noqa: E402

# First Party
from lmcache.integration.vllm import vllm_service_factory as factory_mod  # noqa: E402
from lmcache.integration.vllm.vllm_service_factory import (  # noqa: E402
    VllmServiceFactory,
)
from lmcache.v1.offload_server import zmq_server  # noqa: E402

NUM_WORKERS = 8
ENGINE_ID = "test-engine"


@pytest.fixture(autouse=True)
def rpc_dir(monkeypatch):
    # A short directory: longer socket paths are hashed (see rpc_utils).
    d = tempfile.mkdtemp(prefix="ofs", dir="/tmp")
    monkeypatch.setenv("VLLM_RPC_BASE_PATH", d)
    # Context parallelism with TP=1: every worker reports tensor-parallel rank 0.
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
    yield
    shutil.rmtree(d, ignore_errors=True)


def _socket_path(rank):
    """The ipc socket file ZMQOffloadServer binds for this rank."""
    port = int(os.environ.get("LMCACHE_OFFLOAD_RPC_PORT", 100))
    return zmq_server.get_zmq_rpc_path_lmcache(ENGINE_ID, "offload", port, rank)


@pytest.fixture
def recorded(monkeypatch):
    """Replaces ZMQOffloadServer with a stub that records the rank it is given."""
    ranks = []

    def stub(engine, rank):
        ranks.append(rank)
        return SimpleNamespace(rank=rank, close=lambda: None)

    monkeypatch.setattr(factory_mod, "ZMQOffloadServer", stub)
    return ranks


def _factory(role="worker"):
    engine = SimpleNamespace(metadata=SimpleNamespace(engine_id=ENGINE_ID))
    return SimpleNamespace(
        role=role, _ensure_engine=lambda: None, lmcache_engine=engine
    )


def _create(monkeypatch, world_rank):
    monkeypatch.setattr(
        ps, "get_world_group", lambda: SimpleNamespace(rank_in_group=world_rank)
    )
    return VllmServiceFactory.maybe_create_offload_server(_factory())


def test_each_worker_gets_its_own_socket(monkeypatch, recorded):
    servers = [_create(monkeypatch, rank) for rank in range(NUM_WORKERS)]
    assert all(s is not None for s in servers)
    assert recorded == list(range(NUM_WORKERS))
    assert len({_socket_path(rank) for rank in recorded}) == NUM_WORKERS


def test_simultaneous_binds_of_worker_paths_succeed():
    # The failure was a simultaneous bind of one path; distinct per-rank paths
    # must all bind when every worker binds at once.
    paths = [_socket_path(rank) for rank in range(NUM_WORKERS)]
    assert len(set(paths)) == NUM_WORKERS
    ctx = zmq.Context()
    sockets = [None] * NUM_WORKERS
    errors = []
    start = threading.Barrier(NUM_WORKERS)

    def run(rank):
        try:
            sock = ctx.socket(zmq.REP)
            sock.setsockopt(zmq.LINGER, 0)
            sockets[rank] = sock
            start.wait()
            sock.bind(f"ipc://{paths[rank]}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors, errors
    finally:
        for sock in sockets:
            if sock is not None:
                sock.close()
        ctx.term()


def test_real_server_binds_the_world_rank_path(monkeypatch):
    server = _create(monkeypatch, 5)
    try:
        assert server is not None
        assert os.path.exists(_socket_path(5))
    finally:
        server.close()


def test_bind_failure_does_not_raise(monkeypatch):
    def fail(*args, **kwargs):
        raise zmq.ZMQError(zmq.EADDRINUSE)

    monkeypatch.setattr(factory_mod, "ZMQOffloadServer", fail)
    assert _create(monkeypatch, 3) is None


def test_falls_back_to_tp_rank_without_world_group(monkeypatch, recorded):
    def no_group():
        raise AssertionError("world group is not initialized")

    monkeypatch.setattr(ps, "get_world_group", no_group)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 5)
    assert VllmServiceFactory.maybe_create_offload_server(_factory()) is not None
    assert recorded == [5]


def test_non_worker_role_has_no_offload_server():
    assert VllmServiceFactory.maybe_create_offload_server(_factory("scheduler")) is None

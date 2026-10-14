# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the offload server's socket rank in ``VllmServiceFactory``.

The offload server binds ``engine_<id>_service_offload_lmcache_rpc_port_<100 + rank>``.
It used the tensor-parallel rank, which is 0 on every worker when the engine
runs context parallelism with TP=1, so all workers bound the same ipc path. When
two bound at the same moment one got ``ZMQError: Address already in use``, which
failed ``LMCacheManager`` init for that worker (degraded mode: every lookup then
waited for the lookup timeout).

These tests use the real ``ZMQOffloadServer`` with real ipc sockets; the engine
and vLLM's rank functions are stand-ins.
"""

# Standard
from types import SimpleNamespace
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


@pytest.fixture(autouse=True)
def rpc_dir(monkeypatch):
    # A short directory: longer socket paths are hashed (see rpc_utils).
    d = tempfile.mkdtemp(prefix="ofs", dir="/tmp")
    monkeypatch.setenv("VLLM_RPC_BASE_PATH", d)
    # Context parallelism with TP=1: every worker reports tensor-parallel rank 0.
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 0)
    yield
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def bound(monkeypatch):
    """Records (rank, path) for every offload server socket that is created."""
    calls = []
    real = zmq_server.get_zmq_rpc_path_lmcache

    def spy(engine_id, service_name, rpc_port, rank, *args, **kwargs):
        path = real(engine_id, service_name, rpc_port, rank, *args, **kwargs)
        calls.append((rank, path))
        return path

    monkeypatch.setattr(zmq_server, "get_zmq_rpc_path_lmcache", spy)
    return calls


def _factory(role="worker"):
    engine = SimpleNamespace(metadata=SimpleNamespace(engine_id="test-engine"))
    return SimpleNamespace(
        role=role, _ensure_engine=lambda: None, lmcache_engine=engine
    )


def _create(monkeypatch, world_rank):
    monkeypatch.setattr(
        ps, "get_world_group", lambda: SimpleNamespace(rank_in_group=world_rank)
    )
    return VllmServiceFactory.maybe_create_offload_server(_factory())


def test_each_worker_gets_its_own_socket(monkeypatch, bound):
    servers = [_create(monkeypatch, rank) for rank in range(NUM_WORKERS)]
    try:
        assert all(s is not None for s in servers)
        assert [rank for rank, _ in bound] == list(range(NUM_WORKERS))
        assert len({path for _, path in bound}) == NUM_WORKERS
    finally:
        for s in servers:
            s.close()


def test_simultaneous_binds_with_distinct_ranks_succeed(bound):
    servers = [None] * NUM_WORKERS
    errors = []
    start = threading.Barrier(NUM_WORKERS)

    def run(rank):
        try:
            start.wait()
            servers[rank] = zmq_server.ZMQOffloadServer(_factory().lmcache_engine, rank)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors, errors
        assert len({path for _, path in bound}) == NUM_WORKERS
    finally:
        for s in servers:
            if s is not None:
                s.close()


def test_bind_failure_does_not_raise(monkeypatch):
    def fail(*args, **kwargs):
        raise zmq.ZMQError(zmq.EADDRINUSE)

    monkeypatch.setattr(factory_mod, "ZMQOffloadServer", fail)
    assert _create(monkeypatch, 3) is None


def test_falls_back_to_tp_rank_without_world_group(monkeypatch, bound):
    def no_group():
        raise AssertionError("world group is not initialized")

    monkeypatch.setattr(ps, "get_world_group", no_group)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 5)
    server = VllmServiceFactory.maybe_create_offload_server(_factory())
    try:
        assert server is not None
        assert bound[-1][0] == 5
    finally:
        server.close()


def test_non_worker_role_has_no_offload_server():
    assert VllmServiceFactory.maybe_create_offload_server(_factory("scheduler")) is None

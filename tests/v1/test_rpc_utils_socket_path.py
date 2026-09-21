# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for IPC socket path length handling in ``rpc_utils``.

Regression for https://github.com/LMCache/LMCache/issues/3529: on shared
filesystems (e.g. a supercomputer scratch dir assigned via ``TMPDIR``) the
generated IPC socket path

    {base_url}/engine_{uuid}_service_lookup_lmcache_rpc_port_{n}

can exceed the 107-char ``sockaddr_un.sun_path`` limit, and ZMQ raises
``zmq.error.ZMQError: ipc path "..." is longer than 107 characters``. These
tests pin that the path is kept within the limit while staying deterministic
across the binding server and the connecting client.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.rpc_utils import (
    IPC_SOCKET_PATH_MAX_LEN,
    get_zmq_rpc_path_lmcache,
)

# A realistic long scratch path, modeled on the issue report
# (/p/scratch/<project>/<user>/<model>/tmp on a JSC supercomputer).
LONG_BASE_URL = "/p/scratch/chpsadm/strube1/mistral-small-3.1-24b-lmcache/tmp"
# A full UUID engine id, as produced in practice.
UUID_ENGINE_ID = "f0edd86e-949d-4be4-9ab5-c2279be30be5"


def test_short_path_is_returned_unchanged():
    """The common case (short base_url) keeps the readable descriptive path
    for debuggability — no hashing, full backwards compatibility."""
    path = get_zmq_rpc_path_lmcache(
        engine_id="abc",
        service_name="lookup",
        rpc_port=1,
        rank=0,
        base_url="/tmp/vllm_rpc",
    )
    assert path == "/tmp/vllm_rpc/engine_abc_service_lookup_lmcache_rpc_port_1"
    assert len(path) <= IPC_SOCKET_PATH_MAX_LEN


def test_long_path_is_shortened_within_limit():
    """The reported failure case: long base_url + UUID engine id. The result
    must fit the sockaddr_un limit and stay under the requested base_url."""
    path = get_zmq_rpc_path_lmcache(
        engine_id=UUID_ENGINE_ID,
        service_name="lookup",
        rpc_port=1,
        rank=0,
        base_url=LONG_BASE_URL,
    )
    assert len(path) <= IPC_SOCKET_PATH_MAX_LEN
    assert path.startswith(LONG_BASE_URL + "/")


def test_shortening_is_deterministic_across_ends():
    """Both the binding server and the connecting client call this helper with
    identical inputs; they must derive byte-identical paths or IPC silently
    fails to connect."""
    kwargs = dict(
        engine_id=UUID_ENGINE_ID,
        service_name="lookup_worker",
        rpc_port=5,
        rank=3,
        base_url=LONG_BASE_URL,
    )
    first = get_zmq_rpc_path_lmcache(**kwargs)
    second = get_zmq_rpc_path_lmcache(**kwargs)
    assert first == second


def test_distinct_inputs_yield_distinct_shortened_paths():
    """Shortening must not collapse different sockets onto one path: distinct
    service/rank/port must map to distinct files even after hashing."""
    base = dict(
        engine_id=UUID_ENGINE_ID,
        rpc_port=1,
        rank=0,
        base_url=LONG_BASE_URL,
    )
    worker = get_zmq_rpc_path_lmcache(service_name="lookup_worker", **base)
    scheduler = get_zmq_rpc_path_lmcache(service_name="lookup_scheduler", **base)
    rank_diff = get_zmq_rpc_path_lmcache(
        engine_id=UUID_ENGINE_ID,
        service_name="lookup_worker",
        rpc_port=1,
        rank=7,
        base_url=LONG_BASE_URL,
    )
    paths = {worker, scheduler, rank_diff}
    assert len(paths) == 3
    for p in paths:
        assert len(p) <= IPC_SOCKET_PATH_MAX_LEN


def test_pathological_base_url_raises_clear_error():
    """If base_url alone cannot host even the shortened name, raise a clear
    error rather than silently falling back to a different directory: the
    binding server and connecting client may have different TMPDIR, so a
    temp-dir fallback could resolve differently on each end and break IPC.
    Raising is deterministic on both ends."""
    pathological = "/" + "x" * (IPC_SOCKET_PATH_MAX_LEN + 20)
    with pytest.raises(ValueError, match="too long"):
        get_zmq_rpc_path_lmcache(
            engine_id=UUID_ENGINE_ID,
            service_name="lookup",
            rpc_port=1,
            rank=0,
            base_url=pathological,
        )


def test_limit_measured_in_bytes_not_chars():
    """``sun_path`` is byte-sized: a non-ASCII base_url whose descriptive path
    is under the limit in *characters* but over it in *UTF-8 bytes* must still
    be shortened, or it would pass a char-based guard and then fail inside ZMQ.
    """
    # "/tmp/" + 30×"é" = 35 chars but 65 bytes. The descriptive path is ~80
    # chars (<=107, a char check would leave it unshortened) yet ~130 bytes
    # (>107, so the byte check must shorten it).
    base_url = "/tmp/" + "é" * 30
    descriptive = f"{base_url}/engine_abc_service_lookup_lmcache_rpc_port_1"
    assert len(descriptive) <= IPC_SOCKET_PATH_MAX_LEN  # passes a char check
    assert len(descriptive.encode("utf-8")) > IPC_SOCKET_PATH_MAX_LEN  # fails bytes

    path = get_zmq_rpc_path_lmcache(
        engine_id="abc",
        service_name="lookup",
        rpc_port=1,
        rank=0,
        base_url=base_url,
    )
    assert path != descriptive  # was shortened
    assert len(path.encode("utf-8")) <= IPC_SOCKET_PATH_MAX_LEN


def test_string_rpc_port_with_rank_is_handled():
    """``rpc_port`` may arrive as a str; the function appends the rank to it.
    Ensure the long-path branch still produces a valid bounded path."""
    path = get_zmq_rpc_path_lmcache(
        engine_id=UUID_ENGINE_ID,
        service_name="lookup",
        rpc_port="9000",
        rank=2,
        base_url=LONG_BASE_URL,
    )
    assert len(path) <= IPC_SOCKET_PATH_MAX_LEN


def test_invalid_service_name_rejected():
    """Guard the existing validation still fires (no regression)."""
    with pytest.raises(ValueError):
        get_zmq_rpc_path_lmcache(
            engine_id="abc",
            service_name="not_a_service",  # type: ignore[arg-type]
            rpc_port=1,
            rank=0,
            base_url="/tmp/vllm_rpc",
        )

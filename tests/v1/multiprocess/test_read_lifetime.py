# SPDX-License-Identifier: Apache-2.0
"""Named attempt identity across RPC, result handoff, and request cleanup."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import call

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey, PrefetchResult
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Test helpers
from tests.v1.mp_scheduler_adapter_test_utils import (
    make_request_client,
    make_scheduler_adapter,
    ready_future,
)
from tests.v1.multiprocess.test_lookup_load_policy import _ctx, _key
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)


def named_key(identity: str = "attempt-A") -> IPCCacheServerKey:
    """Return independently scoped attempts for the same external request."""
    return replace(_key(None), lookup_id=identity)


def test_session_starts_a_named_attempt_once() -> None:
    """Concurrent duplicate submissions cannot reserve the same shares twice."""
    session = Session("attempt-A", TokenHasher(chunk_size=256))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: session.begin_lookup(named_key()), range(16)))
    assert sum(results) == 1


@pytest.mark.parametrize("cleanup_first", [False, True])
def test_result_handoff_and_cleanup_have_one_release_owner(cleanup_first: bool) -> None:
    """Either the result or the destroyed session owns cleanup, never neither."""
    session = Session("attempt-A", TokenHasher(chunk_size=256))
    session.begin_lookup(named_key())
    owners = {ObjectKey(ObjectKey.IntHash2Bytes(1), "m", 0): 7}
    if cleanup_first:
        assert session.close_read_ownership() == {}
        assert not session.record_prefetch_result(3, (0,), owners)
    else:
        assert session.record_prefetch_result(3, (0,), owners)
        assert session.close_read_ownership() == owners
    assert session.close_read_ownership() == {}
    assert not session.record_prefetch_result(3, (0,), owners)


def test_cancelled_result_is_discarded_without_resurrecting_session() -> None:
    """A late result releases only its own named shares after END_SESSION."""
    ctx = _ctx()
    sessions = SessionManager(TokenHasher(chunk_size=256), cleanup_interval=None)
    ctx.session_manager = sessions
    module = LookupModule(ctx)
    owners = {ObjectKey(ObjectKey.IntHash2Bytes(1), "m", 0): 7}
    try:
        module.lookup(named_key(), 1)
        spec = ctx.storage_manager.submit_prefetch_task.call_args.args[0]
        assert spec.read_owner == "attempt-A"
        module.end_session("req-1", "attempt-A")
        assert sessions.get("attempt-A") is None
        ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
            hit_cells=[Bitmap(3, 3)],
            l1_hit_cells=[Bitmap(3, 3)],
            l2_hit_cells=[Bitmap(3)],
            l1_owners=owners,
        )
        assert module.query_prefetch_status("req-1", "attempt-A") == 0
        ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
            list(owners),
            l1_owners=owners,
            read_owner="attempt-A",
        )
        assert sessions.get("attempt-A") is None
        assert module.report_status()["active_prefetch_jobs"] == 0
    finally:
        module.close()
        sessions.close()


def test_cancel_old_lookup_preserves_new_lookup_of_same_request() -> None:
    """Session cleanup targets the exact attempt even when request_id is reused."""
    ctx = _ctx()
    sessions = SessionManager(TokenHasher(chunk_size=256), cleanup_interval=None)
    ctx.session_manager = sessions
    module = LookupModule(ctx)
    try:
        for identity in ("attempt-A", "attempt-B"):
            module.lookup(named_key(identity), 1)
        a = sessions.get("attempt-A")
        b = sessions.get("attempt-B")
        assert a is not None and b is not None
        owners = {ObjectKey(ObjectKey.IntHash2Bytes(1), "m", 0): 7}
        a.record_prefetch_result(3, (0,), owners)
        b.record_prefetch_result(3, (0,), owners)
        module.end_session("req-1", "attempt-A")
        assert sessions.get("attempt-A") is None
        assert sessions.get("attempt-B") is b
        assert b.get_prefetch_owners() == owners
        ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
            list(owners),
            l1_owners=owners,
            read_owner="attempt-A",
        )
        module.end_session("req-1", "attempt-B")
    finally:
        module.close()
        sessions.close()


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
def test_named_identity_survives_request_transports(
    transport: RequestTransport,
) -> None:
    """Lookup, status, free and end carry the same identity over ZMQ and gRPC."""
    ctx = _ctx()
    sessions = SessionManager(TokenHasher(chunk_size=256), cleanup_interval=None)
    ctx.session_manager = sessions
    module = LookupModule(ctx)
    management = ManagementModule(ctx)
    url = request_server_url(transport, 15980 if transport == "zmq" else 15981)
    server = create_request_server(
        [module, management], request_server_config(transport, url)
    )
    server.start()
    client = RequestClientFactory.create(url)
    owners = {ObjectKey(ObjectKey.IntHash2Bytes(1), "m", 0): 7}
    try:
        assert client.get_read_lifetime_version().result(timeout=5) == 1
        client.lookup(named_key(), 1).result(timeout=5)
        assert sessions.get("attempt-A") is not None
        ctx.storage_manager.query_prefetch_status.return_value = PrefetchResult(
            hit_cells=[Bitmap(3, 3)],
            l1_hit_cells=[Bitmap(3, 3)],
            l2_hit_cells=[Bitmap(3)],
            l1_owners=owners,
        )
        assert (
            client.wait_prefetch_status("req-1", 1.0, "attempt-A").result(timeout=5)
            == 3
        )
        client.free_lookup_locks(named_key(), 1).result(timeout=5)
        assert (
            ctx.storage_manager.finish_read_prefetched.call_args.kwargs["read_owner"]
            == "attempt-A"
        )
        client.end_session("req-1", "attempt-A").result(timeout=5)
        assert sessions.get("attempt-A") is None
    finally:
        client.close()
        server.close()
        module.close()
        sessions.close()


def test_scheduler_preserves_identity_after_result_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attempt remains available for END_SESSION after metadata handoff."""
    url = "tcp://test"
    client = make_request_client(256)
    client.get_read_lifetime_version.return_value = ready_future(1)
    client.lookup.return_value = ready_future(None)
    client.query_prefetch_status.return_value = ready_future(1)
    adapter = make_scheduler_adapter(
        monkeypatch,
        {url: client},
        16,
        extra_config={"lmcache.mp.read_lifetime_v1": True},
    )
    try:
        tokens = pack_token_ids(range(512))
        assert adapter.maybe_submit_lookup_request(
            "req", packed_token_ids=tokens, lookup_id="A"
        )
        sent_key = client.lookup.call_args.args[0]
        assert sent_key.lookup_id == "A"
        assert adapter.check_lookup_result("req") == 256
        client.query_prefetch_status.assert_called_with("req", "A")
        adapter.cleanup_lookup_result("req")
        adapter.end_session("req")
        assert call("req", "A") in client.end_session.call_args_list
    finally:
        adapter.shutdown()


def test_scheduler_rejects_unsupported_lifetime_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opting in cannot silently use an incompatible server."""
    client = make_request_client(256)
    client.get_read_lifetime_version.return_value = ready_future(0)
    with pytest.raises(RuntimeError, match="does not support"):
        make_scheduler_adapter(
            monkeypatch,
            {"tcp://old": client},
            16,
            extra_config={"lmcache.mp.read_lifetime_v1": True},
        )

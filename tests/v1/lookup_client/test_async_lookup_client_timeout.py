# SPDX-License-Identifier: Apache-2.0
"""Regression test for issue #3685.

When ``enable_async_loading`` is set (e.g. together with the nixl backend), an
async lookup that exceeds ``lookup_timeout_ms`` is cancelled and
``LMCacheAsyncLookupClient.lookup_cache`` returns 0 so vLLM recomputes. The
timeout handler used to pop only ``first_lookup_time`` while leaving
``reqs_status[lookup_id] = None``. The next scheduler call for the same
``lookup_id`` then re-entered the ``req_status is None`` branch and raised
``KeyError`` on the already-popped ``first_lookup_time[lookup_id]``, crashing
the vLLM EngineCore.
"""

# Standard
import threading
import time
import types

# First Party
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)


def _make_client(timeout_ms: int = 50) -> LMCacheAsyncLookupClient:
    """Build a client without the zmq/socket-heavy ``__init__``.

    Only the attributes touched by ``lookup_cache`` / ``cancel_lookup`` /
    ``_cleanup_finished_aborted_lookups`` are wired up.
    """
    client = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
    client.lock = threading.Lock()
    client.reqs_status = {}
    client.first_lookup_time = {}
    client.aborted_lookups = set()
    client.lookup_backoff_time = 0.0
    client.config = types.SimpleNamespace(lookup_timeout_ms=timeout_ms)
    return client


def test_lookup_cache_recall_after_timeout_does_not_keyerror() -> None:
    client = _make_client(timeout_ms=50)
    lookup_id = "req-nixl-async"

    # 1) First call registers the in-flight async lookup.
    assert client.lookup_cache(lookup_id) == -1

    # 2) Make the deadline already passed, then recall: the timeout branch
    #    cancels the lookup and returns 0 so vLLM recomputes.
    client.first_lookup_time[lookup_id] = time.time() - 10.0
    assert client.lookup_cache(lookup_id) == 0

    # 3) Recall again with the SAME id. Pre-fix this raised
    #    ``KeyError: 'req-nixl-async'`` because first_lookup_time[id] had been
    #    popped while reqs_status[id] was still None, so the call re-entered the
    #    timeout branch and dereferenced the missing key. After the fix the
    #    call keeps telling vLLM to recompute (returns 0) without crashing and
    #    without re-registering the id while its abort is still pending.
    assert client.lookup_cache(lookup_id) == 0
    assert client.lookup_cache(lookup_id) == 0

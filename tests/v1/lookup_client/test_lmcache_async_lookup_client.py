# SPDX-License-Identifier: Apache-2.0
"""
Tests for LMCacheAsyncLookupClient.

These tests drive the client's lookup state machine directly through its
public ``lookup_cache`` interface. No lookup server is started, so a request's
status stays ``None`` (ongoing) and we can deterministically exercise the
timeout path without depending on cache contents or hashing.
"""

# Standard
import time
import uuid

# First Party
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)
from tests.v1.utils import create_test_config, create_test_metadata


def test_lookup_cache_requery_after_timeout_does_not_crash() -> None:
    """Regression test for a post-timeout re-query crash.

    When a lookup times out, ``first_lookup_time`` is popped while
    ``reqs_status`` stays ``None`` (still pending). A subsequent
    ``lookup_cache`` call for the same id then re-entered the "ongoing"
    branch and indexed ``first_lookup_time[lookup_id]`` directly, raising
    ``KeyError`` and bringing vLLM down (see nixl + async-loading reports).

    A re-query after timeout must instead report a cache miss (0) so vLLM
    recomputes, not crash.
    """
    engine_id = f"test_async_lookup_{uuid.uuid4().hex[:8]}"
    config = create_test_config()
    # Tiny timeout so the second lookup_cache call expires immediately.
    config.lookup_timeout_ms = 1
    metadata = create_test_metadata(engine_id=engine_id)

    client = LMCacheAsyncLookupClient(config, metadata)
    try:
        lookup_id = "req-timeout-then-requery"

        # First call registers the request as ongoing (status None).
        assert client.lookup_cache(lookup_id) == -1

        # Let the 1ms timeout elapse; this call should expire and return 0,
        # popping first_lookup_time while reqs_status stays None.
        time.sleep(0.05)
        assert client.lookup_cache(lookup_id) == 0

        # Status is still None but first_lookup_time is now gone. The
        # re-query must not raise KeyError; it should report a miss (0).
        assert client.lookup_cache(lookup_id) == 0
    finally:
        client.close()

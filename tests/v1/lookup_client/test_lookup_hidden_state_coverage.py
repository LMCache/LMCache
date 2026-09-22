# SPDX-License-Identifier: Apache-2.0
"""Wire compatibility for the optional hidden-state coverage field.

The reply grew a second 4-byte field, appended rather than substituted, so a
client and a server on different versions still understand each other. These
build the client without an engine and feed it replies directly.
"""

# First Party
from lmcache.v1.lookup_client.lmcache_lookup_client import (
    HS_LAYER_IDXS_CONFIG,
    LMCacheLookupClient,
)
from lmcache.v1.lookup_client.lmcache_lookup_client_bypass import (
    LMCacheBypassLookupClient,
)


class _FakeTransport:
    def __init__(self, responses):
        self._responses = responses
        self.world_size = len(responses)
        self.sent = []

    def send_and_recv_all(self, msg_buf):
        self.sent.append(msg_buf)
        return self._responses


class _FakeTokenDatabase:
    def process_tokens(self, token_ids, make_key=True):
        yield (0, 256, 1111)
        yield (256, 512, 2222)


_ASKED = {HS_LAYER_IDXS_CONFIG: [0]}


def _client(responses):
    client = object.__new__(LMCacheLookupClient)
    client.transport = _FakeTransport(responses)
    client.enable_blending = False
    client.token_database = _FakeTokenDatabase()
    client.reqs_status = {}
    client.hs_status = {}
    return client


def test_four_byte_reply_leaves_coverage_unknown():
    """An older server answers KV only; the caller must fall back, not read zero."""
    client = _client([(512).to_bytes(4, "big")])

    assert client.lookup([1, 2, 3], "req", _ASKED) == 512
    assert client.lookup_hidden_state_coverage("req") is None


def test_eight_byte_reply_carries_coverage():
    client = _client([(512).to_bytes(4, "big") + (256).to_bytes(4, "big")])

    assert client.lookup([1, 2, 3], "req", _ASKED) == 512
    assert client.lookup_hidden_state_coverage("req") == 256


def test_coverage_takes_the_minimum_across_ranks():
    client = _client(
        [
            (512).to_bytes(4, "big") + (512).to_bytes(4, "big"),
            (512).to_bytes(4, "big") + (256).to_bytes(4, "big"),
        ]
    )

    assert client.lookup([1, 2, 3], "req", _ASKED) == 512
    assert client.lookup_hidden_state_coverage("req") == 256


def test_the_layer_request_rides_in_request_configs():
    """No frame-format change, so an older server just ignores the key."""
    client = _client([(512).to_bytes(4, "big")])

    client.lookup([1, 2, 3], "req", request_configs={HS_LAYER_IDXS_CONFIG: [0, -1]})

    sent = client.transport.sent[0]
    assert len(sent) == 4
    assert HS_LAYER_IDXS_CONFIG in sent[-1]


def test_clearing_status_drops_the_coverage_too():
    client = _client([(512).to_bytes(4, "big") + (256).to_bytes(4, "big")])
    client.lookup([1, 2, 3], "req", _ASKED)

    client.clear_lookup_status("req")

    assert client.lookup_hidden_state_coverage("req") is None
    assert client.lookup_cache("req") == -1


class _FakeEngine:
    """Records what the bypass client asked for and answers both queries."""

    def __init__(self, kv=512, coverage=256):
        self.token_database = _FakeTokenDatabase()
        self._kv = kv
        self._coverage = coverage
        self.hs_calls = []

    def lookup(self, **kwargs):
        return self._kv

    def lookup_hidden_states(self, **kwargs):
        self.hs_calls.append(kwargs)
        return self._coverage


def _bypass_client(engine):
    client = object.__new__(LMCacheBypassLookupClient)
    client.lmcache_engine = engine
    client.token_database = engine.token_database
    client.enable_blending = False
    client.hs_status = {}
    return client


def test_bypass_client_reports_coverage():
    """The bypass path holds the engine, so it must answer rather than inherit None."""
    engine = _FakeEngine(kv=512, coverage=256)
    client = _bypass_client(engine)

    assert client.lookup([1, 2, 3], "req", {HS_LAYER_IDXS_CONFIG: [0, 24]}) == 512
    assert client.lookup_hidden_state_coverage("req") == 256
    assert engine.hs_calls[0]["layer_idxs"] == [0, 24]

    client.clear_lookup_status("req")
    assert client.lookup_hidden_state_coverage("req") is None


def test_bypass_client_skips_the_query_when_not_asked():
    engine = _FakeEngine()
    client = _bypass_client(engine)

    assert client.lookup([1, 2, 3], "req") == 512
    assert client.lookup_hidden_state_coverage("req") is None
    assert engine.hs_calls == []

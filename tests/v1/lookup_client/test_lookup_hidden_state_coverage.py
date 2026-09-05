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

    assert client.lookup([1, 2, 3], "req") == 512
    assert client.lookup_hidden_state_coverage("req") is None


def test_eight_byte_reply_carries_coverage():
    client = _client([(512).to_bytes(4, "big") + (256).to_bytes(4, "big")])

    assert client.lookup([1, 2, 3], "req") == 512
    assert client.lookup_hidden_state_coverage("req") == 256


def test_coverage_takes_the_minimum_across_ranks():
    client = _client(
        [
            (512).to_bytes(4, "big") + (512).to_bytes(4, "big"),
            (512).to_bytes(4, "big") + (256).to_bytes(4, "big"),
        ]
    )

    assert client.lookup([1, 2, 3], "req") == 512
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
    client.lookup([1, 2, 3], "req")

    client.clear_lookup_status("req")

    assert client.lookup_hidden_state_coverage("req") is None
    assert client.lookup_cache("req") == -1

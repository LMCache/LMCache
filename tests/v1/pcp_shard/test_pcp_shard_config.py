# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side pieces of the PCP shard store mode: lookup server placement
(config getters) and the lookup client's min-combination."""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client import lmcache_lookup_client as llc
from lmcache.v1.metadata import LMCacheMetadata


def make(extra, world=8, use_mla=True, **flags):
    config = LMCacheEngineConfig.from_defaults(chunk_size=16, local_cpu=True)
    config.extra_config = dict(extra)
    for k, v in flags.items():
        setattr(config, k, v)
    metadata = LMCacheMetadata(
        model_name="m",
        world_size=world,
        local_world_size=world,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 1, 16, 1, 8),
        use_mla=use_mla,
        role="scheduler",
    )
    return config, metadata


def test_lookup_servers_on_every_rank():
    c, _ = make({"pcp_shard_store": True})
    assert c.get_lookup_server_worker_ids(True, 8) == list(range(8))
    c.lookup_server_worker_ids = [0]  # an explicit subset would be wrong
    assert c.get_lookup_server_worker_ids(True, 8) == list(range(8))
    c, _ = make({})
    assert c.get_lookup_server_worker_ids(True, 8) == [0]  # default unchanged
    assert c.get_lookup_server_worker_ids(False, 8) == list(range(8))
    c, _ = make({"pcp_shard_store": True})
    assert c.get_lookup_server_worker_ids(False, 8) == list(range(8))  # non-MLA


def test_lmcache_worker_ids():
    c, _ = make({"pcp_shard_store": True})
    assert c.get_lmcache_worker_ids(True, 8) == list(range(8))
    c.lmcache_worker_ids = [0]  # explicit choice respected
    assert c.get_lmcache_worker_ids(True, 8) == [0]
    c, _ = make({})
    assert c.get_lmcache_worker_ids(True, 8) == [0]


def test_unsupported_combination_is_loud():
    c, _ = make({"pcp_shard_store": True}, enable_p2p=True)
    with pytest.raises(ValueError, match="enable_p2p"):
        c.get_lookup_server_worker_ids(True, 8)
    c, _ = make({}, enable_p2p=True)
    assert c.get_lookup_server_worker_ids(True, 8) == [0]


def test_async_loading_supported():
    # Every rank runs an (async) lookup server and prefetches its own chunks.
    c, _ = make({"pcp_shard_store": True}, enable_async_loading=True)
    assert c.get_lookup_server_worker_ids(True, 8) == list(range(8))
    c, _ = make({}, enable_async_loading=True)
    assert c.get_lookup_server_worker_ids(True, 8) == [0]


class FakeTransport:
    def __init__(self, answers):
        self.answers = answers
        self.world_size = len(answers)

    def send_and_recv_all(self, msg):
        return [a.to_bytes(4, "big") for a in self.answers]

    def close(self):
        pass


@pytest.mark.parametrize("shard,warns", [(True, False), (False, True)])
def test_client_min_and_warning(shard, warns):
    extra = {"pcp_shard_store": True} if shard else {}
    config, metadata = make(extra)
    answers = [160, 80, 160, 160, 160, 160, 160, 160]
    client = llc.LMCacheLookupClient(config, metadata, FakeTransport(answers))
    with mock.patch.object(llc.logger, "warning") as warn:
        got = client.lookup(list(range(170)), "req")
    assert got == 80
    assert warn.called == warns

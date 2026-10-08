# SPDX-License-Identifier: Apache-2.0
"""RemoteBackend._per_rank_url: "{rank}" in remote_url becomes the worker rank."""

# Standard
from types import SimpleNamespace

# First Party
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


def test_rank_substituted():
    url = "resp://valkey-{rank}.ns.svc.cluster.local:6379"
    meta = SimpleNamespace(worker_id=5)
    assert (
        RemoteBackend._per_rank_url(url, meta)
        == "resp://valkey-5.ns.svc.cluster.local:6379"
    )


def test_missing_rank_defaults_to_zero():
    meta = SimpleNamespace(worker_id=None)
    assert (
        RemoteBackend._per_rank_url("resp://v-{rank}:6379", meta) == "resp://v-0:6379"
    )


def test_url_without_placeholder_unchanged():
    meta = SimpleNamespace(worker_id=3)
    assert (
        RemoteBackend._per_rank_url("resp://valkey:6379", meta) == "resp://valkey:6379"
    )

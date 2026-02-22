# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch
from vllm.v1.request import RequestStatus

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeEngine:
    def __init__(self):
        self.unpinned: list[str] = []
        self.store_steps: dict[str, int] = {}
        self.store_calls: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                self.store_steps[req_id] += 1
                yield None

        return _storer()


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine
        self.lookup_client = None


class _FakeLookupClient:
    def __init__(self):
        self.cancelled: list[str] = []

    def cancel_lookup(self, lookup_id: str):
        self.cancelled.append(lookup_id)


def _make_req(req_id: str, can_save: bool = True):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}
    connector.current_layer = 0
    connector.async_loading = False
    return connector, metadata, engine


def test_layerwise_storer_is_request_scoped_across_interleaved_finalize() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1"), _make_req("req-2")]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1", "req-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 1

    metadata.requests = [_make_req("req-1")]
    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert engine.store_steps["req-2"] == 2
    assert engine.unpinned == ["req-1"]
    assert len(connector.layerwise_storers) == 2

    metadata.requests = [_make_req("req-2")]
    connector.wait_for_save()
    assert engine.store_steps["req-2"] == 3
    assert engine.unpinned == ["req-1", "req-2"]
    assert len(connector.layerwise_storers) == 2


def test_wait_for_save_repeated_call_does_not_readvance_finalized_storer() -> None:
    connector, metadata, engine = _make_connector([_make_req("req-1")])
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_steps["req-1"] == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert len(connector.layerwise_storers) == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 3


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector.layerwise_storers == []


def test_request_finished_aborted_cleans_layerwise_storer() -> None:
    connector, _, _ = _make_connector([_make_req("req-1")])
    connector.async_loading = True
    connector._manager.lookup_client = _FakeLookupClient()
    connector._layerwise_save_storers = {"req-1": iter([None])}

    req = SimpleNamespace(status=RequestStatus.FINISHED_ABORTED, request_id="req-1")
    connector.request_finished(req, [])

    assert connector._layerwise_save_storers == {}
    assert connector.lookup_client is not None
    assert connector.lookup_client.cancelled == ["req-1"]


def test_request_finished_normal_cleans_layerwise_storer() -> None:
    connector, _, _ = _make_connector([_make_req("req-2")])
    connector._layerwise_save_storers = {"req-2": iter([None])}

    req = SimpleNamespace(status=RequestStatus.FINISHED_STOPPED, request_id="req-2")
    connector.request_finished(req, [])

    assert connector._layerwise_save_storers == {}


def test_request_finished_without_use_layerwise_attribute() -> None:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    req = SimpleNamespace(status=RequestStatus.FINISHED_STOPPED, request_id="req-x")

    # Ensure this path is safe even if use_layerwise is not initialized yet.
    connector.request_finished(req, [])

# SPDX-License-Identifier: Apache-2.0
"""Check per-request synchronous pin cleanup through the public vLLM load API."""

# Standard
from typing import Literal
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
)
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.cache_engine import LMCacheEngine

# Local
from .test_retrieve_pin_ownership import (  # noqa: F401 -- register shared fixtures
    _CacheCase,
    cache_case,
    pin_monitor,
)

pytestmark = pytest.mark.no_shared_allocator


class _CPUConnector(LMCacheConnectorV1Impl):
    """Initialize only load-path state without starting services or a GPU worker."""

    def __init__(
        self,
        requests: list[ReqMeta],
        engine: LMCacheEngine | MagicMock,
        loading: Literal["sync", "async", "layerwise"],
    ) -> None:
        """Inject recording dependencies while inheriting the real public load API.

        Args:
            requests: Scheduled requests in their load order.
            engine: Real CPU engine or recording mock returning CPU token masks.
            loading: Sync/non-layerwise, async/non-layerwise, or sync/layerwise mode.
        """
        metadata = LMCacheConnectorMetadata(requests=requests)
        self._parent = MagicMock(**{"_get_connector_metadata.return_value": metadata})
        self._manager = MagicMock(lmcache_engine=engine)
        self._stats_monitor = MagicMock(spec=LMCStatsMonitor)
        self._lmcache_chunk_size = 4
        self.device = "cpu"
        self.kv_caches = {"layer0": torch.zeros(1)}
        self.async_loading = loading == "async"
        self.use_layerwise = loading == "layerwise"
        self.enable_blending = False


@pytest.mark.parametrize("loading", ["sync", "async", "layerwise"])
def test_start_load_releases_sync_lookup_before_next_request(
    loading: Literal["sync", "async", "layerwise"],
) -> None:
    """A's sync pins must be released before B can need CPU staging allocation.

    Holding A's CPU hit until wait_for_save can prevent eviction while B's disk
    load allocates in the same start_load_kv call. Assert ordering, not merely
    eventual cleanup. Async and layerwise ownership must remain unchanged.

    Args:
        loading: Loading mode selecting the existing adapter dispatch branch.
    """
    requests = [
        ReqMeta(
            req_id=req_id,
            token_ids=list(range(4)),
            slot_mapping=torch.arange(4, dtype=torch.long),
            load_spec=LoadSpec(
                vllm_cached_tokens=0, lmcache_cached_tokens=4, can_load=True
            ),
        )
        for req_id in ("A", "B")
    ]
    engine = MagicMock(spec=LMCacheEngine)
    engine.retrieve.return_value = torch.ones(4, dtype=torch.bool)
    engine.retrieve_layer.side_effect = [iter([None, None]), iter([None, None])]
    connector = _CPUConnector(requests, engine, loading)
    context = MagicMock(attn_metadata=object())

    connector.start_load_kv(context)

    if loading == "layerwise":
        assert engine.retrieve_layer.call_count == 2
        engine.retrieve.assert_not_called()
        engine.lookup_unpin.assert_not_called()
    else:
        # Extract only method/req_id so Tensor argument equality does not obscure
        # the cross-request ordering that permits staging-space reclamation.
        events = [
            (name, args[0] if name == "lookup_unpin" else kwargs["req_id"])
            for name, args, kwargs in engine.method_calls
        ]
        if loading == "sync":
            assert events == [
                ("retrieve", "A"),
                ("lookup_unpin", "A"),
                ("retrieve", "B"),
                ("lookup_unpin", "B"),
            ]
        else:
            assert events == [("retrieve", "A"), ("retrieve", "B")]
        engine.retrieve_layer.assert_not_called()


@pytest.mark.parametrize("other_lookup_pins", [0, 1])
def test_next_load_observes_real_cache_eviction_eligibility(
    cache_case: _CacheCase,  # noqa: F811 -- pytest injects the imported fixture
    monkeypatch: pytest.MonkeyPatch,
    other_lookup_pins: int,
) -> None:
    """Before B loads, A is evictable unless another request still owns its pin.

    This checks the real MemoryObj.can_evict gate used by cache eviction without
    starting a disk worker or an allocator wait loop. The separate engine test
    verifies that retrieve itself cannot consume either request's lookup pin.

    Args:
        cache_case: Real engine, backend, token database and tiny CPU objects.
        monkeypatch: Observes B's entry before delegating to real engine retrieval.
        other_lookup_pins: Whether another in-flight request C shares A's chunk.
    """
    engine = cache_case.engine
    tokens = cache_case.tokens[:4]
    first_obj = cache_case.memory_objs[0]
    assert engine.lookup(tokens, lookup_id="A", pin=True) == 4
    if other_lookup_pins:
        assert engine.lookup(tokens, lookup_id="C", pin=True) == 4
    assert first_obj.metadata.pin_count == 1 + other_lookup_pins
    assert first_obj.get_ref_count() == 1
    assert not first_obj.can_evict
    requests = [
        ReqMeta(
            req_id="A",
            token_ids=tokens,
            slot_mapping=torch.arange(4, dtype=torch.long),
            load_spec=LoadSpec(0, 4, True),
        ),
        ReqMeta(
            req_id="B",
            token_ids=cache_case.tokens[:8],
            slot_mapping=torch.arange(8, dtype=torch.long),
            load_spec=LoadSpec(4, 8, True),
        ),
    ]
    connector = _CPUConnector(requests, engine, "sync")
    retrieve = engine.retrieve

    def retrieve_with_eviction_check(
        tokens: list[int], mask: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Observe ownership at B's entry, then execute the real retrieval.

        Args:
            tokens: Requested prefix tokens from the adapter.
            mask: Token mask supplied by the adapter.
            **kwargs: Connector options, including the request's req_id.

        Returns:
            The real engine's retrieved-token mask.
        """
        if kwargs["req_id"] == "B":
            assert first_obj.metadata.pin_count == other_lookup_pins
            assert first_obj.get_ref_count() == 1
            assert first_obj.can_evict == (other_lookup_pins == 0)
            assert "A" not in engine.lookup_pins
        return retrieve(tokens, mask, **kwargs)

    retrieve_spy = MagicMock(side_effect=retrieve_with_eviction_check)
    monkeypatch.setattr(engine, "retrieve", retrieve_spy)

    connector.start_load_kv(MagicMock(attn_metadata=object()))

    assert retrieve_spy.call_count == 2
    assert first_obj.metadata.pin_count == other_lookup_pins
    assert first_obj.get_ref_count() == 1
    if other_lookup_pins:
        assert engine.lookup_pins == {"C": {"LocalCPUBackend": cache_case.keys[:1]}}
        engine.lookup_unpin("C")
    assert not engine.lookup_pins
    assert first_obj.metadata.pin_count == 0
    assert first_obj.can_evict

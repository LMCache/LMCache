# SPDX-License-Identifier: Apache-2.0
"""Generator-contract tests for ``LMCacheEngine.store_layer``.

``store_layer`` returns a generator that its callers advance a fixed number of
times: once per model layer from ``LMCacheConnectorV1Impl.save_kv_layer`` and
once more from ``wait_for_save`` -- ``num_layers + 1`` advances in total, which
``tests/v1/test_vllm_layerwise_wait_for_save.py`` pins down from the caller's
side.

The callers cannot know in advance whether the store will happen: whether the
engine is healthy is decided by a background health monitor, and freeze mode by
a control message. Every exit path therefore has to yield the same number of
values, or the caller's ``next()`` raises ``StopIteration``. In vLLM that is not
a skipped store -- it propagates out of the attention layer and the engine dies
with ``EngineDeadError``, turning a degraded cache into an outage.

A skip path that runs past ``on_store_request`` also owes the stats monitor the
matching ``on_store_finished``, or the request it opened stays open forever.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine

NUM_LAYERS = 4
EXPECTED_ADVANCES = NUM_LAYERS + 1


class _StatsMonitorSpy:
    """Records the store-request lifecycle that ``store_layer`` drives."""

    def __init__(self) -> None:
        self.opened: list[SimpleNamespace] = []
        self.finished: list[tuple[SimpleNamespace, int]] = []

    def on_store_request(self, num_tokens: int) -> SimpleNamespace:
        request = SimpleNamespace(request_id=len(self.opened), num_tokens=num_tokens)
        self.opened.append(request)
        return request

    def on_store_finished(self, store_stats, num_stored_tokens: int = -1) -> None:
        self.finished.append((store_stats, num_stored_tokens))


def _engine(*, healthy: bool, frozen: bool) -> LMCacheEngine:
    """An engine stubbed down to what ``store_layer``'s skip paths touch."""
    engine = object.__new__(LMCacheEngine)
    engine.num_layers = NUM_LAYERS
    engine.is_healthy = lambda: healthy
    engine.is_frozen = lambda: frozen
    engine.storage_manager = MagicMock()
    engine.gpu_connector = MagicMock()
    engine.stats_monitor = _StatsMonitorSpy()
    engine._get_req_id = lambda kwargs: kwargs.get("req_id")
    engine._log_kvcache_for_check = lambda **kwargs: None
    return engine


def _storer(engine: LMCacheEngine):
    """``store_layer`` driven with the arguments the vLLM adapter passes."""
    return engine.store_layer(
        tokens=torch.tensor([1, 2, 3]),
        mask=None,
        kvcaches=[],
        slot_mapping=torch.tensor([0, 1, 2]),
        offset=0,
        sync=True,
        req_id="req-1",
    )


@pytest.mark.parametrize(
    "healthy, frozen, why",
    [
        (False, False, "unhealthy"),
        (True, True, "frozen"),
    ],
)
def test_store_layer_skip_paths_yield_the_full_contract(healthy, frozen, why) -> None:
    """A skipped store still advances as many times as the caller will ask."""
    engine = _engine(healthy=healthy, frozen=frozen)
    storer = _storer(engine)

    # once per layer, as save_kv_layer does
    for layer in range(NUM_LAYERS):
        try:
            next(storer)
        except StopIteration:  # pragma: no cover - the bug this test pins
            pytest.fail(
                f"{why}: store_layer stopped after {layer} of {NUM_LAYERS} "
                "per-layer advances; vLLM raises EngineDeadError here"
            )

    # the finalizing advance, as wait_for_save does
    try:
        next(storer)
    except StopIteration:  # pragma: no cover - the bug this test pins
        pytest.fail(
            f"{why}: store_layer stopped before the finalizing advance "
            f"(expected {EXPECTED_ADVANCES} in total)"
        )

    # and it is finished after exactly that many
    with pytest.raises(StopIteration):
        next(storer)


def test_store_layer_skipped_helper_matches_the_caller_count() -> None:
    """The helper the skip paths delegate to yields num_layers + 1 values."""
    engine = _engine(healthy=True, frozen=False)
    assert sum(1 for _ in engine._store_layer_skipped()) == EXPECTED_ADVANCES
    assert engine.stats_monitor.finished == []


def test_freeze_path_closes_the_stats_monitor_request() -> None:
    """Freeze mode opens a store request, so it has to close it as well."""
    engine = _engine(healthy=True, frozen=True)
    monitor = engine.stats_monitor
    storer = _storer(engine)

    for _ in range(NUM_LAYERS):
        next(storer)

    assert len(monitor.opened) == 1
    # still open across the per-layer advances, as on the hit and miss paths
    assert monitor.finished == []

    next(storer)  # the finalizing advance from wait_for_save

    assert monitor.finished == [(monitor.opened[0], 0)]


def test_unhealthy_path_opens_no_stats_monitor_request() -> None:
    """The unhealthy path returns before the request is opened, so none closes."""
    engine = _engine(healthy=False, frozen=False)
    monitor = engine.stats_monitor

    storer = _storer(engine)
    for _ in range(EXPECTED_ADVANCES):
        next(storer)

    assert monitor.opened == []
    assert monitor.finished == []

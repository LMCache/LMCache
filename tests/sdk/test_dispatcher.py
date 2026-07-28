# SPDX-License-Identifier: Apache-2.0
"""Unit tests for experimental-feature gating and dispatch.
Both connector-side dispatcher (feature construction, hook fan-out,
re-registration failure handling) and the server-side --enable gating.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.experimental.dispatcher import (
    Dispatcher,
    FeatureContext,
    QTensorFeature,
    dispatch,
    init_dispatcher,
)
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY

MODEL = "org/model"
HOOKS = tuple(
    name
    for name, attr in vars(Dispatcher).items()
    if not name.startswith("_") and callable(attr)
)


def _feature_context(advertised: set[str] | None = None) -> FeatureContext:
    """Feature needed by the dispatcher's init, united for all types of
    tensors captured."""
    adapter = MagicMock(name="worker_adapter")
    adapter.model_name = MODEL
    adapter.experimental = {TRANSFER_QUERY} if advertised is None else advertised
    return FeatureContext(
        worker_adapter=adapter, send_lmcache_request=MagicMock(name="send")
    )


def test_query_cache_uses_the_suffixed_model_name() -> None:
    """Check that QTensorFeature uses suffixed model name correctly."""
    feature = QTensorFeature(_feature_context())

    assert feature._q_ring_adapter.q_model_name == f"{MODEL}##query"


def test_init_dispatcher_builds_the_requested_feature() -> None:
    """Right now the intermediate tensor transfer only supports query tensors."""
    dispatcher = init_dispatcher(_feature_context(), {TRANSFER_QUERY})

    assert len(dispatcher._features) == 1
    assert isinstance(dispatcher._features[0], QTensorFeature)


def test_init_dispatcher_rejects_if_server_not_enabled() -> None:
    """Version skew must fail loudly at setup, not silently drop capture."""
    with pytest.raises(
        ValueError, match=f"enables {TRANSFER_QUERY} but server does not"
    ):
        init_dispatcher(_feature_context(advertised=set()), {TRANSFER_QUERY})


def test_init_dispatcher_ignores_unrequested_features() -> None:
    """Check that dispatcher ignores calls to functions for features that are
    not requested."""
    dispatcher = init_dispatcher(_feature_context(), set())

    assert dispatcher._features == []
    dispatcher.reclaim()
    dispatcher.shutdown()
    assert dispatcher.reregister() is True


def test_dispatch_without_a_dispatcher_is_a_noop() -> None:
    """Check that the dispatch not returns error even when the feature is
    not built."""
    dispatch(None, "reclaim")
    dispatch(None, "wait_for_save", event=None)


def test_every_dispatcher_hook_has_a_matching_feature_hook() -> None:
    """Check that every hook on the dispatcher has a matching hook on the
    feature."""
    assert HOOKS, "no hooks discovered on Dispatcher"
    for hook in HOOKS:
        assert callable(getattr(QTensorFeature, hook, None)), hook


@pytest.mark.parametrize("feature_ok", [True, False])
def test_reregister_reports_the_feature_result(feature_ok) -> None:
    """Check that reregister returns the result, True if the feature
    re-registration succeeded and False if it failed."""
    feature = MagicMock()
    feature.reregister.return_value = feature_ok

    assert Dispatcher([feature]).reregister() is feature_ok


@pytest.mark.parametrize(
    "error, expected",
    [(ConnectionError("down"), False), (RuntimeError("boom"), False), (None, True)],
)
def test_feature_reregister_reports_failure_without_raising(error, expected) -> None:
    """Re-registration failures are logged and reported, never propagated into
    the heartbeat thread."""
    feature = QTensorFeature(_feature_context())
    adapter = MagicMock(name="q_ring_adapter")
    adapter.reregister_q_ring.side_effect = error
    feature._q_ring_adapter = adapter

    assert feature.reregister() is expected


class _FakeLMCacheDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeEngineDriven:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


class _FakeQStore:
    def __init__(self, ctx) -> None:
        self.ctx = ctx


@pytest.fixture
def stub_server_modules(monkeypatch):
    """Stub the server's module constructors. Returns the ManagementModule mock.
    The transfer modules stay real classes: _build_modules isinstance-checks
    them to pick liveness targets and the lmcache-driven module."""
    monkeypatch.setattr(server_mod, "LookupModule", lambda ctx: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(server_mod, "QStoreModule", _FakeQStore)
    management = MagicMock(name="ManagementModule")
    monkeypatch.setattr(server_mod, "ManagementModule", management)
    return management


def _build(stub_server_modules, **config) -> list:
    return server_mod._build_modules(
        MagicMock(name="ctx"), MPServerConfig(**config), MagicMock(url="")
    )


def test_server_rejects_an_unknown_experimental_module(stub_server_modules) -> None:
    """Check that the server rejects an unknown experimental module.
    Only 'transfer_query' is supported right now."""
    with pytest.raises(ValueError, match="Unknown --enable"):
        _build(stub_server_modules, enable=["query"])


def test_server_requires_lmcache_driven_transfer(stub_server_modules) -> None:
    """Check that the server rejects a request to enable transfer_query when the
    transfer mode is not lmcache-driven."""
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(
            stub_server_modules,
            enable=[TRANSFER_QUERY],
            supported_transfer_mode="engine_driven",
        )


def test_server_builds_q_store_module(stub_server_modules) -> None:
    """Check that the server builds the Q store module and puts it to the
    ManagementModule."""
    modules = _build(
        stub_server_modules,
        enable=[TRANSFER_QUERY],
        supported_transfer_mode="lmcache_driven",
    )

    assert any(isinstance(m, _FakeQStore) for m in modules)
    kwargs = stub_server_modules.call_args.kwargs
    assert kwargs["experimental_transfer"] == [TRANSFER_QUERY]
    assert any(isinstance(t, _FakeQStore) for t in kwargs["liveness_targets"])


def test_server_builds_nothing_when_no_feature_is_enabled(
    stub_server_modules,
) -> None:
    """Check that the server builds nothing when no feature is enabled."""
    modules = _build(stub_server_modules)

    assert not any(isinstance(m, _FakeQStore) for m in modules)
    assert stub_server_modules.call_args.kwargs["experimental_transfer"] == []

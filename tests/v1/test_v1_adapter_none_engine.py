# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# Third Party
from vllm.v1.request import RequestStatus

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


class _FakeParent:
    def __init__(self, metadata=None):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeManager:
    def __init__(self, engine=None):
        self.lmcache_engine = engine
        self.lmcache_engine_metadata = None
        self.lookup_client = None


def test_request_finished_with_none_engine() -> None:
    """Aborted request must not crash when lmcache_engine is None.

    On the scheduler side, when ``enable_scheduler_bypass_lookup`` is
    disabled, the service factory returns ``None`` for the engine.
    The original code had ``assert self.lmcache_engine is not None``
    which killed the EngineCore process on any request abort.
    """
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent()  # type: ignore[assignment]
    connector._manager = _FakeManager(engine=None)  # type: ignore[assignment]
    connector.config = SimpleNamespace(
        get_extra_config_value=lambda key, default: default
    )
    connector.async_loading = False

    request = SimpleNamespace(
        request_id="req-aborted",
        status=RequestStatus.FINISHED_ABORTED,
    )

    res, params = connector.request_finished(request, [1, 2, 3])
    assert res is False
    assert params is None


def test_request_finished_with_none_engine_and_async_loading() -> None:
    """Aborted request with async_loading must not crash when
    both lmcache_engine and lookup_client are None.

    This exercises the second guard (``lookup_client is not None``)
    in the abort cleanup path.
    """
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent()  # type: ignore[assignment]
    connector._manager = _FakeManager(engine=None)  # type: ignore[assignment]
    connector.config = SimpleNamespace(
        get_extra_config_value=lambda key, default: default
    )
    connector.async_loading = True

    request = SimpleNamespace(
        request_id="req-aborted-async",
        status=RequestStatus.FINISHED_ABORTED,
    )

    res, params = connector.request_finished(request, [1, 2, 3])
    assert res is False
    assert params is None

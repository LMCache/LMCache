# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module
from types import ModuleType
import sys

# Third Party
import pytest


def test_preset_entry_point_without_serving_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the preset imports both classes without requiring a GPU runtime."""
    import_module("lmcache.integration.tensorrt_llm")
    canonical = ModuleType(
        "lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter"
    )
    scheduler = type("LMCacheKvConnectorScheduler", (), {})
    worker = type("LMCacheKvConnectorWorker", (), {})
    monkeypatch.setattr(
        canonical, "LMCacheKvConnectorScheduler", scheduler, raising=False
    )
    monkeypatch.setattr(canonical, "LMCacheKvConnectorWorker", worker, raising=False)
    monkeypatch.setitem(sys.modules, canonical.__name__, canonical)
    legacy_path = "lmcache.integration.tensorrt_llm.tensorrt_adapter"
    monkeypatch.delitem(sys.modules, legacy_path, raising=False)

    legacy = import_module(legacy_path)
    # Restore the import cache before removing the canonical stand-in.
    sys.modules.pop(legacy_path)

    assert legacy.LMCacheKvConnectorScheduler is scheduler
    assert legacy.LMCacheKvConnectorWorker is worker


@pytest.mark.parametrize(
    "symbol",
    ["LMCacheKvConnectorScheduler", "LMCacheKvConnectorWorker"],
)
def test_legacy_imports_reexport_canonical_symbols(symbol: str) -> None:
    """Verify TensorRT-LLM's preset resolves to the canonical connector classes."""
    pytest.importorskip("tensorrt_llm")
    legacy = import_module("lmcache.integration.tensorrt_llm.tensorrt_adapter")
    canonical = import_module(
        "lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter"
    )

    assert getattr(legacy, symbol) is getattr(canonical, symbol)

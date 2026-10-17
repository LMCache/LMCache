# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module

# Third Party
import pytest

pytest.importorskip("tensorrt_llm")


def test_legacy_tensorrt_llm_adapter_imports_canonical_connector() -> None:
    legacy = import_module("lmcache.integration.tensorrt_llm.tensorrt_adapter")
    canonical = import_module(
        "lmcache.inprocess.integration.tensorrt_llm.tensorrt_adapter"
    )

    assert legacy.LMCacheKvConnectorScheduler is canonical.LMCacheKvConnectorScheduler
    assert legacy.LMCacheKvConnectorWorker is canonical.LMCacheKvConnectorWorker


def test_legacy_tensorrt_llm_utils_import_canonical_helper() -> None:
    legacy = import_module("lmcache.integration.tensorrt_llm.utils")
    canonical = import_module("lmcache.inprocess.integration.tensorrt_llm.utils")

    assert legacy.lmcache_get_config is canonical.lmcache_get_config

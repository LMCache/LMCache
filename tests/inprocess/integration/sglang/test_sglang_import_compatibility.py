# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module


def test_legacy_sglang_adapter_imports_canonical_connector() -> None:
    legacy = import_module("lmcache.integration.sglang.sglang_adapter")
    canonical = import_module("lmcache.inprocess.integration.sglang.sglang_adapter")

    assert legacy.LMCacheConnector is canonical.LMCacheConnector
    assert legacy.LMCacheLayerwiseConnector is canonical.LMCacheLayerwiseConnector


def test_legacy_sglang_utils_import_canonical_helper() -> None:
    legacy = import_module("lmcache.integration.sglang.utils")
    canonical = import_module("lmcache.inprocess.integration.sglang.utils")

    assert legacy.lmcache_get_config is canonical.lmcache_get_config

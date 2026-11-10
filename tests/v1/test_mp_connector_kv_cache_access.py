# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheMPConnector.kv_cache_access."""

# Standard
from types import ModuleType
from typing import Any

# Third Party
import pytest

_MODULE_PATH = "lmcache.integration.vllm.lmcache_mp_connector"


@pytest.fixture
def connector_module() -> ModuleType:
    """Import lazily: the connector module initializes process-wide MP
    transfer state at import time, which contaminates later cache-server
    tests when imported at collection. Skips where vLLM is not installed
    (e.g. the LMCache-only unit environment)."""
    return pytest.importorskip(_MODULE_PATH, reason="requires vLLM")


@pytest.fixture
def kv_cache_access_cls(connector_module: ModuleType) -> Any:
    """``KVCacheAccess``, skipping on vLLM without the kv_cache_access hook."""
    if connector_module.KVCacheAccess is None:
        pytest.skip("requires vLLM with KVCacheAccess (vllm-project/vllm#66605)")
    return connector_module.KVCacheAccess


def make_config(extra_config: dict[str, Any], module_path: str | None = None) -> Any:
    # Third Party
    from vllm.config import KVTransferConfig

    return KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
        kv_connector_module_path=module_path,
    )


@pytest.mark.parametrize(
    "extra_config, expected",
    [
        ({}, "LEGACY_IPC"),
        ({"lmcache.mp.isolated_ipc": True}, "LEGACY_IPC"),
        ({"lmcache.mp.use_vmm_api": False}, "LEGACY_IPC"),
        ({"lmcache.mp.use_vmm_api": True}, "EXPORTS_FD"),
        (
            {"lmcache.mp.use_vmm_api": True, "lmcache.mp.isolated_ipc": True},
            "EXPORTS_FABRIC",
        ),
        (
            {"lmcache.mp.use_vmm_api": True, "lmcache.mp.isolated_ipc": False},
            "EXPORTS_FD",
        ),
        # CLI passthrough delivers strings; "false" must not read as truthy.
        ({"lmcache.mp.use_vmm_api": "false"}, "LEGACY_IPC"),
        ({"lmcache.mp.use_vmm_api": "true"}, "EXPORTS_FD"),
        (
            {"lmcache.mp.use_vmm_api": "1", "lmcache.mp.isolated_ipc": "yes"},
            "EXPORTS_FABRIC",
        ),
    ],
)
def test_access_follows_ipc_config(
    connector_module: ModuleType,
    kv_cache_access_cls: Any,
    extra_config: dict[str, Any],
    expected: str,
) -> None:
    access = connector_module.LMCacheMPConnector.kv_cache_access(
        make_config(extra_config)
    )
    assert access is kv_cache_access_cls[expected]


def test_returns_none_without_kv_cache_access(
    connector_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connector_module, "KVCacheAccess", None)
    config = make_config({"lmcache.mp.use_vmm_api": True})
    assert connector_module.LMCacheMPConnector.kv_cache_access(config) is None


@pytest.mark.parametrize("wrap_in_multi_connector", [False, True])
def test_vllm_resolves_access_from_module_path(
    kv_cache_access_cls: Any, wrap_in_multi_connector: bool
) -> None:
    """vLLM loads the connector from ``kv_connector_module_path`` and asks
    it, directly or as a MultiConnector child."""
    # Third Party
    from vllm.config import KVTransferConfig

    extra_config = {"lmcache.mp.use_vmm_api": True}
    if wrap_in_multi_connector:
        config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "LMCacheMPConnector",
                        "kv_role": "kv_both",
                        "kv_connector_module_path": _MODULE_PATH,
                        "kv_connector_extra_config": extra_config,
                    }
                ]
            },
        )
    else:
        config = make_config(extra_config, module_path=_MODULE_PATH)
    assert config.kv_cache_access() == kv_cache_access_cls.EXPORTS_FD

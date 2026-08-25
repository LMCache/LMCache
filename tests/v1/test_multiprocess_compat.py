# SPDX-License-Identifier: Apache-2.0
"""Tests for the narrow vLLM MP namespace compatibility bridge."""

# Standard
from importlib import import_module
from importlib.util import find_spec
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.multiprocess import custom_types as canonical_custom_types

SUPPORTED_EXPORTS = (
    "DeviceIPCWrapper",
    "IPCCacheServerKey",
    "KVCache",
    "RegisterEngineDrivenContextPayload",
    "CustomizedSerdeConfig",
    "BlockAllocationRecord",
    "CBMatchResult",
    "CBUnifiedLookupResult",
    "get_customized_encoder",
    "get_customized_decoder",
)


@pytest.mark.parametrize("symbol_name", SUPPORTED_EXPORTS)
def test_legacy_custom_type_export_preserves_identity(symbol_name: str) -> None:
    """Every supported legacy export must be the canonical object."""
    legacy_custom_types = import_module("lmcache.v1.multiprocess.custom_types")

    assert getattr(legacy_custom_types, symbol_name) is getattr(
        canonical_custom_types, symbol_name
    )


def test_legacy_custom_types_all_is_bounded() -> None:
    """The compatibility module must expose only the supported public surface."""
    legacy_custom_types = import_module("lmcache.v1.multiprocess.custom_types")

    assert legacy_custom_types.__all__ == list(SUPPORTED_EXPORTS)


def test_legacy_cb_unified_lookup_result_preserves_module() -> None:
    """The CacheBlend lookup result must retain its canonical module."""
    legacy_custom_types = import_module("lmcache.v1.multiprocess.custom_types")

    assert (
        legacy_custom_types.CBUnifiedLookupResult
        is canonical_custom_types.CBUnifiedLookupResult
    )
    assert (
        legacy_custom_types.CBUnifiedLookupResult.__module__
        == "lmcache.multiprocess.custom_types"
    )


@pytest.mark.parametrize(
    "module_name",
    ("lmcache.v1.multiprocess.mq", "lmcache.v1.multiprocess.protocol"),
)
def test_legacy_mp_submodule_remains_unsupported(module_name: str) -> None:
    """Legacy MP implementation modules must remain unavailable."""
    assert find_spec(module_name) is None


def test_vllm_default_mp_connector_uses_lmcache_adapter() -> None:
    """vLLM's default MP connector must load LMCache's current adapter."""
    pytest.importorskip("vllm")

    code = (
        "from lmcache.multiprocess.custom_types import BlockAllocationRecord\n"
        "from lmcache.integration.vllm.vllm_multi_process_adapter import "
        "LMCacheMPSchedulerAdapter, LMCacheMPWorkerAdapter\n"
        "from vllm.config import KVTransferConfig\n"
        "from vllm.distributed.kv_transfer.kv_connector.factory import "
        "KVConnectorFactory\n"
        "from vllm.distributed.kv_transfer.kv_connector.v1 import "
        "lmcache_mp_connector as connector\n"
        "config = KVTransferConfig(kv_connector='LMCacheMPConnector', "
        "kv_role='kv_both')\n"
        "resolved = KVConnectorFactory.get_connector_class(config)\n"
        "assert config.kv_connector_module_path is None\n"
        "assert resolved is connector.LMCacheMPConnector\n"
        "assert connector.RequestAllocationRecord is BlockAllocationRecord\n"
        "assert connector.LMCacheMPSchedulerAdapter is "
        "LMCacheMPSchedulerAdapter\n"
        "assert connector.LMCacheMPWorkerAdapter is LMCacheMPWorkerAdapter\n"
    )
    env = os.environ.copy()
    env.pop("LMCACHE_USE_UPSTREAM_MP", None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

# SPDX-License-Identifier: Apache-2.0
"""Tests for the narrow vLLM MP namespace compatibility bridge."""

# Standard
from importlib import import_module
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache.multiprocess.custom_types import BlockAllocationRecord


def test_legacy_block_allocation_record_preserves_identity() -> None:
    """The legacy import must resolve to the canonical flattened class."""
    legacy_custom_types = import_module("lmcache.v1.multiprocess.custom_types")

    assert legacy_custom_types.BlockAllocationRecord is BlockAllocationRecord
    assert (
        legacy_custom_types.BlockAllocationRecord.__module__
        == "lmcache.multiprocess.custom_types"
    )


def test_vllm_default_mp_connector_uses_lmcache_adapter() -> None:
    """vLLM's default MP connector must load LMCache's current adapter."""
    pytest.importorskip("vllm")

    code = (
        "from importlib.util import find_spec\n"
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
        "assert find_spec('lmcache.v1.multiprocess.mq') is None\n"
        "assert find_spec('lmcache.v1.multiprocess.protocol') is None\n"
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

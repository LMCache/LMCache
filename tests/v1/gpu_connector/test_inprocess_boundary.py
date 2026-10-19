# SPDX-License-Identifier: Apache-2.0
"""Import boundaries between shared transfer support and in-process connectors."""

# Standard
import subprocess
import sys
import textwrap

# Third Party
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector import CreateGPUConnector
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.inprocess.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata


def test_shared_mp_imports_do_not_load_inprocess_connectors() -> None:
    """Shared layout and MP transfer imports work without the old engine package."""
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        class RejectInprocess(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lmcache.inprocess" or fullname.startswith(
                    "lmcache.inprocess."
                ):
                    raise AssertionError(f"Shared import loaded {fullname}")
                return None

        sys.meta_path.insert(0, RejectInprocess())
        for name in (
            "lmcache.v1.gpu_connector",
            "lmcache.v1.gpu_connector.kv_format",
            "lmcache.v1.gpu_connector.utils",
            "lmcache.v1.gpu_connector.gpu_ops",
            "lmcache.v1.gpu_connector.gds_context",
            "lmcache.v1.platform.base.cache_context",
            "lmcache.v1.multiprocess.object_group_transfer",
        ):
            importlib.import_module(name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mock_factory_uses_inprocess_connector_interface() -> None:
    """The relocated factory constructs a usable connector without GPU hardware."""
    metadata = LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(2, 2, 32, 2, 8),
    )
    connector = CreateGPUConnector(
        LMCacheEngineConfig.from_defaults(), metadata, EngineType.MOCK
    )
    assert isinstance(connector, GPUConnectorInterface)
    assert isinstance(connector, MockGPUConnector)
    assert connector.get_shape(16) == torch.Size([2, 2, 16, 2, 8])

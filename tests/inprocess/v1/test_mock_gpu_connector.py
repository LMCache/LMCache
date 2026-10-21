# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.inprocess.v1.gpu_connector import CreateGPUConnector
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.inprocess.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata


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

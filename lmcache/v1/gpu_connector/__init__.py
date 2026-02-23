# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_vllm_torch_dev
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.gpu_connector.utils import need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata


def CreateGPUConnector(
    config: LMCacheEngineConfig, metadata: LMCacheMetadata, engine: EngineType
) -> GPUConnectorInterface:
    """
    Create a GPU Connector based on the configuration and metadata.

    Args:
        config: The LMCache engine configuration.
        metadata: The LMCache metadata.
        engine: The serving engine type (EngineType.VLLM, EngineType.SGLANG,
                or EngineType.MOCK).
    """
    use_gpu = need_gpu_interm_buffer(config)

    if engine == EngineType.SGLANG:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import (
            SGLangGPUConnector,
            SGLangLayerwiseGPUConnector,
        )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        local_worker_id = metadata.local_worker_id
        torch.cuda.device(local_worker_id)
        device = torch.device(f"cuda:{local_worker_id}")
        kv_dtype = metadata.kv_dtype
        if config.use_layerwise:
            return SGLangLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            return SGLangGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
    elif engine == EngineType.VLLM:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import (
            VLLMBufferLayerwiseGPUConnector,
            VLLMPagedMemGPUConnectorV2,
            VLLMPagedMemGPUConnectorV3,
            VLLMPagedMemLayerwiseGPUConnector,
        )

        local_worker_id = metadata.local_worker_id
        torch_dev, dev_name = get_vllm_torch_dev()
        torch_dev.set_device(local_worker_id)
        device = torch.device(f"{dev_name}:{local_worker_id}")

        if config.use_layerwise:
            if config.enable_blending:
                return VLLMBufferLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu, device
                )
            else:
                return VLLMPagedMemLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu, device
                )

        if dev_name == "cuda":
            if config.use_gpu_connector_v3:
                return VLLMPagedMemGPUConnectorV3.from_metadata(
                    metadata, use_gpu, device
                )
            else:
                return VLLMPagedMemGPUConnectorV2.from_metadata(
                    metadata, use_gpu, device
                )
        elif dev_name == "xpu":
            # First Party
            from lmcache.v1.gpu_connector.xpu_connectors import (
                VLLMPagedMemXPUConnectorV2,
            )

            return VLLMPagedMemXPUConnectorV2.from_metadata(metadata, use_gpu, device)
        else:
            raise RuntimeError("No supported connector found for the current platform.")

    elif engine == EngineType.MOCK:
        kv_shape = metadata.kv_shape
        return MockGPUConnector(kv_shape=kv_shape)
    else:
        raise RuntimeError(f"Unsupported engine: {engine}")

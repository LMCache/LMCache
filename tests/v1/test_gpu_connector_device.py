# SPDX-License-Identifier: Apache-2.0
"""
Device selection in :func:`lmcache.v1.gpu_connector.CreateGPUConnector` for
vLLM workers.

vLLM binds each worker process to its device before creating the KV
connector. With data parallelism, every data-parallel rank is its own engine
whose TP/PP world starts at rank 0, so ``metadata.local_worker_id`` is 0 on
every rank. The vLLM connector must use the device vLLM already selected
instead of switching the process to ``local_worker_id``.

The tests stub ``torch_dev`` and the connector factory method, so they run
without accelerator hardware.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.v1.gpu_connector as gpu_connector_module


class _BoundTorchDev:
    """Stand-in for ``torch.cuda`` in a process already bound to a device."""

    def __init__(self, current: int) -> None:
        self.current = current
        self.set_device_calls: list[Any] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def current_device(self) -> int:
        return self.current

    def set_device(self, idx: Any) -> None:
        self.set_device_calls.append(idx)
        self.current = int(idx)


def _make_metadata(local_worker_id: int) -> LMCacheMetadata:
    """Metadata as built for one vLLM data-parallel rank with TP=PP=1."""
    return LMCacheMetadata(
        model_name="gpu_connector_device_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=local_worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, 16, 8, 64),
    )


def test_vllm_connector_keeps_device_bound_by_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Data-parallel rank 1 stays on device 1 even though its
    ``local_worker_id`` is 0."""
    fake_dev = _BoundTorchDev(current=1)
    monkeypatch.setattr(gpu_connector_module, "torch_device_type", "cuda")
    monkeypatch.setattr(gpu_connector_module, "torch_dev", fake_dev)

    captured: dict[str, Any] = {}

    def fake_from_metadata(
        cls: type, metadata: LMCacheMetadata, use_gpu: bool, device: Any, **kw: Any
    ) -> object:
        captured["device"] = device
        return object()

    monkeypatch.setattr(
        VLLMPagedMemGPUConnectorV2,
        "from_metadata",
        classmethod(fake_from_metadata),
    )

    CreateGPUConnector(
        LMCacheEngineConfig.from_defaults(chunk_size=16),
        _make_metadata(local_worker_id=0),
        EngineType.VLLM,
    )

    assert captured["device"] == torch.device("cuda:1")
    assert fake_dev.current == 1
    assert fake_dev.set_device_calls == []

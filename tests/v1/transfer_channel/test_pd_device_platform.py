# SPDX-License-Identifier: Apache-2.0
"""Tests for platform-agnostic P/D transfer device handling."""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, PagedCpuGpuMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.pd_backend import PDBackend
from lmcache.v1.transfer_channel.nixl_channel import NixlAgentWrapper
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device


def _test_metadata(kv_shape: tuple[int, int, int, int, int]) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def test_get_correct_device_supports_accelerator_families() -> None:
    """Device resolution preserves CPU and appends worker ids for accelerators."""
    assert get_correct_device("cpu", worker_id=7) == "cpu"
    assert get_correct_device("cuda", worker_id=7) == "cuda:7"
    assert get_correct_device("cuda:0", worker_id=7) == "cuda:7"
    assert get_correct_device("xpu", worker_id=7) == "xpu:7"
    assert get_correct_device("hpu:0", worker_id=7) == "hpu:7"
    assert get_correct_device("testaccel", worker_id=7) == "testaccel:7"


@patch("lmcache.v1.storage_backend.pd_backend.PagedCpuGpuMemoryAllocator")
def test_pd_backend_passes_resolved_non_cpu_device_to_allocator(
    mock_allocator_cls: MagicMock,
) -> None:
    """PDBackend selects and forwards the resolved non-CPU transfer device."""
    metadata = _test_metadata(kv_shape=(4, 2, 256, 8, 128))
    # 4 * 2 * 256 * 8 * 128 * 2 bytes (bfloat16) = 4194304 bytes per chunk.
    expected_aligned_size = 12582912

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        pd_buffer_size=13000000,
    )
    allocator = MagicMock(spec=PagedCpuGpuMemoryAllocator)
    mock_allocator_cls.return_value = allocator
    backend_context = SimpleNamespace(corrected_device="xpu:0")
    mock_torch_dev = MagicMock()

    with patch("lmcache.v1.storage_backend.pd_backend.torch_dev", mock_torch_dev):
        returned_allocator = PDBackend.initialize_allocator(
            backend_context, config, metadata
        )

    assert returned_allocator is allocator
    mock_torch_dev.set_device.assert_called_once_with("xpu:0")
    allocator.init_cpu_memory_allocator.assert_not_called()
    allocator.init_gpu_memory_allocator.assert_called_once_with(
        expected_aligned_size,
        [torch.Size(metadata.kv_shape)],
        [metadata.kv_dtype],
        MemoryFormat.KV_2LTD,
        device="xpu:0",
    )


@pytest.mark.parametrize(
    ("device", "expected_mem_type"),
    [
        ("cpu", "cpu"),
        ("cuda:0", "VRAM"),
        ("xpu:0", "VRAM"),
        ("hpu:0", "VRAM"),
    ],
)
def test_nixl_agent_wrapper_maps_supported_devices_to_mem_type(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected_mem_type: str,
) -> None:
    """NixlAgentWrapper maps supported platform devices to NIXL memory types."""
    calls: dict[str, str] = {}

    class FakeNixlAgent:
        def __init__(self, name: str, config: object) -> None:
            self.name = name
            self.config = config

        def get_reg_descs(
            self, descs: list[tuple[int, int, int, str]], mem_type: str
        ) -> str:
            calls["reg_mem_type"] = mem_type
            return "reg_descs"

        def register_memory(self, reg_descs: str) -> None:
            calls["registered"] = reg_descs

        def get_xfer_descs(
            self, descs: list[tuple[int, int, int]], mem_type: str
        ) -> str:
            calls["xfer_mem_type"] = mem_type
            return "xfer_descs"

        def prep_xfer_dlist(self, name: str, xfer_descs: str, mem_type: str) -> str:
            calls["handler_mem_type"] = mem_type
            return "handler"

    def fake_nixl_agent_config(backends: list[str]) -> dict[str, list[str]]:
        return {"backends": backends}

    nixl_api = ModuleType("nixl._api")
    nixl_api.nixl_agent = FakeNixlAgent
    nixl_api.nixl_agent_config = fake_nixl_agent_config
    monkeypatch.setitem(sys.modules, "nixl", ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", nixl_api)

    wrapper = NixlAgentWrapper(
        buffer_ptr=0,
        buffer_size=16,
        page_size=8,
        tp_rank=0,
        backends=["UCX"],
        device=device,
    )

    assert wrapper.reg_descs == "reg_descs"
    assert wrapper.xfer_descs == "xfer_descs"
    assert wrapper.xfer_handler == "handler"
    assert calls == {
        "reg_mem_type": expected_mem_type,
        "registered": "reg_descs",
        "xfer_mem_type": expected_mem_type,
        "handler_mem_type": expected_mem_type,
    }


def test_nixl_agent_wrapper_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NixlAgentWrapper rejects devices without a supported NIXL memory type."""

    class FakeNixlAgent:
        def __init__(self, name: str, config: object) -> None:
            self.name = name
            self.config = config

    def fake_nixl_agent_config(backends: list[str]) -> dict[str, list[str]]:
        return {"backends": backends}

    nixl_api = ModuleType("nixl._api")
    nixl_api.nixl_agent = FakeNixlAgent
    nixl_api.nixl_agent_config = fake_nixl_agent_config
    monkeypatch.setitem(sys.modules, "nixl", ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", nixl_api)

    with pytest.raises(ValueError, match="Unsupported device type 'testaccel'"):
        NixlAgentWrapper(
            buffer_ptr=0,
            buffer_size=16,
            page_size=8,
            tp_rank=0,
            backends=["UCX"],
            device="testaccel:0",
        )

# SPDX-License-Identifier: Apache-2.0
"""Tests for CPU cache-context transfer buffer layouts."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.detectors import vllm as vllm_detector
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_plan import (
    build_object_group_layout_desc,
    export_kv_transfer_metadata,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.cpu.cache_context import CPUCacheContext


class _TensorWrapper(DeviceIPCWrapper):
    """Return an in-process CPU tensor through the IPC wrapper interface."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the wrapped tensor."""
        return self._tensor


@pytest.mark.parametrize(
    ("full_sw_kv", "expected_slots"),
    [(False, 64), (True, 256)],
)
def test_sliding_window_staging_matches_registered_layout(
    full_sw_kv: bool,
    expected_slots: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU staging buffers must match the registered transfer layout."""
    monkeypatch.setattr(vllm_detector, "torch_device_type", "cpu")
    tensor = torch.empty(2, 4, 2, 16, 8, dtype=torch.float32)
    context = CPUCacheContext(
        [_TensorWrapper(tensor)],
        lmcache_tokens_per_chunk=256,
        layout_hints={"kv_layout": "HND"},
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=16,
                sw_size_tokens=64,
            )
        ],
        full_sw_kv=full_sw_kv,
    )

    metadata = export_kv_transfer_metadata(
        context.kv_layer_groups_manager,
        context.lmcache_tokens_per_chunk,
    )
    layout = build_object_group_layout_desc(metadata, 256, object_group_id=0)
    kernel_buffer = context.get_temp_kernel_group_buffer(0, 0)
    object_buffer = context.get_temp_object_group_buffer(0, 0)

    assert kernel_buffer.shape[2] == expected_slots
    assert layout.shapes == [kernel_buffer.shape]
    assert layout.dtypes == [kernel_buffer.dtype]
    assert object_buffer.nbytes == kernel_buffer.nbytes

# SPDX-License-Identifier: Apache-2.0
"""NPU GDS staging registration tests."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.npu import NpuDeviceSpec


def test_npu_device_spec_stream_handle():
    spec = NpuDeviceSpec()
    stream = SimpleNamespace(npu_stream=12345)
    assert spec.get_stream_handle(stream) == 12345
    with pytest.raises(RuntimeError, match="npu_stream"):
        spec.get_stream_handle(SimpleNamespace(npu_stream=0))


def test_npu_cache_context_deregisters_on_close():
    # First Party
    from lmcache.v1.platform.devices.npu import cache_context as mod

    ctx = mod.NpuCacheContext.__new__(mod.NpuCacheContext)
    ctx._ipc_wrappers = (MagicMock(),)
    ctx._gds_buffer_registered = True
    ctx.stream_ = SimpleNamespace(synchronize=lambda: None)
    ctx._temp_buffer = MagicMock()
    ctx._temp_buffer.buffer = torch.empty(1, dtype=torch.uint8)
    ctx.kv_caches_ = []
    with (
        patch.object(mod.torch_dev, "stream", return_value=MagicMock()),
        patch.object(mod, "deregister_gds_gpu_buffer") as mock_deregister,
    ):
        ctx.close()
    mock_deregister.assert_called_once_with(ctx._temp_buffer.buffer)

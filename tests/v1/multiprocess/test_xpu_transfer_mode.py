# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.modules.gpu_transfer import GPUTransferModule
from lmcache.v1.multiprocess.modules.non_gpu_transfer import NonGPUTransferModule
from lmcache.v1.multiprocess.modules.xpu_transfer import XpuTransferModule
from lmcache.v1.multiprocess.server import _build_modules
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    XpuHandleTransferContext,
    create_transfer_context,
)


def test_mp_server_config_accepts_xpu_transfer_mode() -> None:
    """Verify the CLI parser accepts the XPU transfer mode."""
    parser = add_mp_server_args(argparse.ArgumentParser())
    args = parser.parse_args(["--supported-transfer-mode", "xpu"])

    config = parse_args_to_mp_server_config(args)

    assert config.supported_transfer_mode == "xpu"


def test_build_modules_uses_xpu_transfer_module() -> None:
    """Verify explicit XPU transfer mode installs the XPU transfer module."""
    modules = _build_modules(MagicMock(), MPServerConfig(supported_transfer_mode="xpu"))

    assert any(isinstance(module, XpuTransferModule) for module in modules)
    assert not any(isinstance(module, GPUTransferModule) for module in modules)
    assert not any(isinstance(module, NonGPUTransferModule) for module in modules)


def test_build_modules_auto_uses_xpu_transfer_module(monkeypatch) -> None:
    """Verify auto mode selects the XPU IPC path on XPU workers."""
    # First Party
    import lmcache.v1.multiprocess.server as server

    monkeypatch.setattr(server, "torch_device_type", "xpu")

    modules = _build_modules(
        MagicMock(), MPServerConfig(supported_transfer_mode="auto")
    )

    assert any(isinstance(module, XpuTransferModule) for module in modules)
    assert any(isinstance(module, NonGPUTransferModule) for module in modules)
    assert not any(isinstance(module, GPUTransferModule) for module in modules)


def test_build_modules_rejects_blend_with_xpu_transfer_mode() -> None:
    """Verify blend mode rejects XPU because blend requires CUDA IPC events."""
    config = MPServerConfig(supported_transfer_mode="xpu", engine_type="blend")

    try:
        _build_modules(MagicMock(), config)
    except ValueError as exc:
        assert "Blend engine requires" in str(exc)
    else:
        raise AssertionError("Expected blend + xpu transfer mode to fail")


def test_create_transfer_context_uses_xpu_context() -> None:
    """Verify XPU tensors select the XPU handle transfer context."""
    fake_tensor = SimpleNamespace(device=SimpleNamespace(type="xpu"))

    context = create_transfer_context({"layer_0": fake_tensor})

    assert isinstance(context, XpuHandleTransferContext)

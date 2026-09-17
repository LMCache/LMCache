# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for NPU pointer-backed tensor construction."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.npu import device_ops

pytestmark = pytest.mark.no_shared_allocator


def _fake_cpu_storage_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, torch.device, int]]:
    """Redirect NPU storage construction to real CPU storage construction.

    The real constructor covers the metadata plumbing on CPU buffers while
    recording the arguments the NPU backend passed through.
    """
    calls: list[tuple[int, torch.device, int]] = []
    real = torch._C._construct_storage_from_data_pointer

    def fake(ptr: int, device: torch.device, nbytes: int) -> Any:
        calls.append((ptr, device, nbytes))
        return real(ptr, torch.device("cpu"), nbytes)

    monkeypatch.setattr(torch._C, "_construct_storage_from_data_pointer", fake)
    return calls


def test_constructs_aliasing_view_from_data_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The view aliases the pointed-to buffer with the requested metadata."""
    calls = _fake_cpu_storage_constructor(monkeypatch)
    original = torch.arange(24, dtype=torch.float32)

    view = device_ops.NpuDeviceOps().tensor_from_ptr(
        original.data_ptr(),
        (4, 6),
        torch.float32,
        torch.device("npu:0"),
    )

    assert view.shape == (4, 6)
    assert view.stride() == (6, 1)
    assert view.dtype == torch.float32
    assert calls == [(original.data_ptr(), torch.device("npu:0"), 96)]

    # A copy fallback would break the paged-buffer ownership contract:
    # writes through the view must reach the caller's buffer.
    view.fill_(7.0)
    assert torch.equal(original, torch.full_like(original, 7.0))


def test_constructor_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage-construction failures surface as RuntimeError."""

    def broken(ptr: int, device: torch.device, nbytes: int) -> Any:
        raise RuntimeError("no NPU runtime")

    monkeypatch.setattr(torch._C, "_construct_storage_from_data_pointer", broken)

    with pytest.raises(RuntimeError, match="non-owning tensor"):
        device_ops.NpuDeviceOps().tensor_from_ptr(
            0x1000, (2, 3), torch.float16, torch.device("npu:0")
        )

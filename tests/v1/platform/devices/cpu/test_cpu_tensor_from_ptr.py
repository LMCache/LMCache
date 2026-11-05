# SPDX-License-Identifier: Apache-2.0
"""Tests for CPU pointer-backed tensor construction."""

# Standard
import ctypes

# Third Party
import torch

# First Party
from lmcache.v1.platform.devices.cpu.device_ops import CpuDeviceOps


def test_cpu_device_ops_constructs_zero_copy_tensor() -> None:
    """The CPU strategy returns a view over the supplied host allocation."""
    values = (ctypes.c_float * 6)(*range(6))

    tensor = CpuDeviceOps().tensor_from_ptr(
        ctypes.addressof(values),
        (2, 3),
        torch.float32,
        torch.device("cpu"),
    )

    tensor[1, 2] = 17
    assert tensor.shape == (2, 3)
    assert values[5] == 17

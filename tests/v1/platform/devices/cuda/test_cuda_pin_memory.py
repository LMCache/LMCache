# SPDX-License-Identifier: Apache-2.0
"""A handled host-registration failure must not leave an error pending.

The accelerator runtime keeps a failed call's error until its last-error
function reads it, and torch reads it after the thread's next kernel launch.
Callers such as the Device-DAX allocators fall back to pageable copies when
pinning fails, so that launch must not fail on their behalf.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform import current_device_spec

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA-compatible runtime", allow_module_level=True)


@pytest.mark.parametrize("operation", ["pin", "unpin"])
def test_failed_registration_does_not_fail_the_next_kernel_launch(
    operation: str,
) -> None:
    """A rejected pin or unpin leaves the next launch in this thread working."""
    device = torch.device(torch_device_type)
    torch.zeros(1, device=device)
    backend_cls = current_device_spec.pin_memory_backend
    assert backend_cls is not None
    backend = backend_cls()
    host = torch.empty(4096, dtype=torch.uint8)
    if operation == "pin":
        # A NULL pointer is rejected with an invalid-value runtime error.
        assert backend.pin_memory(0, 0) is False
    else:
        assert backend.unpin_memory(host.data_ptr()) is False

    assert torch.ones(4, device=device).sum().item() == 4

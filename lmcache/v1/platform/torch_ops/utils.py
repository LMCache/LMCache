# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.platform._device_detect import (
    get_torch_device,
)


def get_gpu_pci_bus_id(device_id: int = 0) -> str | None:
    """
    Get the PCI bus ID via CUDA/ROCm runtime.
    Other backends return None.

    Args:
        device_id (int): CUDA/ROCm device index.

    Returns:
        str | None: PCI bus ID (e.g., "0000:29:00.0") or None if unavailable.
    """
    torch_dev, _ = get_torch_device()

    try:
        if torch_dev.is_available() and device_id < torch_dev.device_count():
            props = torch_dev.get_device_properties(device_id)
            # PCI function number is always 0 for GPUs
            bus_id = (
                f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
                f"{props.pci_device_id:02x}.0"
            )
            return bus_id.upper()
    except Exception:
        pass

    return None

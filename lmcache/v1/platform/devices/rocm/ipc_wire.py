# SPDX-License-Identifier: Apache-2.0
"""ROCm IPC wire descriptors."""

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wire import VmmCudaIPCData


class VmmRocmIPCData(VmmCudaIPCData, tag="rocm_vmm"):
    """HIP VMM allocation descriptor; same fields as :class:`VmmCudaIPCData`."""

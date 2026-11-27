# SPDX-License-Identifier: Apache-2.0
"""CPU shared-memory IPC wire descriptor."""

# First Party
from lmcache.v1.platform.base.ipc_wire import IPCData, NonNegative


class CpuShmIPCData(IPCData, tag="cpu_shm"):
    """Named CPU shared-memory descriptor."""

    shm_name: str
    nbytes: NonNegative

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.nbytes:
            raise ValueError("IPC view exceeds CPU shared-memory segment")

# SPDX-License-Identifier: Apache-2.0
"""GCU IPC wire descriptor."""

# First Party
from lmcache.v1.platform.base.ipc_wire import IPCData, NonNegative


class GcuIPCData(IPCData, tag="gcu"):
    """PyTorch shared GCU storage descriptor."""

    handle: tuple[
        int,
        bytes | None,
        NonNegative,
        NonNegative,
        bytes | None,
        NonNegative,
        bytes | None,
        bool,
    ]

    def __post_init__(self) -> None:
        if self.metadata.byte_span() > self.handle[2]:
            raise ValueError("IPC view exceeds GCU storage")

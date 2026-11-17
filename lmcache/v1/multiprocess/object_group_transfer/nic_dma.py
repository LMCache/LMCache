# SPDX-License-Identifier: Apache-2.0
"""NIC DMA engine: objects a peer owns. Not implemented yet.

Placeholder so a peer-owned object selects its own engine and fails
clearly. The engine will read such objects into the staging buffer over
the transfer channel (RFC #6399).
"""

# Standard
from typing import Sequence

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
import lmcache.lmcache_native as lmcache_native


class NicDmaEngine:
    """Placeholder for staging objects a peer owns."""

    def stage(
        self,
        memory_objs: Sequence[MemoryObj],
        buffers: Sequence[torch.Tensor],
        direction: "lmcache_native.TransferDirection",
    ) -> None:
        """Reject the transfer: objects a peer owns cannot be staged yet.

        Args:
            memory_objs: Objects a peer owns.
            buffers: One staging region per object.
            direction: H2D or D2H.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "The NIC DMA engine is not implemented: objects a peer owns cannot "
            "be transferred to or from GPU memory"
        )

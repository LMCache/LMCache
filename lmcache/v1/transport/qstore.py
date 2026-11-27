# SPDX-License-Identifier: Apache-2.0
"""Wire types of the qstore server module."""

# Third Party
import msgspec

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class QRingInfo(msgspec.Struct):
    """One worker's Q ring, as an SDK needs it to transfer through the ring.

    Returned by ``get_q_ring``. The ring's GPU memory is reached through its
    original IPC handles, as the serving engine registered them; shapes and
    dtype come from the rebuilt tensors.

    Attributes:
        instance_id: The worker instance the ring is registered under.
        worker_id: The worker's rank, as last reported; -1 before its first
            report.
        q_caches: The ring's per-layer IPC wrappers, in registration order.
        used_blocks: The ring's blocks vLLM holds, as of the worker's last
            report; every other block is free for an SDK transfer.
        num_alloc: The ring's allocation count at the worker's last report;
            a change means vLLM took free blocks.
    """

    instance_id: int
    worker_id: int
    q_caches: list[DeviceIPCWrapper]
    used_blocks: list[int]
    num_alloc: int

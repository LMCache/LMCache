# SPDX-License-Identifier: Apache-2.0
"""ROCm device-event IPC backend.

ROCm takes the CUDA-style interprocess event API from ``torch.cuda``, but a
stream wait on an *imported* IPC event is not a command-processor barrier
there: HIP launches ``__amd_rocclr_streamOpsWait``, a one-work-item kernel that
spins on a compute unit until the event's signal is written. (A CP wait would
need a ROCr runtime signal behind the event's memory, which the host-registered IPC
shared memory does not have.)

The server's transfers (``lmcache_driven`` STORE and RETRIEVE, the CacheBlend
retrieve, the Q store) make a server stream wait on the producer fence the
engine recorded on its compute stream. The engine may
record that fence behind work it has queued but not yet started. If that work
contains a kernel whose workgroups must all be resident at once (a persistent
kernel that takes every CU and has its workgroups wait on each other), the
spinning waiter holds a CU before all of that kernel's workgroups are placed:
one of them never is, the kernel never finishes, the fence never signals, and
the waiter never exits. Every engine rank that synchronizes with that one then
hangs too.

This backend sets ``host_ready_before_imported_wait``: ``wait_producer_event``
polls the imported event from the host (``hipEventQuery``) until it has
completed and only then queues the stream wait, which no longer spins. The
transfer starts a poll interval after the fence completes instead of the
moment it does. If the engine is no longer registered, or the fence is still
pending at the deadline, the poll gives up and that transfer fails instead of
holding the server's worker thread.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend


class RocmEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend over ``torch.cuda`` (HIP) interprocess events.

    Args:
        event_module: Object exposing the CUDA-style ``Event`` class. Defaults
            to ``torch.cuda``; injectable for testing.
        device_type: Device-type label used in error messages.
    """

    host_ready_before_imported_wait = True

    def __init__(
        self,
        event_module: object | None = None,
        device_type: str = "cuda",
    ) -> None:
        if event_module is None:
            # Third Party
            import torch

            event_module = torch.cuda
        super().__init__(event_module=event_module, device_type=device_type)

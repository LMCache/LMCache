# SPDX-License-Identifier: Apache-2.0
"""CPU-only device-context + interprocess Event fallbacks.

These tiny stand-ins satisfy the structural API expected by the
multiprocess server on hosts where CUDA is unavailable.  No
``torch.cuda.*`` calls are issued from this module.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any


class NoopDeviceContext:
    """No-op replacement for ``torch.cuda.device`` + ``torch.cuda.stream``."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:  # noqa: ARG002
        return None


class MockInterprocessEvent:
    """Pure-Python stand-in for ``torch.cuda.Event(interprocess=True)``.

    Implements the minimal subset of the Event API the multiprocess
    server actually calls (``record``, ``wait``, ``ipc_handle``).  All
    methods are no-ops because there is no real GPU work to fence on a
    CPU-only host.
    """

    def record(self, stream: Any = None) -> None:  # noqa: ARG002
        return None

    def wait(self, stream: Any = None) -> None:  # noqa: ARG002
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def ipc_handle(self) -> bytes:
        # 64-byte zero-filled handle keeps downstream length checks
        # happy without exposing any real shareable state.
        return b"\x00" * 64

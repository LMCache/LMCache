# SPDX-License-Identifier: Apache-2.0
"""Let host pinning yield to in-flight transfer submissions.

``cudaHostRegister`` holds a driver lock for its whole duration and that
lock is not fair: a thread that registers chunks back to back starves every
other CUDA API call in the process, including the kernel launches and event
records that submit KV transfers. Between chunks the pinning thread asks the
pacer whether a transfer is being submitted and, if so, waits for it to
finish before taking the lock again. When nothing is in flight the check
returns immediately.
"""

# Standard
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
import threading


def submitting(pacer: "PinPacer | None") -> AbstractContextManager[None]:
    """``pacer.submitting()``, or a no-op when the allocator has no pacer."""
    if pacer is None:
        return nullcontext()
    return pacer.submitting()


class PinPacer:
    """Count in-flight transfer submissions; pinning waits for zero."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._submitting = 0

    @contextmanager
    def submitting(self) -> Iterator[None]:
        """Mark the calling thread as submitting GPU work."""
        with self._cond:
            self._submitting += 1
        try:
            yield
        finally:
            with self._cond:
                self._submitting -= 1
                if self._submitting == 0:
                    self._cond.notify_all()

    def wait_idle(self, max_wait_s: float) -> bool:
        """Block until no submission is in flight or ``max_wait_s`` elapses.

        Args:
            max_wait_s: Upper bound on the wait, in seconds.

        Returns:
            True if idle was observed, False on timeout. The caller pins
            either way; the timeout only bounds starvation under sustained
            load.
        """
        with self._cond:
            return self._cond.wait_for(
                lambda: self._submitting == 0, timeout=max_wait_s
            )

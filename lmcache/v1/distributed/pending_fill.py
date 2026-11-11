# SPDX-License-Identifier: Apache-2.0
"""Fill tracking for L1 objects that become readable before they are written.

A pipelined load commits its L1 objects as soon as their compressed bytes are
in memory, then decompresses them in the background. A :class:`FillTracker`
reports how far that has got: it covers a batch of same-layout objects that
fill front to back together, so one byte watermark holds for all of them.

Readers look an object up by its data pointer with :func:`find` and wait for
the bytes they need. ``StorageManager`` read paths wait for whole objects;
a layer-wise retrieve waits for one layer group at a time.

A tracker stays registered until it settles successfully, or, after a failure,
until its objects are deleted, so a reader never sees a failed object's
garbage as data.
"""

# Standard
from collections.abc import Callable, Iterable
import threading

# First Party
from lmcache.v1.memory_management import MemoryObj

# Upper bound on one wait; decompressing a request's KV takes milliseconds to
# seconds, so this only trips on a hung backend.
_WAIT_TIMEOUT_S = 60.0


class FillFailedError(RuntimeError):
    """The background fill of an object failed or did not finish in time."""


class FillTracker:
    """Fill progress of a batch of L1 objects being written in the background.

    Every object fills front to back; ``filled`` bytes from the start of each
    object are final. Settle callbacks run once, outside the tracker's lock,
    in the thread that settles it or, if it already settled, in the caller's.

    Args:
        objs: Objects the fill writes; registered for :func:`find` until the
            tracker is released.
    """

    def __init__(self, objs: Iterable[MemoryObj]) -> None:
        self._cond = threading.Condition()
        self._filled = 0
        self._settled = False
        self._ok = False
        self._callbacks: list[Callable[[bool], None]] = []
        self._ptrs = [obj.data_ptr for obj in objs]
        _register(self)

    @property
    def settled(self) -> bool:
        """Whether the fill finished, successfully or not."""
        with self._cond:
            return self._settled

    @property
    def failed(self) -> bool:
        """Whether the fill settled unsuccessfully."""
        with self._cond:
            return self._settled and not self._ok

    def advance(self, filled: int) -> None:
        """Mark the first ``filled`` bytes of every object final."""
        with self._cond:
            if filled > self._filled:
                self._filled = filled
                self._cond.notify_all()

    def settle(self, ok: bool) -> None:
        """Finish the fill. A successful fill is released before its
        callbacks run; a failed one stays registered until :meth:`release`.
        """
        with self._cond:
            if self._settled:
                return
            self._settled = True
            self._ok = ok
            callbacks, self._callbacks = self._callbacks, []
            self._cond.notify_all()
        if ok:
            self.release()
        for callback in callbacks:
            callback(ok)

    def on_settled(self, callback: Callable[[bool], None]) -> None:
        """Run ``callback(ok)`` once the fill settles, now if it has."""
        with self._cond:
            if not self._settled:
                self._callbacks.append(callback)
                return
            ok = self._ok
        callback(ok)

    def wait(self, end: int | None = None, timeout: float = _WAIT_TIMEOUT_S) -> None:
        """Wait until the first ``end`` bytes are final, or the whole fill
        succeeded when ``end`` is None.

        Raises:
            FillFailedError: If the fill failed before covering ``end``, or
                did not get there within ``timeout`` seconds.
        """
        with self._cond:
            done = self._cond.wait_for(
                lambda: self._settled or (end is not None and self._filled >= end),
                timeout,
            )
            if end is not None and self._filled >= end:
                return
            if not done:
                raise FillFailedError(f"fill did not finish within {timeout} s")
            if not self._ok:
                raise FillFailedError("background fill failed")

    def release(self) -> None:
        """Stop serving :func:`find` for this tracker's objects."""
        with _lock:
            for ptr in self._ptrs:
                if _fills.get(ptr) is self:
                    del _fills[ptr]


_lock = threading.Lock()
_fills: dict[int, FillTracker] = {}


def _register(tracker: FillTracker) -> None:
    with _lock:
        for ptr in tracker._ptrs:
            _fills[ptr] = tracker


def find(obj: MemoryObj) -> FillTracker | None:
    """Return the fill still writing ``obj``, or None."""
    if not _fills:
        return None
    with _lock:
        return _fills.get(obj.data_ptr)


def wait_filled(objs: Iterable[MemoryObj]) -> list[bool]:
    """Wait for every object's fill; return per object whether it is valid."""
    if not _fills:
        return [True for _ in objs]
    valid: list[bool] = []
    for obj in objs:
        tracker = find(obj)
        if tracker is None:
            valid.append(True)
            continue
        try:
            tracker.wait()
        except FillFailedError:
            valid.append(False)
        else:
            valid.append(True)
    return valid

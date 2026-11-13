# SPDX-License-Identifier: Apache-2.0
"""Hard byte cap on a scratch dir for cache layers that have none.

vLLM's fs offload tier never evicts and has no size option. Taking the write
bits off every directory stops new block files without root; the store then
fails with EACCES, handled like ENOSPC (block dropped, no request fails).
"""

# Standard
from pathlib import Path
from typing import Any
import os
import stat
import threading
import time


def allocated_bytes(path: Path) -> int:
    """Return the bytes allocated under ``path``.

    Not ``du``: it aborts when the tree changes under it.

    Args:
        path: Directory to measure.

    Returns:
        Sum of ``st_blocks * 512`` over every non-directory entry under it.
    """
    total = 0
    stack = [str(path)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except FileNotFoundError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        total += e.stat(follow_symlinks=False).st_blocks * 512
                except FileNotFoundError:
                    pass
    return total


def set_dirs_writable(path: Path, writable: bool) -> None:
    """Add or remove the write bits of ``path`` and every directory under it.

    Top-down; a second freeze pass catches dirs created during the walk.

    Args:
        path: Root of the tree.
        writable: True restores the owner's write bit, False removes every
            write bit.
    """
    for d, _, _ in os.walk(path):
        mode = stat.S_IMODE(os.stat(d).st_mode)
        os.chmod(d, mode | stat.S_IWUSR if writable else mode & ~0o222)


class DiskCap:
    """Freeze ``path`` before it holds more than ``cap_bytes``.

    A thread measures the dir every ``poll_s`` and freezes it once used bytes
    plus twice the last interval's growth reach the cap. It keeps measuring
    after the freeze, because in-flight stores still land.
    """

    def __init__(self, path: Path, cap_bytes: int, poll_s: float = 5.0) -> None:
        """Args:
        path: Scratch dir the cache layer writes into.
        cap_bytes: Bytes it may hold.
        poll_s: Seconds between measurements.
        """
        self.path = path
        self.cap_bytes = cap_bytes
        self.poll_s = poll_s
        self.peak_bytes = 0
        self.frozen_at: float | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start measuring."""
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop measuring and return the result.

        Returns:
            ``cap_bytes``, ``used_bytes`` (measured now), ``peak_bytes``
            (largest measurement, this one included), ``frozen_at`` (unix
            time, None if the cap was never approached) and ``error`` (why
            the watch died early, else None: the cap was then not enforced).
        """
        self._stop.set()
        self._thread.join()
        used = allocated_bytes(self.path)
        self.peak_bytes = max(self.peak_bytes, used)
        return {
            "cap_bytes": self.cap_bytes,
            "used_bytes": used,
            "peak_bytes": self.peak_bytes,
            "frozen_at": self.frozen_at,
            "error": self.error,
        }

    def _loop(self) -> None:
        last = 0
        try:
            while not self._stop.wait(self.poll_s):
                used = allocated_bytes(self.path)
                self.peak_bytes = max(self.peak_bytes, used)
                margin = 2 * max(used - last, 0)
                last = used
                if self.frozen_at is None and used + margin >= self.cap_bytes:
                    set_dirs_writable(self.path, False)
                    set_dirs_writable(self.path, False)
                    self.frozen_at = time.time()
        except Exception as e:
            # Recorded, not raised: a thread's exception would only be printed.
            self.error = repr(e)

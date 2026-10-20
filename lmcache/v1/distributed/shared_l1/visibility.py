# SPDX-License-Identifier: Apache-2.0
"""CPU cache maintenance around copies into and out of a shared region.

CXL 2.0 multi-host memory gives no cross-host cache coherence: a host may keep
a stale line of bytes another host just wrote, and an Intel host with DDIO may
hold DMA-written lines in its LLC instead of the device. ``software_fenced``
writes the range back after the device-to-host copy completes (publish) and
drops it before a host-to-device copy reads it (acquire). ``coherent`` does
nothing; it fits a region that only one host maps, or a coherent fabric.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported
from lmcache.v1.memory_orchestrator.api import VISIBILITY_MODES

# A batch of objects is flushed on this many threads: one 36 MB object takes
# longer to flush on one core than to copy over CXL.
_FLUSH_THREADS = 8


class Visibility:
    """Publish/acquire fences for a region's ``visibility_mode``.

    Args:
        mode: ``"software_fenced"`` (CLFLUSHOPT over the range, then SFENCE, on
            both sides of every copy) or ``"coherent"`` (no maintenance).

    Raises:
        ValueError: The mode is unknown.
        RuntimeError: ``software_fenced`` on a CPU architecture without a
            cache-flush implementation.
    """

    def __init__(self, mode: str) -> None:
        if mode not in VISIBILITY_MODES:
            raise ValueError(f"Unknown shared-L1 visibility_mode {mode!r}")
        self.mode = mode
        self._executor: ThreadPoolExecutor | None = None
        if mode == "software_fenced":
            if not cache_flush_supported():
                raise RuntimeError(
                    "visibility_mode software_fenced needs cache-line flushes, "
                    "which this CPU architecture does not implement"
                )
            self._executor = ThreadPoolExecutor(
                max_workers=_FLUSH_THREADS, thread_name_prefix="shared-l1-flush"
            )

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Make written ``(address, size)`` ranges visible to other hosts.

        Args:
            ranges: ``(address, size)`` pairs in this process whose
                device-to-host copies have completed.
        """
        self._flush(ranges)

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Drop this host's cached lines of ``(address, size)`` ranges.

        Args:
            ranges: ``(address, size)`` pairs a host-to-device copy will read.
        """
        self._flush(ranges)

    def close(self) -> None:
        """Stop the flush threads, if any."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _flush(self, ranges: list[tuple[int, int]]) -> None:
        # Each flush ends in its own SFENCE; waiting on the futures orders
        # those fences before whatever the caller does next.
        if self._executor is None:
            return
        if len(ranges) == 1:
            cache_flush_range(*ranges[0])
            return
        for future in [
            self._executor.submit(cache_flush_range, address, size)
            for address, size in ranges
        ]:
            future.result()

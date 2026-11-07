# SPDX-License-Identifier: Apache-2.0
"""CPU cache maintenance around copies into and out of a shared region.

CXL 2.0 multi-host memory is not coherent across hosts, and DDIO may keep
DMA-written lines in the LLC, so ``software_fenced`` writes ranges back after
device-to-host copies and drops them before host-to-device copies.
"""

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported
from lmcache.v1.memory_orchestrator.api import VISIBILITY_MODES


class Visibility:
    """Publish/acquire fences for a region's ``visibility_mode``.

    Args:
        mode: ``"software_fenced"`` (CLFLUSHOPT over the range, then SFENCE, on
            both sides of every copy) or ``"coherent"`` (no maintenance: one
            host, or a coherent fabric).

    Raises:
        ValueError: The mode is unknown.
        RuntimeError: ``software_fenced`` on a CPU without CLFLUSHOPT.
    """

    def __init__(self, mode: str) -> None:
        if mode not in VISIBILITY_MODES:
            raise ValueError(f"Unknown shared-L1 visibility_mode {mode!r}")
        if mode == "software_fenced" and not cache_flush_supported():
            raise RuntimeError(
                "visibility_mode software_fenced needs CLFLUSHOPT, which this CPU "
                "does not implement"
            )
        self.mode = mode

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Write back ranges whose device-to-host copies completed.

        Args:
            ranges: ``(address, size)`` pairs in this process.
        """
        self._flush(ranges)

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Drop this host's cached lines of ranges a host-to-device copy reads.

        Args:
            ranges: ``(address, size)`` pairs in this process.
        """
        self._flush(ranges)

    def _flush(self, ranges: list[tuple[int, int]]) -> None:
        # Each flush ends in an SFENCE, which orders it before whatever the
        # caller does next.
        if self.mode == "software_fenced":
            for address, size in ranges:
                cache_flush_range(address, size)

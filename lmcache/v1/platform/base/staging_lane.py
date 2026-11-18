# SPDX-License-Identifier: Apache-2.0
"""Staging lanes: a load stream with two staging banks, so a batch loads
into one bank while the stream that empties them scatters the other::

    load stream     | load b0 -> bank 0 | load b1 -> bank 1 | load b2 -> bank 0 |
    scatter stream  |                   | scatter bank 0    | scatter bank 1    |
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import time

# Third Party
import torch

# First Party
from lmcache.v1.platform import stream as platform_stream

BANKS_PER_LANE = 2
_HOST_POLL_SECONDS = 20e-6


class StagingBank(Protocol):
    """One staging buffer, addressed like the cache context's own."""

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Return the typed view of a (slot, kernel group) pair."""
        ...

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Return the flat uint8 view of a (slot, object group) pair."""
        ...


class BankSignals:
    """Per bank, "staged" (the scatter may read it) and "free" (the next
    load may overwrite it), as stream events.

    One round per batch: :meth:`wait_free`, :meth:`signal_staged`,
    :meth:`wait_staged`, :meth:`signal_free`. A round that signals "staged"
    must also signal "free", or the bank's next load never starts.

    Args:
        device: The device that owns the lane's streams.
    """

    def __init__(self, device: object) -> None:
        self._device = device
        self._staged: list[platform_stream.CompletionEvent | None] = [
            None
        ] * BANKS_PER_LANE
        self._free: list[platform_stream.CompletionEvent | None] = [
            None
        ] * BANKS_PER_LANE

    def wait_free(self, bank: int, stream: object) -> None:
        """Make ``stream`` wait until the last scatter of ``bank`` ran."""
        event = self._free[bank]
        if event is not None:
            event.wait_on(stream)

    def wait_free_on_host(self, bank: int) -> None:
        """Block until the last scatter of ``bank`` ran; for loads done by
        host threads, which cannot queue the wait on a stream."""
        event = self._free[bank]
        while event is not None and not event.is_complete():
            time.sleep(_HOST_POLL_SECONDS)

    def signal_staged(self, bank: int, stream: object) -> None:
        """Signal "staged" for ``bank`` after the work queued on ``stream``."""
        self._staged[bank] = platform_stream.record_completion_event(
            self._device, stream
        )

    def wait_staged(self, bank: int, stream: object) -> None:
        """Make ``stream`` wait for the last "staged" signal of ``bank``."""
        event = self._staged[bank]
        if event is not None:
            event.wait_on(stream)

    def signal_free(self, bank: int, stream: object) -> None:
        """Signal "free" for ``bank`` after the work queued on ``stream``."""
        self._free[bank] = platform_stream.record_completion_event(self._device, stream)


@dataclass
class StagingLane:
    """A load stream, its staging banks and their signals. Used by one
    thread at a time, like the cache context's own staging buffer.

    Attributes:
        stream: The stream the lane's loads are queued on.
        banks: The lane's staging banks.
        signals: The banks' "staged" and "free" signals.
    """

    stream: Any
    banks: Sequence[StagingBank]
    signals: BankSignals
    _next_bank: int = field(default=0, repr=False)

    def take_bank(self) -> int:
        """Return the bank index for the next batch; banks alternate."""
        bank = self._next_bank
        self._next_bank = (bank + 1) % len(self.banks)
        return bank

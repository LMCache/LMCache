# SPDX-License-Identifier: Apache-2.0
"""Staging lanes: a load stream with two staging banks, and the signals
that hand a bank between the load stream and the stream that empties it.

A retrieve that stages its chunks runs two stream-ordered steps per batch:
the load (L1 object to staging bank) and the scatter (staging bank to the
paged KV blocks). On one stream the two alternate. With a lane the load
runs on the lane's stream and the scatter on another stream, so the next
batch loads into the second bank while the previous one scatters::

    load stream     | load b0 -> bank 0 | load b1 -> bank 1 | load b2 -> bank 0 |
    scatter stream  |                   | scatter bank 0    | scatter bank 1    |

Each bank needs two signals: "staged" (the load is done, the scatter may
read the bank) and "free" (the scatter is done, the next load may overwrite
the bank). :class:`BankSignals` is that pair; :class:`EventBankSignals`
implements it with stream events and works on any device that has them.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# Third Party
import torch

# First Party
from lmcache.v1.platform import stream as platform_stream

BANKS_PER_LANE = 2


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


class BankSignals(ABC):
    """The "staged" and "free" signals of one lane's banks.

    Per bank the four calls repeat in this order, one round per batch:
    :meth:`wait_free`, :meth:`signal_staged`, :meth:`wait_staged`,
    :meth:`signal_free`. Every call only queues work on ``stream``; none
    blocks the caller. A round that queued :meth:`signal_staged` must also
    queue :meth:`signal_free`, or the bank's next load never starts.
    """

    @abstractmethod
    def wait_free(self, bank: int, stream: object) -> None:
        """Make ``stream`` wait until the previous scatter of ``bank`` ran.

        Args:
            bank: Bank index in the lane.
            stream: The stream about to load into the bank.
        """

    @abstractmethod
    def signal_staged(self, bank: int, stream: object) -> None:
        """Signal "staged" for ``bank`` after the work queued on ``stream``.

        Args:
            bank: Bank index in the lane.
            stream: The stream that loaded into the bank.
        """

    @abstractmethod
    def wait_staged(self, bank: int, stream: object) -> None:
        """Make ``stream`` wait for the last "staged" signal of ``bank``.

        Args:
            bank: Bank index in the lane.
            stream: The stream about to read the bank.
        """

    @abstractmethod
    def signal_free(self, bank: int, stream: object) -> None:
        """Signal "free" for ``bank`` after the work queued on ``stream``.

        Args:
            bank: Bank index in the lane.
            stream: The stream that read the bank.
        """


class EventBankSignals(BankSignals):
    """Bank signals as stream events: record on one stream, wait on the other.

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
        event = self._free[bank]
        # A bank no scatter has read yet is free.
        if event is not None:
            event.wait_on(stream)

    def signal_staged(self, bank: int, stream: object) -> None:
        self._staged[bank] = platform_stream.record_completion_event(
            self._device, stream
        )

    def wait_staged(self, bank: int, stream: object) -> None:
        event = self._staged[bank]
        if event is not None:
            event.wait_on(stream)

    def signal_free(self, bank: int, stream: object) -> None:
        self._free[bank] = platform_stream.record_completion_event(self._device, stream)


@dataclass
class StagingLane:
    """A load stream, its staging banks and their signals.

    One thread uses a lane at a time: the cache context that owns it is
    driven by one transfer at a time, like its own staging buffer.

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
        """Return the index of the bank for the next batch.

        Banks are handed out in turn, so a batch never loads into the bank
        the batch before it is still scattering from.

        Returns:
            The bank index.
        """
        bank = self._next_bank
        self._next_bank = (bank + 1) % len(self.banks)
        return bank

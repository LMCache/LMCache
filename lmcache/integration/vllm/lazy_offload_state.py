# SPDX-License-Identifier: Apache-2.0
"""Request-epoch state for scheduler-side lazy offload orchestration."""

# Standard
from dataclasses import dataclass
import enum


class RequestPhase(enum.Enum):
    """Lifecycle phase of the request generation currently using an id."""

    ACTIVE = enum.auto()
    FINISHED = enum.auto()


@dataclass(frozen=True)
class SubmittedStoreBatch:
    """One submitted batch and the store epoch that produced it."""

    epoch: int
    block_ids: tuple[int, ...]


@dataclass
class RequestSlot:
    """All controller-owned state associated with one reusable request id."""

    epoch: int = 0
    phase: RequestPhase = RequestPhase.ACTIVE
    in_flight: SubmittedStoreBatch | None = None
    awaiting_rearrival: bool = False


class LazyOffloadRequestRegistry:
    """Own request epochs and the single submitted batch allowed per id.

    vLLM may recreate a tracker after preemption and may reuse a finished
    request id while its previous store receipt is outstanding. Epochs make
    those generations explicit. Worker receipts are still keyed by request id,
    so the registry permits only one submitted batch at a time.
    """

    def __init__(self) -> None:
        self._slots: dict[str, RequestSlot] = {}

    def ensure_active(self, request_id: str) -> int:
        """Create initial state for a request first observed via admission."""
        slot = self._slots.setdefault(request_id, RequestSlot())
        return slot.epoch

    def arrive(self, request_id: str) -> int:
        """Record a newly created tracker and return its store epoch."""
        slot = self._slots.get(request_id)
        if slot is None:
            self._slots[request_id] = RequestSlot()
            return 0
        if slot.awaiting_rearrival:
            # on_request_reset already advanced the epoch before vLLM
            # recreated the tracker.
            slot.awaiting_rearrival = False
        elif slot.phase is RequestPhase.FINISHED:
            # A distinct client request reused a finished predecessor's id.
            slot.epoch += 1
        slot.phase = RequestPhase.ACTIVE
        return slot.epoch

    def reset(self, request_id: str) -> int:
        """Advance the store epoch before a preempted tracker is recreated."""
        slot = self._slots.setdefault(request_id, RequestSlot())
        slot.epoch += 1
        slot.phase = RequestPhase.ACTIVE
        slot.awaiting_rearrival = True
        return slot.epoch

    def finish(self, request_id: str) -> None:
        """Mark the current generation finished."""
        slot = self._slots.setdefault(request_id, RequestSlot())
        slot.phase = RequestPhase.FINISHED
        slot.awaiting_rearrival = False

    def is_finished(self, request_id: str) -> bool:
        slot = self._slots.get(request_id)
        return slot is not None and slot.phase is RequestPhase.FINISHED

    def finished_request_ids(self) -> set[str]:
        return {
            request_id
            for request_id, slot in self._slots.items()
            if slot.phase is RequestPhase.FINISHED
        }

    def is_active(self, request_id: str) -> bool:
        slot = self._slots.get(request_id)
        return slot is not None and slot.phase is RequestPhase.ACTIVE

    def has_in_flight(self, request_id: str) -> bool:
        slot = self._slots.get(request_id)
        return slot is not None and slot.in_flight is not None

    def in_flight_is_current(self, request_id: str) -> bool:
        """Whether the submitted batch belongs to the current store epoch."""
        slot = self._slots.get(request_id)
        return (
            slot is not None
            and slot.in_flight is not None
            and slot.in_flight.epoch == slot.epoch
        )

    def in_flight_request_ids(self) -> set[str]:
        return {
            request_id
            for request_id, slot in self._slots.items()
            if slot.in_flight is not None
        }

    def register_batch(self, request_id: str, block_ids: list[int]) -> None:
        """Open one receipt window for the request's current epoch."""
        slot = self._slots.setdefault(request_id, RequestSlot())
        if slot.in_flight is not None:
            raise RuntimeError(
                f"request {request_id!r} already has an in-flight store batch"
            )
        slot.in_flight = SubmittedStoreBatch(slot.epoch, tuple(block_ids))

    def complete_batch(self, request_id: str) -> SubmittedStoreBatch:
        """Close and return the request's submitted batch."""
        slot = self._slots[request_id]
        if slot.in_flight is None:
            raise KeyError(request_id)
        batch = slot.in_flight
        slot.in_flight = None
        return batch

    def can_end_session(self, request_id: str) -> bool:
        """Whether controller-owned state permits session teardown."""
        slot = self._slots.get(request_id)
        return (
            slot is not None
            and slot.phase is RequestPhase.FINISHED
            and slot.in_flight is None
        )

    def session_ended(self, request_id: str) -> None:
        """Prune an idle finished slot after its session teardown action."""
        if self.can_end_session(request_id):
            del self._slots[request_id]

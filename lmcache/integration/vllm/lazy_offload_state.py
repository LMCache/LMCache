# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle state for scheduler-side lazy offload orchestration.

Generation, receipt, pin and orphaned are defined in
``lazy_offload_manager``; store batch, in flight and session in
``lazy_offload_policy.base``.
"""

# Standard
from dataclasses import dataclass, replace
import enum


class RequestPhase(enum.Enum):
    """Lifecycle phase of the request generation currently using an id."""

    ACTIVE = enum.auto()
    FINISHED = enum.auto()


@dataclass(frozen=True)
class SubmittedStoreBatch:
    """The blocks pinned by one submitted store batch.

    Attributes:
        block_ids: The GPU blocks this batch pinned at submission. Its
            completion receipt unpins exactly these.
        orphaned: Whether a preemption reset or an id reuse detached the
            batch from the request generation now using its id. An orphaned
            batch still owns its pins, but its failure is not charged to the
            current generation.
    """

    block_ids: tuple[int, ...]
    orphaned: bool = False


@dataclass
class RequestSlot:
    """All manager-owned state associated with one reusable request id.

    Attributes:
        phase: Whether the generation currently holding the id is still
            running or has finished.
        in_flight: The submitted store batch awaiting its receipt, or None.
            At most one batch may be open, because worker receipts are keyed
            by request id alone.
    """

    phase: RequestPhase = RequestPhase.ACTIVE
    in_flight: SubmittedStoreBatch | None = None


class LazyOffloadRequestRegistry:
    """Own request lifecycle state and the single submitted batch per id.

    A request id outlives one request: vLLM recreates a tracker under the
    same id after preemption, and a later, unrelated request may reuse a
    finished id while its store receipt is still outstanding. Both events
    orphan whatever batch is in flight, so its late receipt releases pins
    without its failure being blamed on the generation now using the id.

    Methods are grouped into those that update the internal state and
    those that query it. Scheduler thread only.
    """

    def __init__(self) -> None:
        """Create an empty registry, holding no request id."""
        self._slots: dict[str, RequestSlot] = {}

    ####
    # Update internal states
    ####

    def arrive(self, request_id: str) -> None:
        """Record that a tracker was created for this request id.

        Three arrivals reach here and the slot's phase tells them apart: a
        first arrival has no slot; a preempted request comes back ACTIVE,
        :meth:`reset` having already orphaned its batch; a distinct request
        reusing a finished id finds the slot FINISHED, and its predecessor's
        outstanding batch is orphaned here.

        Args:
            request_id: The id whose tracker vLLM just created.
        """
        slot = self._slots.get(request_id)
        if slot is None:
            self._slots[request_id] = RequestSlot()
            return
        if slot.phase is RequestPhase.FINISHED:
            self._orphan_in_flight(slot)
        slot.phase = RequestPhase.ACTIVE

    def reset(self, request_id: str) -> None:
        """Detach submitted state before a preempted tracker is recreated.

        The resumed request restarts at token zero under the same id, so a
        batch submitted before the preemption belongs to no live generation
        and is orphaned here. The slot stays ACTIVE, so the re-arrival that
        follows is not mistaken for a distinct request reusing a finished id.

        Args:
            request_id: The preempted request.
        """
        slot = self._slots.setdefault(request_id, RequestSlot())
        self._orphan_in_flight(slot)
        slot.phase = RequestPhase.ACTIVE

    def finish(self, request_id: str) -> None:
        """Mark the generation currently holding this id finished.

        Args:
            request_id: The request vLLM reported as finished.
        """
        slot = self._slots.setdefault(request_id, RequestSlot())
        slot.phase = RequestPhase.FINISHED

    def register_batch(self, request_id: str, block_ids: list[int]) -> None:
        """Record the batch a submission put in flight.

        Args:
            request_id: The request whose store batch was submitted.
            block_ids: The GPU blocks the manager pinned for that batch.

        Raises:
            RuntimeError: If the request already has a batch in flight.
                Receipts are keyed by request id, so a second open batch
                would make them ambiguous.
        """
        slot = self._slots.setdefault(request_id, RequestSlot())
        if slot.in_flight is not None:
            raise RuntimeError(
                f"request {request_id!r} already has an in-flight store batch"
            )
        slot.in_flight = SubmittedStoreBatch(tuple(block_ids))

    def complete_batch(self, request_id: str) -> SubmittedStoreBatch:
        """Clear the in-flight batch and return it.

        Args:
            request_id: The request whose batch every worker has reported.

        Returns:
            The closed batch, whose ``block_ids`` the caller unpins and
            whose ``orphaned`` flag says whether it outlived its generation.

        Raises:
            KeyError: If the request has no slot or no batch in flight.
        """
        slot = self._slots[request_id]
        if slot.in_flight is None:
            raise KeyError(request_id)
        batch = slot.in_flight
        slot.in_flight = None
        return batch

    def session_ended(self, request_id: str) -> None:
        """Drop the slot of a finished, settled request after its teardown.

        Keeps the registry from growing over a long run. A slot that no
        longer satisfies :meth:`can_end_session` is left in place.

        Args:
            request_id: The request whose session the manager just ended.
        """
        if self.can_end_session(request_id):
            del self._slots[request_id]

    @staticmethod
    def _orphan_in_flight(slot: RequestSlot) -> None:
        """Detach the slot's submitted batch from the current generation.

        Args:
            slot: The slot whose open batch, if any, is marked orphaned.
        """
        if slot.in_flight is not None and not slot.in_flight.orphaned:
            slot.in_flight = replace(slot.in_flight, orphaned=True)

    ####
    # Query internal states
    ####

    def is_finished(self, request_id: str) -> bool:
        """Whether the generation holding this id has finished.

        Args:
            request_id: The request id to query.

        Returns:
            True if a slot exists and its phase is FINISHED.
        """
        slot = self._slots.get(request_id)
        return slot is not None and slot.phase is RequestPhase.FINISHED

    def finished_request_ids(self) -> set[str]:
        """The ids whose current generation has finished.

        Returns:
            One drain signal's worth of finished ids, as a new set.
        """
        return {
            request_id
            for request_id, slot in self._slots.items()
            if slot.phase is RequestPhase.FINISHED
        }

    def has_in_flight(self, request_id: str) -> bool:
        """Whether the request has a submitted batch awaiting its receipt.

        Args:
            request_id: The request id to query.

        Returns:
            True while a batch is open, orphaned or not.
        """
        slot = self._slots.get(request_id)
        return slot is not None and slot.in_flight is not None

    def in_flight_is_orphaned(self, request_id: str) -> bool:
        """Whether a preemption reset or an id reuse detached the batch.

        Args:
            request_id: The request id to query.

        Returns:
            True only when a batch is in flight and it no longer belongs to
            the generation currently using the id, which is when its failure
            must not break that generation's prefix chain.
        """
        slot = self._slots.get(request_id)
        return (
            slot is not None and slot.in_flight is not None and slot.in_flight.orphaned
        )

    def in_flight_request_ids(self) -> set[str]:
        """The ids that must stay buffered because a batch is open.

        Returns:
            One drain signal's worth of blocked ids, as a new set.
        """
        return {
            request_id
            for request_id, slot in self._slots.items()
            if slot.in_flight is not None
        }

    def can_end_session(self, request_id: str) -> bool:
        """Whether manager-owned state permits session teardown.

        Args:
            request_id: The request id to query.

        Returns:
            True if the generation finished and no batch is outstanding.
            The caller additionally requires the policy to hold nothing
            buffered for the request.
        """
        slot = self._slots.get(request_id)
        return (
            slot is not None
            and slot.phase is RequestPhase.FINISHED
            and slot.in_flight is None
        )

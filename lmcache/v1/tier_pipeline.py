# SPDX-License-Identifier: Apache-2.0
"""Split metadata lookup from host GET and from HBM admission.

Off by default. An empty lookup response still means the tokens are already
in host memory. When ``split_metadata_lookup`` is on, the scheduler and the
worker each keep an ordered queue:

* ``kind="lookup"`` records the hit length and does not read the body.
* ``kind="reading"`` means host GET has started and carries no length.
* ``kind="preloaded"`` means the body is in host memory.

A miss does not start a GET. A prefix that does not fit in host stays queued.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Optional
import threading

STATUS_INIT = "INIT"
STATUS_READING = "READING"
STATUS_MISS = "MISS"

PHASE_ABSENT = "absent"
PHASE_PENDING = "pending"
PHASE_MISS = "miss"
PHASE_READY = "ready"


def admission_tokens(phase: str, preloaded: int) -> Optional[int]:
    """What ``get_num_new_matched_tokens`` may report. It does not start I/O.

    Pending and absent stay ``None`` so vLLM leaves the request in
    ``skipped_waiting``. A confirmed miss is ``0``. A positive length is
    returned only after the body is in host memory.
    """
    if phase in (PHASE_ABSENT, PHASE_PENDING):
        return None
    if phase == PHASE_MISS:
        return 0
    if phase == PHASE_READY:
        return max(0, int(preloaded))
    return None


def completion_kind(split_enabled: bool) -> str:
    """Legacy completion has an empty kind. The split path says ``preloaded``."""
    return "preloaded" if split_enabled else ""


@dataclass
class TierLoad:
    """One prefix that already has a metadata hit and may still need a GET."""

    num_chunks: int
    tiers: list[tuple[Any, list[Any], int, list[int]]]
    cum_chunk_lengths_total: list[int]
    tier_expected_chunks: list[int]
    keys_per_chunk: int


class TierWorkerQueue:
    """Worker-side ordered queue. The oldest hit is read first if it fits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, Optional[TierLoad]]] = {}
        self._order: list[str] = []

    def note_lookup(
        self, req_id: str, hit_tokens: int, job: Optional[TierLoad]
    ) -> tuple[bool, Optional[TierLoad]]:
        """Record a metadata result.

        A non-positive hit becomes a miss and returns no GET job. Replacing a
        queued job returns the previous job so the caller can unpin it. A GET
        that has already started is left alone.
        """
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current[0] == STATUS_READING:
                return False, None
            replaced = current[1] if current is not None else None
            if current is not None:
                self._order = [item for item in self._order if item != key]
            if hit_tokens <= 0 or job is None:
                self._entries[key] = (STATUS_MISS, None)
                return True, replaced
            self._entries[key] = (STATUS_INIT, job)
            self._order.append(key)
            return True, replaced

    def take_next_fitting(
        self, fits: Callable[[TierLoad], bool]
    ) -> Optional[tuple[str, TierLoad]]:
        """Claim the oldest queued hit whose whole prefix fits. Leave the rest."""
        with self._lock:
            for key in self._order:
                status, job = self._entries.get(key, (STATUS_MISS, None))
                if status != STATUS_INIT or job is None:
                    continue
                if not fits(job):
                    continue
                self._entries[key] = (STATUS_READING, job)
                return key, job
            return None

    def has_waiting(self) -> bool:
        with self._lock:
            return any(
                status == STATUS_INIT and job is not None
                for status, job in self._entries.values()
            )

    def drop_if_waiting(self, req_id: str) -> Optional[TierLoad]:
        """Drop a hit that has not started GET. A started GET is left running."""
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is None or current[0] != STATUS_INIT:
                return None
            self._entries.pop(key, None)
            self._order = [item for item in self._order if item != key]
            return current[1]

    def is_reading(self, req_id: str) -> bool:
        with self._lock:
            current = self._entries.get(str(req_id))
            return current is not None and current[0] == STATUS_READING

    def finish(self, req_id: str) -> None:
        """Forget a GET that has completed or been discarded."""
        key = str(req_id)
        with self._lock:
            current = self._entries.get(key)
            if current is None or current[0] != STATUS_READING:
                return
            self._entries.pop(key, None)
            self._order = [item for item in self._order if item != key]


class TierAdmission:
    """Scheduler-side phases, in the order lookups were offered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase: dict[str, str] = {}
        self._preloaded: dict[str, int] = {}
        self._order: list[str] = []

    def offer(self, req_id: str) -> bool:
        """Mark a new lookup pending. False if this id is already tracked."""
        key = str(req_id)
        with self._lock:
            phase = self._phase.get(key, PHASE_ABSENT)
            if phase != PHASE_ABSENT:
                return False
            self._phase[key] = PHASE_PENDING
            self._order.append(key)
            return True

    def note(self, req_id: str, kind: str, tokens: int) -> None:
        """Apply one worker ACK. A later ACK does not revive a miss or a hit."""
        key = str(req_id)
        with self._lock:
            phase = self._phase.get(key, PHASE_ABSENT)
            if phase in (PHASE_MISS, PHASE_READY):
                return
            if kind == "reading":
                return
            if kind == "lookup":
                if tokens <= 0:
                    self._phase[key] = PHASE_MISS
                return
            if kind in ("preloaded", ""):
                if tokens <= 0:
                    self._phase[key] = PHASE_MISS
                else:
                    self._phase[key] = PHASE_READY
                    self._preloaded[key] = int(tokens)

    def phase(self, req_id: str) -> str:
        with self._lock:
            return self._phase.get(str(req_id), PHASE_ABSENT)

    def preloaded_tokens(self, req_id: str) -> int:
        with self._lock:
            return int(self._preloaded.get(str(req_id), 0))

    def inflight(self) -> int:
        with self._lock:
            return sum(1 for phase in self._phase.values() if phase == PHASE_PENDING)

    def clear(self, req_id: str) -> None:
        key = str(req_id)
        with self._lock:
            self._phase.pop(key, None)
            self._preloaded.pop(key, None)
            self._order = [item for item in self._order if item != key]


def offer_waiting_lookups(
    scheduler: Any,
    client: Any,
    prepare_tokens: Callable[[Any], Optional[tuple[Any, Optional[dict]]]],
    window: int,
    preempt_seen: dict[str, int],
) -> None:
    """Offer one metadata lookup per waiting request, capped by ``window``.

    Does not pop vLLM's waiting queue and does not read free HBM. ``window``
    is ``max_num_seqs``: only in-flight lookups hold a slot.
    """
    admission = getattr(client, "tier_admission", None)
    submit = getattr(client, "submit_lookup", None)
    if admission is None or not callable(submit):
        return
    try:
        limit = max(1, int(window))
    except (TypeError, ValueError):
        limit = 1
    occupied = int(admission.inflight())
    for queue_name in ("waiting", "skipped_waiting"):
        waiting = getattr(scheduler, queue_name, None)
        if waiting is None:
            continue
        try:
            requests = list(waiting)
        except TypeError:
            heap = getattr(waiting, "_heap", None)
            requests = list(heap) if heap is not None else []
        for request in requests:
            if occupied >= limit:
                return
            req_id = str(getattr(request, "request_id", "") or "")
            if not req_id or req_id.startswith("mock_req"):
                continue
            preempt_n = int(getattr(request, "num_preemptions", 0) or 0)
            seen = int(preempt_seen.get(req_id, 0) or 0)
            phase = str(admission.phase(req_id))
            if preempt_n > seen and phase in (PHASE_READY, PHASE_MISS):
                admission.clear(req_id)
                phase = PHASE_ABSENT
            if preempt_n > seen:
                preempt_seen[req_id] = preempt_n
            if phase == PHASE_PENDING:
                continue
            if phase in (PHASE_READY, PHASE_MISS):
                continue
            prepared = prepare_tokens(request)
            if prepared is None:
                continue
            token_ids, request_configs = prepared
            # submit_lookup marks the id pending. A duplicate offer does not send.
            if submit(token_ids, lookup_id=req_id, request_configs=request_configs):
                occupied += 1

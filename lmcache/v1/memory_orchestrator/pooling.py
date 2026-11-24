# SPDX-License-Identifier: Apache-2.0
"""Assign provider-owned memory to existing private DAX hotplug backends."""

# Standard
from dataclasses import dataclass
from typing import Protocol
import enum
import threading

# Third Party
import httpx

# First Party
from lmcache.v1.distributed.api import Tier


class ShrinkResult(enum.Enum):
    """Whether a requested lease is still draining or has been returned."""

    DRAINING = "draining"
    RELEASED = "released"


class _AssignmentPhase(enum.Enum):
    UNCERTAIN = "uncertain"
    ATTACHED = "attached"
    DRAINING = "draining"
    REMOVED = "removed"


@dataclass(frozen=True)
class MemoryAllocation:
    """Exclusive provider lease and its worker-local DAX byte range."""

    allocation_id: str
    device_path: str
    offset_bytes: int
    size_bytes: int


class MemoryPoolProvider(Protocol):
    """Deployment adapter that owns durable resource leases.

    Acquisition is idempotent by (consumer_id, request_id), including uncertain
    failures; reuse with different parameters is rejected. Leases never expire
    or revoke themselves. Release is idempotent;
    released request IDs must remain retired and cannot acquire backing again.
    Providers must fence delayed requests before acknowledging release.
    """

    def acquire(
        self, consumer_id: str, request_id: str, size_bytes: int
    ) -> MemoryAllocation:
        """Provision at least size_bytes on consumer_id; return an exclusive range.

        Args:
            consumer_id: Opaque worker/host identity understood by the provider.
            request_id: Idempotency identity within this consumer. Reuse with
                different arguments or after release must be rejected.
            size_bytes: Positive minimum allocation size in bytes; rounding up
                to the provider's allocation granularity is allowed.

        Returns:
            An exclusive allocation ready to map on the specified worker. Its
            lease must remain valid until explicit release, including failures.

        Raises:
            Exception: Provider-specific provisioning or transport error. The
                outcome may be uncertain; an exact retry must recover the same
                lease rather than allocate another range.
        """
        ...

    def release(self, allocation_id: str) -> None:
        """Return a fully unmapped lease to the provider.

        Args:
            allocation_id: Identity returned by acquire, after all consumers
                have confirmed that the allocation is no longer mapped.

        Returns:
            None after release is complete. Exact retries must be idempotent;
            the original acquire request remains retired after release.

        Raises:
            Exception: Provider-specific release or transport error, including
                an uncertain outcome. Retry this allocation ID; do not reacquire.
        """
        ...


@dataclass
class _Assignment:
    """Fence the request even when no valid allocation descriptor was received."""

    requested_bytes: int
    allocation: MemoryAllocation | None = None
    phase: _AssignmentPhase = _AssignmentPhase.UNCERTAIN


class MemoryPoolController:
    """Serialize explicit grow/shrink operations for one MP DAX backend.

    Args:
        pool: Provider whose leases survive this controller's lifetime.
        client: Caller-owned HTTP client with base_url, authentication and timeout.
        consumer_id: Provider's identity for this worker/host.
        tier: Tier.L1 or Tier.L2 for the private DAX backend.
        adapter_index: Existing DAX L2 adapter index; zero for L1.

    One controller must exclusively manage its assigned ranges. State is local:
    after controller loss, reconcile leases and fence outstanding requests offline.
    Ambiguous backend outcomes prohibit further actions on that assignment.
    Initialization requires instance-fenced hotplug endpoints and can raise HTTP,
    transport or validation errors. No shared-region allocation state is involved.
    """

    def __init__(
        self,
        pool: MemoryPoolProvider,
        client: httpx.Client,
        consumer_id: str,
        tier: Tier,
        adapter_index: int = 0,
    ) -> None:
        if (
            not isinstance(consumer_id, str)
            or not consumer_id
            or tier not in (Tier.L1, Tier.L2)
        ):
            raise ValueError("consumer_id and a DAX tier (l1/l2) are required")
        if type(adapter_index) is not int or adapter_index < 0:
            raise ValueError("adapter_index must be a nonnegative integer")
        if tier == Tier.L1 and adapter_index:
            raise ValueError("L1 does not have an adapter index")
        self._pool, self._client = pool, client
        self._consumer_id, self._tier = consumer_id, Tier(tier)
        self._adapter_index = adapter_index
        self._instance = ""
        # shortcut: local assignments; add a journal for automatic crash recovery.
        self._assignments: dict[str, _Assignment] = {}
        self._lock = threading.Lock()
        status = self._request("GET", "status")
        if tier == Tier.L1:
            if not isinstance(status.get("arenas"), list):
                raise RuntimeError("invalid L1 arena status")
        else:
            adapters = status.get("adapters")
            if not isinstance(adapters, list) or adapter_index >= len(adapters):
                raise ValueError("DAX L2 adapter is unavailable")
            adapter = adapters[adapter_index]
            if (
                not isinstance(adapter, dict)
                or not isinstance(adapter.get("status"), dict)
                or adapter["status"].get("hotplug_enabled") is not True
            ):
                raise ValueError("DAX L2 hotplug must be enabled")

    def grow(self, request_id: str, size_bytes: int) -> MemoryAllocation:
        """Acquire and attach a range; repeat an attached request idempotently.

        Args:
            request_id: Nonempty caller retry identity, retired after release.
            size_bytes: Positive requested bytes; provider rounding is allowed.

        Returns:
            The assigned range after backend acknowledgement.

        Raises:
            ValueError: Invalid input/provider result or mismatched retry size.
            RuntimeError: A prior action is uncertain or removal already started.
            httpx.HTTPError: Backend failure; acquired backing remains held.
        """
        if (
            not isinstance(request_id, str)
            or not request_id
            or type(size_bytes) is not int
            or size_bytes <= 0
        ):
            raise ValueError("request_id and a positive integer size are required")
        with self._lock:
            previous = self._assignments.get(request_id)
            if previous is not None:
                if previous.requested_bytes != size_bytes:
                    raise ValueError("request_id was used with a different size")
                if previous.phase is not _AssignmentPhase.ATTACHED:
                    raise RuntimeError("assignment is uncertain or being removed")
                assert previous.allocation is not None
                return previous.allocation
            allocation = self._pool.acquire(
                self._consumer_id,
                f"{self._instance}/{self._tier.value}/{self._adapter_index}/{request_id}",
                size_bytes,
            )
            assignment = _Assignment(size_bytes)
            self._assignments[request_id] = assignment
            if (
                not isinstance(allocation, MemoryAllocation)
                or not isinstance(allocation.allocation_id, str)
                or not allocation.allocation_id
                or not isinstance(allocation.device_path, str)
                or not allocation.device_path
                or type(allocation.offset_bytes) is not int
                or allocation.offset_bytes < 0
                or type(allocation.size_bytes) is not int
                or allocation.size_bytes < size_bytes
                or any(
                    other.allocation is not None
                    and other.allocation.allocation_id == allocation.allocation_id
                    for other in self._assignments.values()
                )
            ):
                raise ValueError("provider returned an invalid memory allocation")
            assignment.allocation = allocation
            result = self._request(
                "POST",
                "add",
                self._payload(allocation) | {"size": allocation.size_bytes},
            )
            entry = result.get("added" if self._tier is Tier.L1 else "device")
            if not isinstance(entry, dict) or entry.get("state") != "active":
                raise RuntimeError("backend did not acknowledge an active mapping")
            self._check_range(entry, allocation)
            assignment.phase = _AssignmentPhase.ATTACHED
            return allocation

    def shrink(self, request_id: str) -> ShrinkResult:
        """Drain/unmap one assignment and only then release its provider lease.

        Args:
            request_id: Identity of an assignment created by grow.

        Returns:
            ShrinkResult.DRAINING while L1 still holds the mapping, or
            ShrinkResult.RELEASED after the provider confirms release.

        Raises:
            KeyError: The request is unknown or was already released.
            RuntimeError: The assignment or backend outcome is uncertain.
            httpx.HTTPError: A backend request failed. Only the explicit L2
                locked/borrowed-slots rejection allows another removal attempt.
            Exception: Provider-specific release failure. The detached
                assignment remains available for a release-only retry.
        """
        with self._lock:
            assignment = self._assignments[request_id]
            allocation = assignment.allocation
            phase = assignment.phase
            if phase is _AssignmentPhase.UNCERTAIN:
                raise RuntimeError(
                    "uncertain assignment requires offline reconciliation"
                )
            assert allocation is not None
            if phase is not _AssignmentPhase.REMOVED:
                assignment.phase = _AssignmentPhase.UNCERTAIN
                payload = self._payload(allocation)
                payload["mode"] = "drain" if self._tier is Tier.L1 else "evict"
                if self._tier is Tier.L2:
                    payload["force"] = False
                try:
                    result = self._request("POST", "remove", payload)
                except httpx.HTTPStatusError as exc:
                    # Reaping may finish between an acknowledged drain and its retry.
                    if not (
                        phase is _AssignmentPhase.DRAINING
                        and exc.response.status_code == 404
                    ):
                        if self._tier is Tier.L2 and exc.response.status_code == 409:
                            try:
                                error = exc.response.json()
                            except ValueError:
                                raise exc from None
                            if (
                                isinstance(error, dict)
                                and error.get("status") == "blocked"
                                and error.get("reason")
                                == "device has externally locked or borrowed slots"
                            ):
                                assignment.phase = _AssignmentPhase.ATTACHED
                        raise
                else:
                    if self._tier is Tier.L1:
                        removed = result.get("removed")
                        if not isinstance(removed, dict):
                            raise RuntimeError("missing removal acknowledgement")
                        arenas = removed.get("arenas")
                        if not isinstance(arenas, list) or len(arenas) != 1:
                            raise RuntimeError("invalid removal acknowledgement")
                        result = arenas[0]
                        self._check_range(result, allocation)
                        if result.get("state") == "draining":
                            assignment.phase = _AssignmentPhase.DRAINING
                            return ShrinkResult.DRAINING
                    if result.get("state") != "removed":
                        raise RuntimeError("backend has not confirmed unmapping")
                    if self._tier is Tier.L2 and (
                        result.get("operation") != "remove"
                        or result.get("device_path") != allocation.device_path
                    ):
                        raise RuntimeError("backend acknowledged a different removal")
                assignment.phase = _AssignmentPhase.REMOVED
            self._pool.release(allocation.allocation_id)
            del self._assignments[request_id]
            return ShrinkResult.RELEASED

    def _request(
        self, method: str, operation: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        response = self._client.request(
            method,
            f"/reconfigure/dax/{self._tier.value}/{operation}",
            json=payload,
            headers={"X-LMCache-Expected-Instance": self._instance}
            if self._instance
            else {},
        )
        # Authentication can reject the request before incarnation middleware.
        if response.status_code in (401, 403):
            response.raise_for_status()
        instance = response.headers.get("X-LMCache-Instance")
        if not instance or (self._instance and instance != self._instance):
            raise RuntimeError("worker instance changed or lacks hotplug fencing")
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("invalid hotplug response")
        self._instance = instance
        return result

    def _payload(self, allocation: MemoryAllocation) -> dict[str, object]:
        payload: dict[str, object] = {
            "device_path": allocation.device_path,
            "offset": allocation.offset_bytes,
        }
        if self._tier is Tier.L2:
            payload["adapter_index"] = self._adapter_index
        return payload

    def _check_range(self, entry: object, allocation: MemoryAllocation) -> None:
        offset, size = (
            ("offset_in_bytes", "size_in_bytes")
            if self._tier is Tier.L1
            else ("offset_bytes", "max_dax_size_bytes")
        )
        if not isinstance(entry, dict) or (
            entry.get("device_path"),
            entry.get(offset),
            entry.get(size),
        ) != (
            allocation.device_path,
            allocation.offset_bytes,
            allocation.size_bytes,
        ):
            raise RuntimeError("backend acknowledged a different memory range")

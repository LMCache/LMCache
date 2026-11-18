# SPDX-License-Identifier: Apache-2.0
"""Assign provider-owned memory to existing private DAX hotplug backends."""

# Standard
from dataclasses import dataclass
from typing import Literal, Protocol
import threading

# Third Party
import httpx


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

        Args are an opaque worker identity, retry identity and positive byte count.
        Raise the provider's error on failure; retries must recover the same lease.
        """
        ...

    def release(self, allocation_id: str) -> None:
        """Release a fully unmapped lease; raise on unavailable/uncertain results."""
        ...


@dataclass
class _Assignment:
    allocation: MemoryAllocation
    requested_bytes: int
    phase: Literal["uncertain", "attached", "draining", "removed"] = "uncertain"


class MemoryPoolController:
    """Serialize explicit grow/shrink operations for one MP DAX backend.

    Args:
        pool: Provider whose leases survive this controller's lifetime.
        client: Caller-owned HTTP client with base_url, authentication and timeout.
        consumer_id: Provider's identity for this worker/host.
        tier: Private DAX L1 or L2 backend.
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
        tier: Literal["l1", "l2"],
        adapter_index: int = 0,
    ) -> None:
        if (
            not isinstance(consumer_id, str)
            or not consumer_id
            or tier not in ("l1", "l2")
        ):
            raise ValueError("consumer_id and a DAX tier (l1/l2) are required")
        if type(adapter_index) is not int or adapter_index < 0:
            raise ValueError("adapter_index must be a nonnegative integer")
        if tier == "l1" and adapter_index:
            raise ValueError("L1 does not have an adapter index")
        self._pool, self._client = pool, client
        self._consumer_id, self._tier = consumer_id, tier
        self._adapter_index = adapter_index
        self._instance = ""
        # shortcut: local assignments; add a journal for automatic crash recovery.
        self._assignments: dict[str, _Assignment] = {}
        self._lock = threading.Lock()
        status = self._request("GET", "status")
        if tier == "l1":
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
                if previous.phase != "attached":
                    raise RuntimeError("assignment is uncertain or being removed")
                return previous.allocation
            allocation = self._pool.acquire(
                self._consumer_id,
                f"{self._instance}/{self._tier}/{self._adapter_index}/{request_id}",
                size_bytes,
            )
            assignment = _Assignment(allocation, size_bytes)
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
                    other is not assignment
                    and other.allocation.allocation_id == allocation.allocation_id
                    for other in self._assignments.values()
                )
            ):
                raise ValueError("provider returned an invalid memory allocation")
            result = self._request(
                "POST",
                "add",
                self._payload(allocation) | {"size": allocation.size_bytes},
            )
            entry = result.get("added" if self._tier == "l1" else "device")
            if not isinstance(entry, dict) or entry.get("state") != "active":
                raise RuntimeError("backend did not acknowledge an active mapping")
            self._check_range(entry, allocation)
            assignment.phase = "attached"
            return allocation

    def shrink(self, request_id: str) -> bool:
        """Drain/unmap one assignment and only then release its provider lease.

        Return False while L1 drains; True after release. Unknown/released IDs
        raise KeyError. Provider release failures retain the removed phase for
        retry. Backend transport/validation errors retain an uncertain assignment
        requiring offline reconciliation; a same-instance L2 409 can be retried.
        """
        with self._lock:
            assignment = self._assignments[request_id]
            allocation = assignment.allocation
            phase = assignment.phase
            if phase == "uncertain":
                raise RuntimeError(
                    "uncertain assignment requires offline reconciliation"
                )
            if phase != "removed":
                assignment.phase = "uncertain"
                payload = self._payload(allocation)
                payload["mode"] = "drain" if self._tier == "l1" else "evict"
                if self._tier == "l2":
                    payload["force"] = False
                try:
                    result = self._request("POST", "remove", payload)
                except httpx.HTTPStatusError as exc:
                    # Reaping may finish between an acknowledged drain and its retry.
                    if not (phase == "draining" and exc.response.status_code == 404):
                        if self._tier == "l2" and exc.response.status_code == 409:
                            assignment.phase = "attached"
                        raise
                else:
                    if self._tier == "l1":
                        removed = result.get("removed")
                        if not isinstance(removed, dict):
                            raise RuntimeError("missing removal acknowledgement")
                        arenas = removed.get("arenas")
                        if not isinstance(arenas, list) or len(arenas) != 1:
                            raise RuntimeError("invalid removal acknowledgement")
                        result = arenas[0]
                        self._check_range(result, allocation)
                        if result.get("state") == "draining":
                            assignment.phase = "draining"
                            return False
                    if result.get("state") != "removed":
                        raise RuntimeError("backend has not confirmed unmapping")
                    if self._tier == "l2" and (
                        result.get("operation") != "remove"
                        or result.get("device_path") != allocation.device_path
                    ):
                        raise RuntimeError("backend acknowledged a different removal")
                assignment.phase = "removed"
            self._pool.release(allocation.allocation_id)
            del self._assignments[request_id]
            return True

    def _request(
        self, method: str, operation: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        response = self._client.request(
            method,
            f"/reconfigure/dax/{self._tier}/{operation}",
            json=payload,
            headers={"X-LMCache-Expected-Instance": self._instance}
            if self._instance
            else {},
        )
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
        if self._tier == "l2":
            payload["adapter_index"] = self._adapter_index
        return payload

    def _check_range(self, entry: object, allocation: MemoryAllocation) -> None:
        offset, size = (
            ("offset_in_bytes", "size_in_bytes")
            if self._tier == "l1"
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

# SPDX-License-Identifier: Apache-2.0
"""Grants and leases held by one Device-DAX manager in an orchestrated region."""

# Standard
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from mmap import PAGESIZE
from typing import TypeVar
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig, SharedDevDaxConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.memory_manager.shared_devdax_region import (
    SharedDevDaxRegion,
)
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    DeleteStatus,
    ExtentHandle,
    OrchestratorError,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionFencedError,
    RequestRejectedError,
    TokenStatus,
    WireLayout,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient

logger = init_logger(__name__)
_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


@dataclass
class _ReadHold:
    view: TensorMemoryObj
    handle: ExtentHandle
    layout: WireLayout
    leases: list[bytes] = field(default_factory=list)


class OrchestratedObjectState:
    """Hold local views until their tagged writes or counted reads finish.

    The orchestrator owns allocation and publication. Never wait for an RPC
    under the state lock. A fencing reply permanently stops local admission.
    """

    def __init__(
        self, shared: SharedDevDaxConfig, memory: L1MemoryManagerConfig
    ) -> None:
        self._lock = threading.Lock()
        self._writes: dict[tuple[ObjectKey, str], tuple[bytes, TensorMemoryObj]] = {}
        self._reads: dict[ObjectKey, _ReadHold] = {}
        self._staging_bytes = 0
        self._reachable = True
        self._disabled_reason: str | None = None
        with ExitStack() as cleanup:
            self._client = MemoryOrchestratorClient(
                shared.orchestrator,
                shared.region_id,
                shared.client_id,
                rpc_timeout_s=shared.rpc_timeout_seconds,
            )
            cleanup.callback(self._client.close)
            contract = self._client.describe_region()
            if contract.reset_required:
                raise RuntimeError(
                    f"Shared region {shared.region_id!r} requires an offline reset"
                )
            if not 0 < contract.capacity_bytes <= memory.size_in_bytes:
                raise ValueError(
                    "Shared region capacity does not fit the configured L1"
                )
            self._alignment = contract.alignment_bytes
            if self._alignment <= 0 or (
                contract.visibility_mode == "software_fenced"
                and self._alignment % PAGESIZE
            ):
                raise ValueError("Invalid shared region extent alignment")
            self.capacity_bytes = contract.capacity_bytes
            self._visibility = Visibility(contract.visibility_mode)
            self.region = SharedDevDaxRegion(memory, contract)
            cleanup.callback(self.region.close)
            registered = self._client.register(
                layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
                mapped_bytes=self.capacity_bytes,
                visibility_mode=contract.visibility_mode,
            )
            if registered.region_epoch != contract.region_epoch:
                raise RegionFencedError(
                    "Shared region changed while it was being mapped"
                )
            cleanup.pop_all()

    def _request(
        self,
        name: str,
        call: Callable[[Sequence[_Item]], list[_Result]],
        entries: Sequence[_Item],
    ) -> list[_Result] | None:
        return self._rpc(name, lambda: call(entries)) if entries else []

    def _rpc(self, name: str, call: Callable[[], _Result]) -> _Result | None:
        with self._lock:
            if self._disabled_reason is not None:
                return None
        try:
            reply = call()
        except OrchestratorError as exc:
            if name == "ReserveWrite" and isinstance(exc, RequestRejectedError):
                raise  # A conflicting layout is not capacity pressure.
            with self._lock:
                if isinstance(exc, RegionFencedError):
                    self._disabled_reason = str(exc)
                    logger.error("Shared L1 fenced by %s: %s", name, exc)
                elif self._reachable:
                    logger.warning("Shared L1 %s failed: %s", name, exc)
                self._reachable = False
            return None
        with self._lock:
            self._reachable = True
        return reply

    def _view(
        self, handle: ExtentHandle | None, layout: MemoryLayoutDesc
    ) -> TensorMemoryObj:
        if handle is None or handle.generation != 1 or handle.length % self._alignment:
            raise ValueError("Invalid shared extent handle")
        return self.region.view(handle.offset, handle.length, layout)

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve extents for this writer; outages and full regions skip storage."""
        if len(keys) != len(is_temporary) or any(is_temporary):
            raise ValueError("Shared writes require one non-temporary flag per key")
        ret: dict[ObjectKey, L1OperationResult] = dict.fromkeys(
            keys, (L1Error.KEY_NOT_WRITABLE, None)
        )
        with self._lock:
            pending = [key for key in ret if (key, tag) not in self._writes]
        size = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        wire = WireLayout(
            tuple(map(tuple, layout_desc.shapes)),
            tuple(str(dtype).removeprefix("torch.") for dtype in layout_desc.dtypes),
        )
        grants = self._request(
            "ReserveWrite",
            self._client.reserve_write,
            [WriteRequest(key, size, wire) for key in pending],
        )
        rejected = []
        with self._lock:
            if grants is None or self._disabled_reason is not None:
                ret.update(dict.fromkeys(pending, (L1Error.OUT_OF_MEMORY, None)))
                return ret
            for key, grant in zip(pending, grants, strict=True):
                if grant.status is WriteStatus.OUT_OF_SPACE:
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                if grant.status is not WriteStatus.WRITE_GRANTED:
                    continue
                try:
                    if not grant.token:
                        raise ValueError("Missing write token")
                    view = self._view(grant.handle, layout_desc)
                except ValueError as exc:
                    logger.error("Invalid shared write grant: %s", exc)
                    if grant.token:
                        rejected.append(grant.token)
                    continue
                # A concurrent RPC may have installed a newer grant for this writer.
                if (key, tag) in self._writes:
                    rejected.append(grant.token)
                    continue
                self._writes[key, tag] = (grant.token, view)
                self._staging_bytes += size
                ret[key] = (L1Error.SUCCESS, view)
        self._request("AbortWrite", self._client.abort_write, rejected)
        return ret

    def finish_writes(
        self, keys: list[ObjectKey], tag: str, *, publish: bool
    ) -> dict[ObjectKey, L1Error]:
        """Commit or abort this writer's grants after copies drain."""
        ret = dict.fromkeys(keys, L1Error.KEY_NOT_EXIST)
        taken = []
        tokens = []
        ranges = []
        with self._lock:
            for key in ret:
                grant = self._writes.pop((key, tag), None)
                if grant is not None:
                    token, view = grant
                    size = view.get_size()
                    self._staging_bytes -= size
                    taken.append(key)
                    tokens.append(token)
                    ranges.append((view.data_ptr, size))
            if self._disabled_reason is not None:
                ret.update(dict.fromkeys(taken, L1Error.KEY_IN_WRONG_STATE))
                return ret
        if publish:
            try:
                self._visibility.publish(ranges)
            except BaseException:
                self._request("AbortWrite", self._client.abort_write, tokens)
                raise
            statuses = self._request("FinishWrite", self._client.finish_write, tokens)
        else:
            statuses = self._request("AbortWrite", self._client.abort_write, tokens)
        for key, status in zip(
            taken, statuses or [TokenStatus.STALE_TOKEN] * len(taken), strict=True
        ):
            ret[key] = (
                L1Error.SUCCESS
                if status is TokenStatus.OK
                else L1Error.KEY_IN_WRONG_STATE
            )
        return ret

    def reserve_read(
        self, keys: list[ObjectKey], count: int
    ) -> dict[ObjectKey, L1OperationResult]:
        """Acquire visibility before exposing newly leased views."""
        ret: dict[ObjectKey, L1OperationResult] = dict.fromkeys(
            keys, (L1Error.KEY_NOT_EXIST, None)
        )
        grants = self._request(
            "ReserveRead",
            self._client.reserve_read,
            [ReadRequest(key, count) for key in ret],
        )
        if grants is None:
            return ret
        rejected: list[bytes] = []
        accepted = []
        with ExitStack() as cleanup:
            cleanup.callback(
                self._request, "FinishRead", self._client.finish_read, rejected
            )
            with self._lock:
                if self._disabled_reason is not None:
                    return ret
                for key, grant in zip(ret, grants, strict=True):
                    if grant.status is not ReadStatus.READ_GRANTED:
                        continue
                    try:
                        hold = self._read_hold(key, grant, count)
                    except ValueError as exc:
                        logger.error("Invalid shared read grant: %s", exc)
                        rejected.extend(grant.leases)
                        continue
                    accepted.append((key, hold, grant.leases))
                try:
                    self._visibility.acquire(
                        [(h.view.data_ptr, h.view.get_size()) for _, h, _ in accepted]
                    )
                except BaseException:
                    for _, _, leases in accepted:
                        rejected.extend(leases)
                    raise
                for key, hold, leases in accepted:
                    hold.leases.extend(leases)
                    self._reads[key] = hold
                    ret[key] = (L1Error.SUCCESS, hold.view)
        return ret

    def _read_hold(
        self, key: ObjectKey, grant: ReadGrantResult, count: int
    ) -> _ReadHold:
        wire, handle = grant.layout, grant.handle
        if (
            wire is None
            or handle is None
            or len(grant.leases) != count
            or not all(grant.leases)
        ):
            raise ValueError("Incomplete shared read grant")
        hold = self._reads.get(key)
        if hold is not None:
            if (
                hold.handle != handle
                or hold.layout != wire
                or hold.view.get_size() != grant.payload_bytes
            ):
                raise ValueError("A leased object changed extent or layout")
            return hold
        dtypes = [getattr(torch, name, None) for name in wire.dtypes]
        if not wire.shapes or not all(
            isinstance(dtype, torch.dtype) for dtype in dtypes
        ):
            raise ValueError("Invalid shared read layout")
        if any(dim < 0 for shape in wire.shapes for dim in shape):
            raise ValueError("Invalid shared read shape")
        layout = MemoryLayoutDesc([torch.Size(s) for s in wire.shapes], dtypes)
        view = self._view(handle, layout)
        if view.get_size() != grant.payload_bytes or grant.payload_bytes <= 0:
            raise ValueError("Read payload does not match its layout")
        return _ReadHold(view, handle, wire)

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return only locally leased views from an unfenced region."""
        with self._lock:
            return {
                key: (L1Error.SUCCESS, self._reads[key].view)
                if self._disabled_reason is None and key in self._reads
                else (L1Error.KEY_NOT_EXIST, None)
                for key in keys
            }

    def finish_read(
        self, keys: list[ObjectKey], count: int
    ) -> dict[ObjectKey, L1Error]:
        """Drop up to count holds per key and release their tokens."""
        ret = dict.fromkeys(keys, L1Error.KEY_NOT_EXIST)
        owners = []
        tokens = []
        with self._lock:
            for key in ret:
                hold = self._reads.get(key)
                if hold is not None:
                    batch = hold.leases[-count:]
                    del hold.leases[-count:]
                    if not hold.leases:
                        del self._reads[key]
                    tokens.extend(batch)
                    owners.extend([key] * len(batch))
                    ret[key] = L1Error.SUCCESS
        statuses = self._request("FinishRead", self._client.finish_read, tokens)
        for key, status in zip(
            owners, statuses or [TokenStatus.STALE_TOKEN] * len(tokens), strict=True
        ):
            if status is not TokenStatus.OK:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        return ret

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Hide committed objects globally while preserving active read leases."""
        statuses = self._request("DeleteObjects", self._client.delete_objects, keys)
        if statuses is None:
            return dict.fromkeys(keys, L1Error.KEY_IS_LOCKED)
        errors = {
            DeleteStatus.DELETED: L1Error.SUCCESS,
            DeleteStatus.MISS: L1Error.KEY_NOT_EXIST,
            DeleteStatus.BUSY_WRITING: L1Error.KEY_IS_LOCKED,
        }
        return {key: errors[status] for key, status in zip(keys, statuses, strict=True)}

    def delete_published(self) -> int | None:
        """Hide all committed objects written under this client ID."""
        return self._rpc("DeletePublished", self._client.delete_published)

    def get_staging_memory_usage(self) -> int:
        """Return this client's outstanding write payload bytes."""
        with self._lock:
            return self._staging_bytes

    def is_healthy(self) -> bool:
        """Return whether the last RPC succeeded and the client can serve."""
        with self._lock:
            return self._reachable and self._disabled_reason is None

    def report_status(self) -> dict:
        """Report local holds, staging bytes, health and region capacity."""
        with self._lock:
            writes, reads = len(self._writes), len(self._reads)
            staging, reason = self._staging_bytes, self._disabled_reason
            state = "available" if self._reachable else "unavailable"
            if reason is not None:
                state = "disabled"
        return {
            "is_healthy": state == "available",
            "total_object_count": writes + reads,
            "write_locked_count": writes,
            "read_locked_count": reads,
            "temporary_count": 0,
            "staging_object_count": writes,
            "staging_bytes": staging,
            "memory_used_bytes": staging,
            "memory_total_bytes": self.capacity_bytes,
            "memory_configured_bytes": self.capacity_bytes,
            "backend": L1BackendType.DEVDAX.value,
            "memory_usage_ratio": staging / self.capacity_bytes,
            "write_ttl_seconds": None,
            "read_ttl_seconds": None,
            "shared": {"state": state, "disabled_reason": reason},
        }

    def close(self) -> None:
        """Retire the client after copies drain; retry unmapping after views drop."""
        with self._lock:
            self._disabled_reason = "closed"
            while self._writes:
                self._writes.popitem()[1][1].invalidate()
            self._reads.clear()
            self._staging_bytes = 0
        self._client.close()
        self.region.close()

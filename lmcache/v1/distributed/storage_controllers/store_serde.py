# SPDX-License-Identifier: Apache-2.0
"""Serde helpers for ``StoreController``.

Extracted to keep ``store_controller.py`` focused on the state machine.
Helpers take explicit L1Manager / SerdeProcessor / primitive args — no
hidden controller state — and return data the controller then applies
to its own bookkeeping (``_in_flight_requests``, counters, etc.). Using
primitive args avoids a circular import with ``store_controller.py``.
"""

# Standard
from dataclasses import dataclass
import enum

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.serde import (
    SerdeProcessor,
    SerdeTaskId,
    make_temp_key,
    serialized_layout_desc,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


@dataclass
class SerializeReservation:
    """Data needed by ``StoreController`` to build an ``InFlightStoreRequest``
    after a successful serialize submission."""

    read_locked_keys: list[ObjectKey]
    """Original keys that hold L1 read locks (subset of inputs that got temps)."""

    temp_keys: list[ObjectKey]
    """Temp buffer keys (write-locked)."""

    temp_objs: list[MemoryObj]
    """Temp MemoryObjs, same order as ``temp_keys``."""

    serde_task_id: SerdeTaskId
    """ID returned by ``serde.submit_serialize``."""


def reserve_and_submit_serialize(
    l1_mgr: L1Manager,
    serde: SerdeProcessor,
    adapter_index: int,
    orig_keys: list[ObjectKey],
    orig_objs: list[MemoryObj],
) -> SerializeReservation | None:
    """Allocate temp buffers and submit a serialize task.

    ``orig_keys`` must already hold L1 read locks (from the controller's
    prior ``reserve_read``). This helper owns all subsequent lock transitions:

    - Keys whose temp alloc failed have their original read lock released.
    - On ``submit_serialize`` failure, all read locks (on originals) and
      write locks (on temps, plus deletes) are released.

    Returns ``None`` if no keys could be serialized (all temp allocs
    failed, or submit raised). Otherwise returns the reservation the
    controller uses to build an ``InFlightStoreRequest``.
    """
    temp_keys = [make_temp_key(key) for key in orig_keys]
    ser_layout = serialized_layout_desc(
        MemoryLayoutDesc(
            shapes=orig_objs[0].get_shapes(),
            dtypes=orig_objs[0].get_dtypes(),
        ),
        serde,
    )
    temp_write_results = l1_mgr.reserve_write(
        keys=temp_keys,
        is_temporary=[True] * len(temp_keys),
        layout_desc=ser_layout,
        mode="new",
    )

    final_read_keys: list[ObjectKey] = []
    final_read_objs: list[MemoryObj] = []
    final_temp_keys: list[ObjectKey] = []
    final_temp_objs: list[MemoryObj] = []
    failed_orig: set[ObjectKey] = set()

    for orig_key, orig_obj, temp_key in zip(
        orig_keys, orig_objs, temp_keys, strict=True
    ):
        temp_result = temp_write_results.get(temp_key)
        if temp_result is None or temp_result[0] != L1Error.SUCCESS:
            failed_orig.add(orig_key)
            continue
        final_read_keys.append(orig_key)
        final_read_objs.append(orig_obj)
        final_temp_keys.append(temp_key)
        final_temp_objs.append(temp_result[1])

    if failed_orig:
        l1_mgr.finish_read(list(failed_orig))

    # Failed temp-alloc keys were never added to L1's object store, so
    # there's nothing to finish_write / delete for them.

    if not final_read_keys:
        return None

    try:
        serde_task_id = serde.submit_serialize(final_read_objs, final_temp_objs)
    except Exception:
        logger.exception(
            "Failed to submit serialize task for adapter %d",
            adapter_index,
        )
        l1_mgr.finish_read(final_read_keys)
        l1_mgr.finish_write(final_temp_keys)
        l1_mgr.delete(final_temp_keys)
        return None

    return SerializeReservation(
        read_locked_keys=final_read_keys,
        temp_keys=final_temp_keys,
        temp_objs=final_temp_objs,
        serde_task_id=serde_task_id,
    )


class SerializeOutcome(enum.Enum):
    """Result of polling a pending serialize task."""

    PENDING = enum.auto()
    """Result not yet available; caller does nothing."""

    FAILED = enum.auto()
    """Serialize failed; all locks already released — caller should
    drop the request from tracking."""

    READY = enum.auto()
    """Success; original read locks released, temp buffers transitioned
    write-locked → read-locked. Caller should move the request to
    ``STORE`` phase and submit the L2 store task with the temp objs."""


def advance_serialize(
    l1_mgr: L1Manager,
    serde: SerdeProcessor,
    serde_task_id: SerdeTaskId,
    read_locked_keys: list[ObjectKey],
    temp_keys: list[ObjectKey],
) -> SerializeOutcome:
    """Poll the serde result and apply the L1 lock transitions it triggers.

    See :class:`SerializeOutcome` for what each return value implies for
    the caller.
    """
    result = serde.query_serialize_result(serde_task_id)
    if result is None:
        return SerializeOutcome.PENDING

    l1_mgr.finish_read(read_locked_keys)

    if result:
        # Transition temp buffers write-locked → read-locked so L2 can
        # safely read them during the subsequent store.
        l1_mgr.finish_write_and_reserve_read(temp_keys)
        return SerializeOutcome.READY

    if temp_keys:
        l1_mgr.finish_write(temp_keys)
        l1_mgr.delete(temp_keys)
    return SerializeOutcome.FAILED


def release_serialize_locks(
    l1_mgr: L1Manager,
    read_locked_keys: list[ObjectKey],
    temp_keys: list[ObjectKey],
) -> None:
    """Shutdown cleanup: release all locks held by a SERIALIZE-phase request.

    Reads on originals plus writes on temps (which get deleted).
    """
    l1_mgr.finish_read(read_locked_keys)
    if temp_keys:
        l1_mgr.finish_write(temp_keys)
        l1_mgr.delete(temp_keys)

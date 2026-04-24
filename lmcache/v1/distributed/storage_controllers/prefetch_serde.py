# SPDX-License-Identifier: Apache-2.0
"""Serde helpers for ``PrefetchController``.

Extracted to keep ``prefetch_controller.py`` focused on the state
machine. Helpers mutate the passed-in request in place (fields like
``temp_reserved_*``, ``pending_deserialize_tasks``) so the controller's
per-phase dispatch remains a short linear read.

The request type is referenced via ``TYPE_CHECKING`` to avoid a
circular import with ``prefetch_controller.py``; at runtime these
helpers duck-type the request's fields.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.serde import (
    SerdeProcessor,
    make_temp_key,
    serialized_layout_desc,
)
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
        InFlightPrefetchRequest,
    )

logger = init_logger(__name__)


@dataclass
class AdapterTempReservation:
    """Per-adapter temp buffer reservation result.

    ``successful`` carries ``(orig_key, temp_key, temp_obj)`` tuples
    whose L1 ``reserve_write`` returned success. ``failed_orig_keys``
    is the original keys whose temp alloc failed — the caller must
    release their real KV buffers and prune them from the load plan.
    """

    successful: list[tuple[ObjectKey, ObjectKey, MemoryObj]]
    failed_orig_keys: list[ObjectKey]


def _reserve_adapter_temps(
    l1_mgr: L1Manager,
    serde: SerdeProcessor,
    kv_layout: MemoryLayoutDesc,
    orig_keys: list[ObjectKey],
) -> AdapterTempReservation:
    """Reserve temp byte buffers for one adapter's share of keys.

    Pure reservation — does not release anything. Caller handles the
    consequences of ``failed_orig_keys`` (typically: release real
    buffers, prune the load plan).
    """
    temp_layout = serialized_layout_desc(kv_layout, serde)
    temp_keys = [make_temp_key(k) for k in orig_keys]
    temp_results = l1_mgr.reserve_write(
        keys=temp_keys,
        is_temporary=[True] * len(temp_keys),
        layout_desc=temp_layout,
        mode="new",
    )
    successful: list[tuple[ObjectKey, ObjectKey, MemoryObj]] = []
    failed: list[ObjectKey] = []
    for orig_key, temp_key in zip(orig_keys, temp_keys, strict=True):
        result = temp_results.get(temp_key)
        if result is not None and result[0] == L1Error.SUCCESS:
            successful.append((orig_key, temp_key, result[1]))
        else:
            failed.append(orig_key)
    return AdapterTempReservation(successful=successful, failed_orig_keys=failed)


def apply_adapter_temp_reservations(
    l1_mgr: L1Manager,
    serde: SerdeProcessor,
    request: InFlightPrefetchRequest,
    adapter_index: int,
    plan_bitmap: Bitmap,
    reserved_key_set: set[ObjectKey],
) -> None:
    """Reserve temps for one adapter's keys and wire them into ``request``.

    Mutates ``request.temp_reserved_keys_for_serde`` (appends),
    ``request.temp_reserved_objs_for_serde`` (adds),
    ``request.original_to_temp_key`` (adds),
    ``request.write_reserved_keys`` / ``_objs`` (prunes failed keys),
    and ``reserved_key_set`` (removes failed keys in place).
    """
    adapter_keys = [
        k for k in plan_bitmap.gather(request.keys) if k in reserved_key_set
    ]
    if not adapter_keys:
        return

    reservation = _reserve_adapter_temps(
        l1_mgr, serde, request.layout_desc, adapter_keys
    )

    for orig_key, temp_key, temp_obj in reservation.successful:
        request.temp_reserved_keys_for_serde.append(temp_key)
        request.temp_reserved_objs_for_serde[temp_key] = temp_obj
        request.original_to_temp_key[orig_key] = temp_key

    if reservation.failed_orig_keys:
        l1_mgr.finish_write(reservation.failed_orig_keys)
        l1_mgr.delete(reservation.failed_orig_keys)
        failed_set = set(reservation.failed_orig_keys)
        request.write_reserved_keys = [
            k for k in request.write_reserved_keys if k not in failed_set
        ]
        request.write_reserved_objs = {
            k: v
            for k, v in request.write_reserved_objs.items()
            if k not in failed_set
        }
        reserved_key_set -= failed_set


def get_load_buffer(
    request: InFlightPrefetchRequest, key: ObjectKey
) -> MemoryObj:
    """Return the MemoryObj that L2 should load into for ``key``.

    Serde-enabled adapters route through a temp byte buffer; otherwise
    L2 loads directly into the real KV buffer.
    """
    temp_key = request.original_to_temp_key.get(key)
    if temp_key is not None:
        return request.temp_reserved_objs_for_serde[temp_key]
    return request.write_reserved_objs[key]


def release_adapter_temp_buffers(
    l1_mgr: L1Manager,
    request: InFlightPrefetchRequest,
    adapter_index: int,
) -> None:
    """Release temp byte buffers belonging to a specific adapter.

    Called after deserialization completes (success or failure) for an
    adapter, or when all L2 loads failed for a serde adapter and no
    deserialize was submitted.
    """
    adapter_temp_keys: list[ObjectKey] = []
    for orig_key in request.load_plan[adapter_index].gather(request.keys):
        temp_key = request.original_to_temp_key.get(orig_key)
        if temp_key is not None and temp_key in request.temp_reserved_objs_for_serde:
            adapter_temp_keys.append(temp_key)

    if adapter_temp_keys:
        l1_mgr.finish_write(adapter_temp_keys)
        l1_mgr.delete(adapter_temp_keys)
        temp_set = set(adapter_temp_keys)
        request.temp_reserved_keys_for_serde = [
            k for k in request.temp_reserved_keys_for_serde if k not in temp_set
        ]
        for tk in adapter_temp_keys:
            request.temp_reserved_objs_for_serde.pop(tk, None)


def submit_deserialize_for_adapter(
    l1_mgr: L1Manager,
    serde: SerdeProcessor,
    request: InFlightPrefetchRequest,
    adapter_index: int,
    load_result: Bitmap,
) -> None:
    """Submit async deserialize for an adapter's successfully-loaded keys.

    Only L2-loaded keys (per ``load_result`` bitmap) are deserialized;
    keys L2 failed to load have invalid temp buffers and are skipped.

    If no keys loaded OR ``submit_deserialize`` raises, releases this
    adapter's temp buffers and zeros its ``load_results`` bitmap so the
    finalize pass treats those keys as failed. Otherwise registers the
    task in ``request.pending_deserialize_tasks``.
    """
    plan_bitmap = request.load_plan[adapter_index]
    plan_indices = plan_bitmap.get_indices_list()

    src_objs: list[MemoryObj] = []
    dst_objs: list[MemoryObj] = []
    for local_i, global_i in enumerate(plan_indices):
        if not load_result.test(local_i):
            continue
        orig_key = request.keys[global_i]
        temp_key = request.original_to_temp_key.get(orig_key)
        if temp_key is None or orig_key not in request.write_reserved_objs:
            continue
        src_objs.append(request.temp_reserved_objs_for_serde[temp_key])
        dst_objs.append(request.write_reserved_objs[orig_key])

    if not src_objs:
        release_adapter_temp_buffers(l1_mgr, request, adapter_index)
        return

    logger.debug(
        "Prefetch request %d: submitting deserialize for adapter %d "
        "(%d objects loaded from L2).",
        request.request_id,
        adapter_index,
        len(src_objs),
    )
    try:
        task_id = serde.submit_deserialize(src_objs, dst_objs)
    except Exception:
        logger.exception(
            "Prefetch request %d: failed to submit deserialize for adapter %d",
            request.request_id,
            adapter_index,
        )
        release_adapter_temp_buffers(l1_mgr, request, adapter_index)
        # Zero the adapter's load result so _finalize_load treats these
        # keys as failed — the real KV buffers are uninitialized since
        # the deserialize that would have populated them never ran.
        request.load_results[adapter_index] = Bitmap(plan_bitmap.popcount())
        return
    request.pending_deserialize_tasks[adapter_index] = task_id


def poll_deserialize_results(
    l1_mgr: L1Manager,
    serde_processors: list[SerdeProcessor | None],
    request: InFlightPrefetchRequest,
    signaled_adapters: set[int],
) -> None:
    """Query pending deserialize results from signaled adapters.

    Releases each adapter's temp buffers on completion. On failure,
    zeros the adapter's ``load_results`` bitmap so its keys become
    "failed" in ``_finalize_load``.
    """
    for adapter_idx in list(request.pending_deserialize_tasks):
        if adapter_idx not in signaled_adapters:
            continue
        serde = serde_processors[adapter_idx]
        if serde is None:
            raise RuntimeError(
                f"pending deserialize task for adapter {adapter_idx} "
                f"but no serde processor"
            )
        task_id = request.pending_deserialize_tasks[adapter_idx]
        deserialize_ok = serde.query_deserialize_result(task_id)
        if deserialize_ok is None:
            continue
        del request.pending_deserialize_tasks[adapter_idx]

        if deserialize_ok:
            logger.debug(
                "Prefetch request %d: deserialize completed for "
                "adapter %d — data ready in L1 KV buffers.",
                request.request_id,
                adapter_idx,
            )
        release_adapter_temp_buffers(l1_mgr, request, adapter_idx)

        if not deserialize_ok:
            logger.warning(
                "Prefetch request %d: deserialize failed for adapter %d",
                request.request_id,
                adapter_idx,
            )
            plan_bitmap = request.load_plan.get(adapter_idx)
            if plan_bitmap is not None:
                request.load_results[adapter_idx] = Bitmap(plan_bitmap.popcount())


def release_all_temp_buffers(
    l1_mgr: L1Manager, request: InFlightPrefetchRequest
) -> None:
    """Shutdown cleanup: release every temp buffer held by ``request``."""
    if request.temp_reserved_keys_for_serde:
        l1_mgr.finish_write(request.temp_reserved_keys_for_serde)
        l1_mgr.delete(request.temp_reserved_keys_for_serde)

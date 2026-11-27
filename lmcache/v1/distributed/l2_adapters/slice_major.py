# SPDX-License-Identifier: Apache-2.0
"""Slice-major storage for native connector L2 adapters.

Each L1 object is stored as ``num_slices`` remote keys,
``<key>/s<i>of<n>``, each holding one contiguous byte slice of the object.
In a layer-major object a slice is a run of layers, so a load can hand the
objects to their readers once the first slice lands and let the readers
follow the later slices through a :class:`FillTracker`, instead of waiting
for the whole objects.

:class:`SlicedClient` wraps a native connector client and keeps its
interface, so :class:`NativeConnectorL2Adapter` runs on it unchanged: each
batch call is fanned out to the slice keys and its completion folded back
to one result per object. :class:`SliceMajorL2Adapter` adds the pipelined
load on top.

Slices are split evenly by bytes; pick ``num_slices`` dividing the number
of layers so slice boundaries fall on layer boundaries. All writers and
readers of a store must use the same ``num_slices``: the slice count is
part of the key, so a mismatch reads as a miss, never as wrong data.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import Any
import threading

# First Party
from lmcache.v1.common.logging import init_logger
from lmcache.v1.distributed.api import MemoryTags, ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _obj_to_memoryview,
    _object_key_to_string,
)
from lmcache.v1.distributed.pending_fill import FillTracker
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# A native completion: (future_id, ok, error, per-key results or None).
_Completion = tuple[int, bool, str, "list[bool] | None"]


def _slice_bounds(size: int, num_slices: int) -> list[tuple[int, int]]:
    """Split ``size`` bytes into ``num_slices`` contiguous, non-empty ranges."""
    if size < num_slices:
        raise ValueError(f"cannot split {size} bytes into {num_slices} slices")
    return [
        (i * size // num_slices, (i + 1) * size // num_slices)
        for i in range(num_slices)
    ]


def _fold(
    results: list[bool] | None, num_slices: int, combine: Callable[..., bool]
) -> list[bool] | None:
    """Fold object-major per-slice results to one result per object."""
    if results is None:
        return None
    return [
        combine(results[i : i + num_slices]) for i in range(0, len(results), num_slices)
    ]


class _SlicedLoad:
    """State of one load fanned out to one native get per slice index."""

    def __init__(
        self,
        keys: list[str],
        bufs: list[memoryview],
        num_slices: int,
        fill: FillTracker | None,
    ) -> None:
        bounds = [_slice_bounds(len(buf), num_slices) for buf in bufs]
        self.slice_keys = [
            [f"{key}/s{g}of{num_slices}" for key in keys] for g in range(num_slices)
        ]
        self.slice_bufs = [
            [buf[b[g][0] : b[g][1]] for buf, b in zip(bufs, bounds, strict=True)]
            for g in range(num_slices)
        ]
        # Bytes final in every object once slices [0, g] landed.
        self.slice_ends = [min(b[g][1] for b in bounds) for g in range(num_slices)]
        self.fill = fill
        self.head_fid = -1
        self.next_slice = 0
        self.inflight = 0
        self.landed = [False] * num_slices
        self.prefix = 0
        # Per object: every slice landed so far succeeded.
        self.ok = [True] * len(keys)
        # Per object: reported loaded; set once slice 0 lands (pipelined).
        self.reported: list[bool] | None = None
        self.error = ""

    def failed(self) -> bool:
        """Whether an object reported loaded missed a later slice."""
        if self.reported is None:
            return False
        return any(r and not ok for r, ok in zip(self.reported, self.ok, strict=True))

    def done(self) -> bool:
        """Whether the slices still to come can no longer change the result."""
        if self.reported is None:
            return not any(self.ok)
        return self.failed() or not any(
            r and ok for r, ok in zip(self.reported, self.ok, strict=True)
        )


class SlicedClient:
    """Native connector client facade that stores objects as slices.

    Keeps the native client interface used by
    :class:`NativeConnectorL2Adapter`. Set, exists and delete fan out to one
    native batch over all slice keys. A get fans out to one native batch per
    slice index, at most ``max_inflight_slices`` at a time, slice 0 first.

    Args:
        client: The native connector client.
        num_slices: Slices per object (> 1).
        max_inflight_slices: Slice batches of one get in flight at once.
    """

    def __init__(self, client: Any, num_slices: int, max_inflight_slices: int) -> None:
        if num_slices < 2:
            raise ValueError(f"num_slices must be > 1, got {num_slices}")
        if max_inflight_slices < 1:
            raise ValueError(
                f"max_inflight_slices must be > 0, got {max_inflight_slices}"
            )
        self._client = client
        self._num_slices = num_slices
        self._max_inflight = max_inflight_slices
        self._lock = threading.Lock()
        # Native future id -> how to fold its per-slice results.
        self._folds: dict[int, Callable[..., bool]] = {}
        # Native future id of a slice get -> (its load, slice index).
        self._slice_gets: dict[int, tuple[_SlicedLoad, int]] = {}
        if not callable(getattr(client, "submit_batch_delete", None)):
            self.submit_batch_delete = None  # type: ignore[assignment]

    @property
    def num_slices(self) -> int:
        return self._num_slices

    def event_fd(self) -> int:
        return int(self._client.event_fd())

    def close(self) -> None:
        self._client.close()

    def _keys(self, keys: list[str]) -> list[str]:
        n = self._num_slices
        return [f"{key}/s{g}of{n}" for key in keys for g in range(n)]

    def _submit_folded(
        self, submit: Callable[[], Any], combine: Callable[..., bool]
    ) -> int:
        with self._lock:
            fid = int(submit())
            self._folds[fid] = combine
        return fid

    def submit_batch_set(self, keys: list[str], bufs: list[memoryview]) -> int:
        slice_bufs = [
            buf[start:end]
            for buf in bufs
            for start, end in _slice_bounds(len(buf), self._num_slices)
        ]
        slice_keys = self._keys(keys)
        return self._submit_folded(
            lambda: self._client.submit_batch_set(slice_keys, slice_bufs), all
        )

    def submit_batch_exists(self, keys: list[str]) -> int:
        slice_keys = self._keys(keys)
        return self._submit_folded(
            lambda: self._client.submit_batch_exists(slice_keys), all
        )

    def submit_batch_delete(self, keys: list[str]) -> int:
        slice_keys = self._keys(keys)
        # Reported deleted once any slice is gone: the object is unreadable.
        return self._submit_folded(
            lambda: self._client.submit_batch_delete(slice_keys), any
        )

    def submit_batch_get(
        self,
        keys: list[str],
        bufs: list[memoryview],
        fill: FillTracker | None = None,
    ) -> int:
        """Load ``keys`` into ``bufs`` slice by slice.

        Without ``fill`` the returned future completes once every slice
        landed. With ``fill`` it completes when slice 0 lands; ``fill`` then
        advances as later slices land in order, and settles once every
        slice batch finished, successfully only if every object reported
        loaded got all its slices.
        """
        load = _SlicedLoad(keys, bufs, self._num_slices, fill)
        with self._lock:
            # Slice 0 failing to submit fails the call; a later slice fails
            # the load, which still completes through slice 0.
            load.head_fid = self._submit_next(load)
            self._submit_window(load)
        return load.head_fid

    def _submit_window(self, load: _SlicedLoad) -> None:
        """Keep the load's slice window full. Must be called under _lock."""
        while load.next_slice < self._num_slices and (
            load.inflight < self._max_inflight
        ):
            try:
                self._submit_next(load)
            except Exception:
                logger.exception("Slice-major load: submitting a slice failed")
                load.ok = [False] * len(load.ok)
                load.next_slice = self._num_slices

    def _submit_next(self, load: _SlicedLoad) -> int:
        """Submit the load's next slice batch. Must be called under _lock."""
        g = load.next_slice
        fid = int(self._client.submit_batch_get(load.slice_keys[g], load.slice_bufs[g]))
        load.next_slice += 1
        load.inflight += 1
        self._slice_gets[fid] = (load, g)
        return fid

    def drain_completions(self) -> list[_Completion]:
        out: list[_Completion] = []
        progress: list[tuple[FillTracker, int]] = []
        settles: list[tuple[FillTracker, bool]] = []
        for future_id, ok, error, results in self._client.drain_completions():
            fid = int(future_id)
            with self._lock:
                combine = self._folds.pop(fid, None)
                if combine is not None:
                    out.append(
                        (fid, ok, error, _fold(results, self._num_slices, combine))
                    )
                    continue
                entry = self._slice_gets.pop(fid, None)
                if entry is None:
                    out.append((fid, ok, error, results))
                    continue
                load, g = entry
                self._on_slice(load, g, ok, error, results, out, progress, settles)
        # Outside the lock: settle callbacks may take other locks.
        for fill, end in progress:
            fill.advance(end)
        for fill, success in settles:
            fill.settle(success)
        return out

    def _on_slice(
        self,
        load: _SlicedLoad,
        g: int,
        ok: bool,
        error: str,
        results: list[bool] | None,
        out: list[_Completion],
        progress: list[tuple[FillTracker, int]],
        settles: list[tuple[FillTracker, bool]],
    ) -> None:
        """Account one landed slice batch. Must be called under _lock."""
        load.inflight -= 1
        load.landed[g] = True
        if error:
            load.error = error
        landed = results if results is not None else [ok] * len(load.ok)
        for i, hit in enumerate(landed):
            if not hit:
                load.ok[i] = False

        if g == 0 and load.fill is not None:
            load.reported = list(load.ok)
            out.append((load.head_fid, True, load.error, list(load.reported)))

        if load.done():
            # Wait only for the slice batches already in flight.
            load.next_slice = self._num_slices
        self._submit_window(load)

        if load.fill is not None and not load.failed():
            prefix = load.prefix
            while prefix < self._num_slices and load.landed[prefix]:
                prefix += 1
            if prefix > load.prefix:
                load.prefix = prefix
                progress.append((load.fill, load.slice_ends[prefix - 1]))

        if load.inflight > 0 or load.next_slice < self._num_slices:
            return
        if load.fill is None:
            out.append((load.head_fid, True, load.error, list(load.ok)))
        else:
            settles.append((load.fill, not load.failed()))


class SliceMajorL2Adapter(NativeConnectorL2Adapter):
    """Native connector L2 adapter over a :class:`SlicedClient`.

    With ``pipelined_load`` a load is reported once slice 0 of its objects
    landed; :meth:`take_load_fill` hands out the tracker of the rest.

    Args:
        sliced_client: The slicing client facade.
        memory_tags: Memory tags of the connector's storage medium.
        pipelined_load: Report loads before their later slices land.
        **kwargs: Forwarded to :class:`NativeConnectorL2Adapter`.
    """

    def __init__(
        self,
        sliced_client: SlicedClient,
        memory_tags: MemoryTags,
        pipelined_load: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(sliced_client, memory_tags, **kwargs)
        self._sliced_client = sliced_client
        self._pipelined_load = pipelined_load
        self._load_fills: dict[L2TaskId, FillTracker] = {}

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        if not self._pipelined_load or not keys:
            return super().submit_load_task(keys, objects)
        key_strings = [_object_key_to_string(k) for k in keys]
        memviews = [
            _obj_to_memoryview(obj, self._pad_buffers_to_alignment) for obj in objects
        ]
        fill = FillTracker(objects)
        with self._lock:
            task_id = self._get_next_task_id()
            try:
                future_id = self._sliced_client.submit_batch_get(
                    key_strings, memviews, fill
                )
            except BaseException:
                fill.release()
                raise
            self._pending_ops[future_id] = (
                self._OP_LOAD,
                task_id,
                len(keys),
                list(keys),
            )
            self._load_fills[task_id] = fill
        return task_id

    def take_load_fill(self, task_id: L2TaskId) -> FillTracker | None:
        """Take a pipelined load's fill; see the base class."""
        with self._lock:
            return self._load_fills.pop(task_id, None)

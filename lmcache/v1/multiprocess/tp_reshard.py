# SPDX-License-Identifier: Apache-2.0
"""Head-sharded KV resharding between tensor-parallel sizes.

A head-sharded object stores, per kernel group, a dense
``[kv_size, num_layers, num_tokens, local_heads * head_elems]`` tensor. Heads
are contiguous and innermost, so the heads a rank needs are a slice of the
last dimension. This module maps the heads of a destination rank at one TP
size onto the source ranks at another TP size and copies those slices.

The TP size must divide the KV head count, so rank ``r`` holds the
contiguous heads ``[r * n, (r + 1) * n)`` with ``n = num_kv_heads // tp``.
A TP size larger than the head count replicates heads across ranks by an
engine-specific rule, so it is not supported.

:class:`L1Resharder` runs such a reshard against the storage manager: it
reads read-locked source objects and writes the destination objects into L1,
read-locked for the prefetch's readers, on its own threads.
:class:`HeteroTPPlanner` decides which prefetches may complete their hit from
another TP size, and :class:`HeteroTPPlan` says how: which source keys to
prefetch for the miss, how to reshard that hit and merge it.
:class:`~lmcache.v1.multiprocess.hetero_kv_view.HeteroKVView` runs the plan.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, replace
from functools import partial
from typing import cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchResult,
    PrefetchTaskSpec,
    lookup_kv_ranks,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

logger = init_logger(__name__)

# Concurrent reshards. The copies are torch CPU copies, which release the GIL,
# so a few lookups can reshard in parallel.
_RESHARD_WORKERS = 4


def _kernel_group_tensors(obj: MemoryObj) -> list[torch.Tensor]:
    """Return one tensor view per kernel group of a memory object."""
    # get_tensor is None only for an invalidated object; callers hold a read
    # lock or a write reservation, which keeps the object valid.
    return cast(
        list[torch.Tensor], [obj.get_tensor(i) for i in range(len(obj.get_shapes()))]
    )


def _group_rows(rows: list[GroupedObjectKeys]) -> list[list[GroupedObjectKeys]]:
    """Split prefetch rows into one list per object group, keeping row order."""
    groups: list[list[GroupedObjectKeys]] = []
    for row in rows:
        if groups and groups[-1][0].object_group_id == row.object_group_id:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _object_keys(
    rows: list[GroupedObjectKeys], start: int, end: int
) -> list[ObjectKey]:
    """Keys of chunks ``[start, end)`` of one object group's rank rows.

    Chunk-major and rank-minor, as :class:`ReshardTask` takes them.
    """
    return [row.keys[c] for c in range(start, end) for row in rows]


def _at_kv_rank(key: ObjectKey, kv_rank: int) -> ObjectKey:
    """The same chunk's key at another kv rank."""
    return ObjectKey(
        chunk_hash=key.chunk_hash,
        model_name=key.model_name,
        kv_rank=kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=key.cache_salt,
    )


def _check_tp(num_kv_heads: int, tp: int) -> None:
    if not is_valid_tp(num_kv_heads, tp):
        raise ValueError(f"tp={tp} must divide num_kv_heads={num_kv_heads}")


def is_valid_tp(num_kv_heads: int, tp: int) -> bool:
    """Return whether ``num_kv_heads`` KV heads split evenly over ``tp`` ranks.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.

    Returns:
        True if both are positive and ``tp`` divides ``num_kv_heads``.
    """
    if num_kv_heads < 1 or tp < 1:
        return False
    return num_kv_heads % tp == 0


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    """Return the number of KV heads one rank holds at a TP size.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.

    Returns:
        ``num_kv_heads // tp``.

    Raises:
        ValueError: If ``tp`` does not divide ``num_kv_heads``.
    """
    _check_tp(num_kv_heads, tp)
    return num_kv_heads // tp


def kv_head_range(num_kv_heads: int, tp: int, rank: int) -> tuple[int, int]:
    """Return the KV heads a rank holds as ``(first_head, num_heads)``.

    Args:
        num_kv_heads: Total KV heads of the model.
        tp: Tensor-parallel size.
        rank: Rank in ``[0, tp)``.

    Returns:
        The first global head index and the number of heads.

    Raises:
        ValueError: If the TP size is invalid for ``num_kv_heads`` or
            ``rank`` is out of range.
    """
    if not 0 <= rank < tp:
        raise ValueError(f"rank={rank} out of range for tp={tp}")
    n = local_kv_heads(num_kv_heads, tp)
    return rank * n, n


@dataclass(frozen=True)
class HeadCopy:
    """Copy ``num_heads`` heads from a source rank into a destination rank.

    Attributes:
        src_rank: Source rank holding the heads.
        src_head: First head to copy, local to the source rank.
        dst_head: First head to write, local to the destination rank.
        num_heads: Number of contiguous heads.
    """

    src_rank: int
    src_head: int
    dst_head: int
    num_heads: int


def plan_head_reshard(
    num_kv_heads: int, src_tp: int, dst_tp: int, dst_rank: int
) -> list[HeadCopy]:
    """Plan the copies that build one destination rank's heads.

    Every destination head is covered exactly once.

    Args:
        num_kv_heads: Total KV heads of the model.
        src_tp: TP size the objects were stored with.
        dst_tp: TP size of the reader.
        dst_rank: Destination rank in ``[0, dst_tp)``.

    Returns:
        Copies ordered by destination head.

    Raises:
        ValueError: If either TP size is invalid for ``num_kv_heads`` or
            ``dst_rank`` is out of range.
    """
    dst_first, dst_n = kv_head_range(num_kv_heads, dst_tp, dst_rank)
    src_n = local_kv_heads(num_kv_heads, src_tp)
    copies = []
    head = dst_first
    while head < dst_first + dst_n:
        src_rank = head // src_n
        src_first = src_rank * src_n
        n = min(src_first + src_n, dst_first + dst_n) - head
        copies.append(
            HeadCopy(
                src_rank=src_rank,
                src_head=head - src_first,
                dst_head=head - dst_first,
                num_heads=n,
            )
        )
        head += n
    return copies


def reshard_layout_desc(
    layout_desc: MemoryLayoutDesc, num_kv_heads: int, from_tp: int, to_tp: int
) -> MemoryLayoutDesc:
    """Derive the layout one rank's object has at another TP size.

    Only the last (``local_heads * head_elems``) dimension of each kernel
    group changes.

    Args:
        layout_desc: Layout at ``from_tp``.
        num_kv_heads: Total KV heads of the model.
        from_tp: TP size ``layout_desc`` describes.
        to_tp: TP size to derive the layout for.

    Returns:
        The layout at ``to_tp``.

    Raises:
        ValueError: If a TP size is invalid for ``num_kv_heads`` or a shape's
            last dimension is not a multiple of the local head count.
    """
    from_n = local_kv_heads(num_kv_heads, from_tp)
    to_n = local_kv_heads(num_kv_heads, to_tp)
    shapes = []
    for shape in layout_desc.shapes:
        if shape[-1] % from_n != 0:
            raise ValueError(
                f"last dim {shape[-1]} of {tuple(shape)} is not a multiple of "
                f"{from_n} local heads"
            )
        shapes.append(torch.Size((*shape[:-1], shape[-1] // from_n * to_n)))
    return MemoryLayoutDesc(shapes=shapes, dtypes=list(layout_desc.dtypes))


def copy_heads(
    dst: list[torch.Tensor],
    src_by_rank: dict[int, list[torch.Tensor]],
    copies: list[HeadCopy],
    num_dst_heads: int,
) -> None:
    """Execute a head reshard plan for every kernel group of one object.

    Args:
        dst: Destination tensors, one per kernel group, last dimension
            ``num_dst_heads * head_elems``.
        src_by_rank: Source tensors per source rank, one per kernel group.
        copies: Plan from :func:`plan_head_reshard`.
        num_dst_heads: Local heads of the destination rank.

    Raises:
        KeyError: If a source rank in the plan is missing from
            ``src_by_rank``.
        ValueError: If a source rank and the destination have different
            kernel group counts.
    """
    for copy in copies:
        for src_t, dst_t in zip(src_by_rank[copy.src_rank], dst, strict=True):
            head_elems = dst_t.shape[-1] // num_dst_heads
            s = copy.src_head * head_elems
            d = copy.dst_head * head_elems
            n = copy.num_heads * head_elems
            dst_t[..., d : d + n].copy_(src_t[..., s : s + n])


@dataclass(frozen=True)
class ReshardTask:
    """Reshard of the leading chunks of one request from one TP size to another.

    Key lists are per object group, chunk-major and rank-minor, as
    ``ipc_key_to_object_keys`` returns them, and cover the same chunks.

    Attributes:
        src_keys: Source object keys, each read-locked once by a prefetch.
        dst_keys: Destination object keys.
        dst_layout_descs: Layout of one destination object, per object
            group, in the order of the key lists.
        read_locks: Read locks to take per destination object, for the
            readers of the lookup's hit.
        num_kv_heads: KV heads of the whole model.
        src_tp: TP size of the source objects.
        dst_tp: TP size of the destination objects.
        l1_owners: L1 owners of the source objects, from the prefetch result.
        request_id: Request the reshard serves; used only for logging.
    """

    src_keys: list[list[ObjectKey]]
    dst_keys: list[list[ObjectKey]]
    dst_layout_descs: list[MemoryLayoutDesc]
    read_locks: int
    num_kv_heads: int
    src_tp: int
    dst_tp: int
    l1_owners: dict[ObjectKey, int]
    request_id: str


@dataclass(frozen=True)
class ReshardResult:
    """Destination objects a reshard left read-locked.

    Attributes:
        num_chunks: Leading chunks of the task whose destination objects are
            all read-locked, ``ReshardTask.read_locks`` times each.
        l1_owners: L1 owner of each of those objects.
    """

    num_chunks: int
    l1_owners: dict[ObjectKey, int]


class L1Resharder:
    """Reshard read-locked source objects into new L1 objects off the caller's thread.

    The copy grows with the hit, so it runs on a small thread pool instead of
    the request thread. Two reshards that write some of the same destination
    keys (two lookups of one prefix) are serialized: the later one waits for
    the earlier, then writes only the keys still free and read-locks the
    rest, since a key reserved by the earlier one cannot be reserved again.

    Args:
        storage_manager: Storage manager holding both the source and the
            destination objects.
    """

    def __init__(self, storage_manager: StorageManager) -> None:
        self._sm = storage_manager
        # Threads start on first use, so a server that never reshards pays
        # nothing.
        self._executor = ThreadPoolExecutor(
            max_workers=_RESHARD_WORKERS, thread_name_prefix="hetero-tp-reshard"
        )
        # Latest running reshard of each destination key.
        self._writers: dict[ObjectKey, Future[ReshardResult]] = {}
        self._writers_lock = threading.Lock()

    def submit(self, task: ReshardTask) -> Future[ReshardResult]:
        """Reshard ``task`` on a reshard thread.

        Each destination object is read-locked ``task.read_locks`` times: one
        the reshard writes as it is admitted, so eviction cannot take it
        first, and one already resident (another writer was first) as it is
        found. Only the leading chunks with every object locked are kept;
        locks past the first gap (a key out of memory, or still being
        written) are released. The source read locks are released whether or
        not the reshard succeeds; a failed reshard is logged and keeps no
        destination lock.

        Args:
            task: The reshard to run.

        Returns:
            A future resolving to what the reshard left read-locked; it never
            raises.
        """
        dst_keys = [k for group in task.dst_keys for k in group]
        with self._writers_lock:
            writers = {self._writers[k] for k in dst_keys if k in self._writers}
            future = self._executor.submit(self._run, task, writers, time.monotonic())
            # The latest reshard of a key is its writer: it already waits for
            # the earlier ones, so a later reshard waits for it alone.
            for k in dst_keys:
                self._writers[k] = future
        future.add_done_callback(partial(self._drop_writer, dst_keys))
        return future

    def close(self) -> None:
        """Stop the reshard threads after every submitted reshard finishes.

        Queued reshards still run: cancelling one would leave its source
        read locks held until their TTL.
        """
        self._executor.shutdown(wait=True)

    def _drop_writer(
        self, keys: list[ObjectKey], future: Future[ReshardResult]
    ) -> None:
        """Unregister ``future`` as the writer of ``keys`` once it is done."""
        with self._writers_lock:
            for k in keys:
                if self._writers.get(k) is future:
                    del self._writers[k]

    def _run(
        self,
        task: ReshardTask,
        writers: set[Future[ReshardResult]],
        queued_at: float,
    ) -> ReshardResult:
        """Wait for ``writers``, reshard ``task`` and log its cost.

        Args:
            task: The reshard to run.
            writers: Running reshards that write some of the same destination
                keys; this one writes only what they left free.
            queued_at: ``time.monotonic()`` when the reshard was submitted.

        Returns:
            What the reshard left read-locked; nothing if it failed.
        """
        dequeued = time.monotonic()
        # Writers were registered before this reshard, so waits never cycle.
        wait_futures(writers)
        started = time.monotonic()
        try:
            nbytes, result = self._reshard(task)
            logger.info(
                "Resharded %d chunks from TP%d to TP%d for request %s: wrote "
                "%.1f MiB in %.2f ms (queued %.2f ms, waited %.2f ms on %d "
                "other reshards)",
                result.num_chunks,
                task.src_tp,
                task.dst_tp,
                task.request_id,
                nbytes / (1 << 20),
                (time.monotonic() - started) * 1e3,
                (dequeued - queued_at) * 1e3,
                (started - dequeued) * 1e3,
                len(writers),
            )
            return result
        except Exception:
            logger.exception("Cross-TP reshard failed for request %s", task.request_id)
            return ReshardResult(0, {})

    def _reshard(self, task: ReshardTask) -> tuple[int, ReshardResult]:
        """Write and read-lock the destination objects, release the sources.

        Args:
            task: The reshard to run.

        Returns:
            Bytes written into destination objects, and what stays locked.

        Raises:
            Exception: An allocation or copy error propagates. The source
                read locks and every destination lock taken are released;
                destination objects reserved for the failing group stay
                reserved until the write TTL, since the storage manager has
                no way to abandon them.
        """
        sm = self._sm
        locked: dict[ObjectKey, int] = {}
        try:
            nbytes = self._write_and_lock(task, locked)
        except Exception:
            sm.finish_read_prefetched(
                list(locked), read_locks=task.read_locks, l1_owners=locked
            )
            raise
        dst_tp = task.dst_tp
        num_chunks = 0
        while num_chunks < len(task.dst_keys[0]) // dst_tp and all(
            k in locked
            for group in task.dst_keys
            for k in group[num_chunks * dst_tp : (num_chunks + 1) * dst_tp]
        ):
            num_chunks += 1
        kept = {
            k: locked[k]
            for group in task.dst_keys
            for k in group[: num_chunks * dst_tp]
        }
        past_gap = [k for k in locked if k not in kept]
        if past_gap:
            sm.finish_read_prefetched(
                past_gap, read_locks=task.read_locks, l1_owners=locked
            )
        return nbytes, ReshardResult(num_chunks, kept)

    def _write_and_lock(self, task: ReshardTask, locked: dict[ObjectKey, int]) -> int:
        """Write the destination objects and read-lock them into ``locked``.

        Args:
            task: The reshard to run.
            locked: Filled with the L1 owner of each destination key locked,
                as it is locked, so a caller can release them on an error.

        Returns:
            Bytes written into destination objects.

        Raises:
            Exception: An allocation or copy error propagates. The source
                read locks are released either way.
        """
        sm = self._sm
        src_tp, dst_tp = task.src_tp, task.dst_tp
        flat_src = [k for group in task.src_keys for k in group]
        nbytes = 0
        with sm.read_prefetched_results(flat_src, task.l1_owners) as src_objs:
            if src_objs is None:
                # The context already released the locks it took.
                return 0
            plans = [
                plan_head_reshard(task.num_kv_heads, src_tp, dst_tp, b)
                for b in range(dst_tp)
            ]
            dst_heads = local_kv_heads(task.num_kv_heads, dst_tp)
            src_by_key = dict(zip(flat_src, src_objs, strict=True))
            for g_src, g_dst, layout_desc in zip(
                task.src_keys, task.dst_keys, task.dst_layout_descs, strict=True
            ):
                reserved = sm.reserve_write(g_dst, layout_desc)
                for chunk in range(len(g_src) // src_tp):
                    srcs = {
                        a: _kernel_group_tensors(src_by_key[g_src[chunk * src_tp + a]])
                        for a in range(src_tp)
                    }
                    for b in range(dst_tp):
                        obj = reserved.get(g_dst[chunk * dst_tp + b])
                        if obj is not None:
                            dst = _kernel_group_tensors(obj)
                            copy_heads(dst, srcs, plans[b], dst_heads)
                            nbytes += sum(t.nbytes for t in dst)
                if reserved:
                    locked.update(
                        sm.finish_write_and_reserve_read(
                            sm.prepare_write_completion(reserved), task.read_locks
                        )
                    )
                # Another writer admitted these first: an earlier reshard of
                # the same prefix, or the engine storing it.
                taken = [k for k in g_dst if k not in reserved]
                if taken:
                    locked.update(sm.reserve_read(taken, task.read_locks))
        sm.finish_read_prefetched(flat_src, read_locks=1, l1_owners=task.l1_owners)
        return nbytes


class HeteroTPPlan:
    """How one prefetch completes its hit from objects stored at another TP size.

    Created by :meth:`HeteroTPPlanner.plan` from the prefetch's spec. After the
    spec's own prefetch hits its first ``k`` chunks, the caller submits
    :meth:`cross_tp_spec` for the rest; if that hits, :meth:`convert`
    reshards the hit into the spec's TP size in L1, and :meth:`merge` appends
    it to the first prefetch's result. Call them in that order, at most once
    each.

    Args:
        spec: The prefetch's spec, with every row at the destination TP size.
        num_kv_heads: KV heads of the whole model.
        source_tp: TP size the source objects are stored with.
        resharder: Runs the reshard of a source hit.
    """

    def __init__(
        self,
        spec: PrefetchTaskSpec,
        num_kv_heads: int,
        source_tp: int,
        resharder: L1Resharder,
    ) -> None:
        self._spec = spec
        self._groups = _group_rows(spec.key_groups)
        self._num_kv_heads = num_kv_heads
        self._dst_tp = len(self._groups[0])
        self._source_tp = source_tp
        self._resharder = resharder
        self._start = 0
        self._cross_tp_groups: list[list[GroupedObjectKeys]] = []

    @property
    def num_chunks(self) -> int:
        """Chunks the spec prefetches."""
        return self._spec.group_size

    def cross_tp_spec(self, start: int) -> PrefetchTaskSpec:
        """Describe the prefetch of the spec's chunks from ``start`` on.

        Args:
            start: Chunks the spec's own prefetch hit; the cross-TP prefetch
                covers the rest.

        Returns:
            The spec's rows rewritten to the source TP size's ranks and
            layouts, with one read lock per object: only the reshard reads
            the source objects. Policy and tier filters are the spec's own.
        """
        self._start = start
        self._cross_tp_groups = [
            [
                GroupedObjectKeys(
                    keys=[_at_kv_rank(k, rank) for k in rows[0].keys[start:]],
                    object_group_id=rows[0].object_group_id,
                    layout_desc=reshard_layout_desc(
                        rows[0].layout_desc,
                        self._num_kv_heads,
                        self._dst_tp,
                        self._source_tp,
                    ),
                    sliding_window_size=rows[0].sliding_window_size,
                )
                for rank in lookup_kv_ranks(self._source_tp)
            ]
            for rows in self._groups
        ]
        return replace(
            self._spec,
            key_groups=[row for rows in self._cross_tp_groups for row in rows],
            num_kv_readers=1,
        )

    def convert(
        self, found: int, result: PrefetchResult, request_id: str
    ) -> Future[ReshardResult]:
        """Reshard the cross-TP prefetch's hit into the spec's TP size, in L1.

        Runs on a reshard thread, releases the cross-TP prefetch's read locks
        and leaves the new objects read-locked ``num_kv_readers`` times, as
        the spec's own prefetch left its hit; see :meth:`L1Resharder.submit`.

        Args:
            found: Leading chunks the cross-TP prefetch hit.
            result: The cross-TP prefetch's result.
            request_id: Request the reshard serves; used only for logging.

        Returns:
            A future resolving to what the reshard left read-locked.
        """
        start = self._start
        return self._resharder.submit(
            ReshardTask(
                src_keys=[_object_keys(g, 0, found) for g in self._cross_tp_groups],
                dst_keys=[_object_keys(g, start, start + found) for g in self._groups],
                dst_layout_descs=[rows[0].layout_desc for rows in self._groups],
                read_locks=self._spec.num_kv_readers,
                num_kv_heads=self._num_kv_heads,
                src_tp=self._source_tp,
                dst_tp=self._dst_tp,
                l1_owners=result.l1_owners,
                request_id=request_id,
            )
        )

    def merge(self, first: PrefetchResult, resharded: ReshardResult) -> PrefetchResult:
        """Append a reshard's hit to the spec's own prefetch result.

        Args:
            first: The spec's own prefetch result; it hit the chunks before
                the cross-TP prefetch's start.
            resharded: What :meth:`convert` left read-locked.

        Returns:
            One result over the spec's rows, as if the spec's own prefetch
            had hit both. The resharded chunks count as L2 hits: the spec's
            TP size did not have them in L1.
        """
        end = self._start + resharded.num_chunks

        def extend(rows: list[Bitmap]) -> list[Bitmap]:
            out = [row.copy() for row in rows]
            for row in out:
                row.set_range(self._start, end)
            return out

        return PrefetchResult(
            hit_cells=extend(first.hit_cells),
            l1_hit_cells=[row.copy() for row in first.l1_hit_cells],
            l2_hit_cells=extend(first.l2_hit_cells),
            l1_owners={**first.l1_owners, **resharded.l1_owners},
        )


class HeteroTPPlanner:
    """Decides which prefetches complete their hit from another TP size.

    Args:
        storage_manager: Storage manager the reshards read and write.
        registry: Layout registry holding the KV head count engines report.
        source_world_size: TP size other engines of the same model store
            with; ``None`` turns cross-TP plans off.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        registry: LayoutDescRegistry,
        source_world_size: int | None,
    ) -> None:
        self._registry = registry
        self._source_world_size = source_world_size
        self._resharder = L1Resharder(storage_manager)
        # KV head counts the source world size was already warned about.
        self._warned_heads: set[int] = set()

    def plan(self, spec: PrefetchTaskSpec) -> HeteroTPPlan | None:
        """Build the cross-TP plan of a prefetch, or ``None`` if it has none.

        A prefetch qualifies when its rows fan out to every rank of one world
        size (a lookup), a source world size other than that one is
        configured, the engine reported the model-wide KV head count (it does
        so only for attention-only models whose world size equals their TP
        size), both world sizes divide that head count, and no row has a
        sliding window. A source world size that does not divide the head
        count is warned about once per head count.

        Args:
            spec: The prefetch's spec, with at least one chunk.

        Returns:
            The plan, or ``None`` if the prefetch does not qualify.
        """
        source = self._source_world_size
        if source is None or any(
            row.sliding_window_size != FULL_ATTENTION_WINDOW_CHUNKS
            for row in spec.key_groups
        ):
            return None
        first = spec.key_groups[0].keys[0]
        world_size = ObjectKey.WorldSizeFromKVRank(first.kv_rank)
        if world_size == source:
            return None
        ranks = lookup_kv_ranks(world_size)
        if any(
            [row.keys[0].kv_rank for row in rows] != ranks
            for rows in _group_rows(spec.key_groups)
        ):
            return None
        num_heads = self._registry.find_total_num_kv_heads(first.model_name, world_size)
        if num_heads is None or not is_valid_tp(num_heads, world_size):
            return None
        if not is_valid_tp(num_heads, source):
            if num_heads not in self._warned_heads:
                self._warned_heads.add(num_heads)
                logger.warning(
                    "Skipping hetero TP source world size %d: %d KV heads",
                    source,
                    num_heads,
                )
            return None
        return HeteroTPPlan(spec, num_heads, source, self._resharder)

    def close(self) -> None:
        """Stop the reshard threads; see :meth:`L1Resharder.close`."""
        self._resharder.close()

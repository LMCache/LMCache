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
reads read-locked source objects and writes the destination objects into L1
on its own threads. :class:`HeteroTPFallback` decides when a lookup may fall
back to another TP size and hands the lookup each source to try as a
:class:`SourceStage`, so the lookup itself never deals with heads or TP sizes.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Sequence, TypeVar, cast
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchResult,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

logger = init_logger(__name__)

# Concurrent reshards. The copies are torch CPU copies, which release the GIL,
# so a few lookups can reshard in parallel.
_RESHARD_WORKERS = 4

_T = TypeVar("_T")


def _kernel_group_tensors(obj: MemoryObj) -> list[torch.Tensor]:
    """Return one tensor view per kernel group of a memory object."""
    # get_tensor is None only for an invalidated object; callers hold a read
    # lock or a write reservation, which keeps the object valid.
    return cast(
        list[torch.Tensor], [obj.get_tensor(i) for i in range(len(obj.get_shapes()))]
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
        dst_layout_descs: Layout of one destination object, per object group.
        num_kv_heads: KV heads of the whole model.
        src_tp: TP size of the source objects.
        dst_tp: TP size of the destination objects.
        l1_owners: L1 owners of the source objects, from the prefetch result.
        request_id: Request the reshard serves; used only for logging.
    """

    src_keys: list[list[ObjectKey]]
    dst_keys: list[list[ObjectKey]]
    dst_layout_descs: dict[int, MemoryLayoutDesc]
    num_kv_heads: int
    src_tp: int
    dst_tp: int
    l1_owners: dict[ObjectKey, int]
    request_id: str


class L1Resharder:
    """Reshard read-locked source objects into new L1 objects off the caller's thread.

    The copy grows with the hit, so it runs on a small thread pool instead of
    the request thread. Two reshards that write some of the same destination
    keys (two lookups of one prefix) are serialized: the later one waits for
    the earlier, then writes only the keys still free, since a key reserved
    by the earlier one cannot be reserved again.

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
        # Running reshard writing each destination key.
        self._writers: dict[ObjectKey, Future[Any]] = {}
        self._writers_lock = threading.Lock()

    def submit(self, task: ReshardTask, then: Callable[[], _T]) -> Future[_T]:
        """Reshard ``task`` on a reshard thread, then call ``then`` there.

        The source read locks are released whether or not the reshard
        succeeds. A failed reshard is logged and ``then`` still runs, so the
        caller learns what reached L1 the same way in both cases. Destination
        keys that cannot be reserved (already resident, or out of memory) are
        skipped.

        Args:
            task: The reshard to run.
            then: Called on the reshard thread once the reshard is done.

        Returns:
            A future resolving to ``then``'s result, or raising its exception.
        """
        dst_keys = [k for group in task.dst_keys for k in group]
        with self._writers_lock:
            writers = {self._writers[k] for k in dst_keys if k in self._writers}
            future = self._executor.submit(
                self._run, task, then, writers, time.monotonic()
            )
            mine = [k for k in dst_keys if k not in self._writers]
            for k in mine:
                self._writers[k] = future
        future.add_done_callback(partial(self._drop_writer, mine))
        return future

    def close(self) -> None:
        """Stop the reshard threads after every submitted reshard finishes.

        Queued reshards still run: cancelling one would leave its source
        read locks held until their TTL.
        """
        self._executor.shutdown(wait=True)

    def _drop_writer(self, keys: list[ObjectKey], future: Future[Any]) -> None:
        """Unregister ``future`` as the writer of ``keys`` once it is done."""
        with self._writers_lock:
            for k in keys:
                if self._writers.get(k) is future:
                    del self._writers[k]

    def _run(
        self,
        task: ReshardTask,
        then: Callable[[], _T],
        writers: set[Future[Any]],
        queued_at: float,
    ) -> _T:
        """Wait for ``writers``, reshard ``task``, log its cost, call ``then``.

        Args:
            task: The reshard to run.
            then: Called once the reshard is done or has failed.
            writers: Running reshards that write some of the same destination
                keys; this one writes only what they left free.
            queued_at: ``time.monotonic()`` when the reshard was submitted.

        Returns:
            ``then``'s result.
        """
        dequeued = time.monotonic()
        # Writers were registered before this reshard, so waits never cycle.
        wait_futures(writers)
        started = time.monotonic()
        try:
            nbytes = self._reshard(task)
            logger.info(
                "Resharded %d chunks from TP%d to TP%d for request %s: wrote "
                "%.1f MiB in %.2f ms (queued %.2f ms, waited %.2f ms on %d "
                "other reshards)",
                len(task.src_keys[0]) // task.src_tp,
                task.src_tp,
                task.dst_tp,
                task.request_id,
                nbytes / (1 << 20),
                (time.monotonic() - started) * 1e3,
                (dequeued - queued_at) * 1e3,
                (started - dequeued) * 1e3,
                len(writers),
            )
        except Exception:
            logger.exception("Cross-TP reshard failed for request %s", task.request_id)
        return then()

    def _reshard(self, task: ReshardTask) -> int:
        """Write the destination objects of ``task`` and release its sources.

        Args:
            task: The reshard to run.

        Returns:
            Bytes written into destination objects.

        Raises:
            Exception: An allocation or copy error propagates. The source
                read locks are released either way; destination objects
                reserved for the failing group stay reserved until the write
                TTL, since the storage manager has no way to abandon them.
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
            for gid, (g_src, g_dst) in enumerate(
                zip(task.src_keys, task.dst_keys, strict=True)
            ):
                reserved = sm.reserve_write(g_dst, task.dst_layout_descs[gid])
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
                    sm.finish_write_by_owner(sm.prepare_write_completion(reserved))
        sm.finish_read_prefetched(flat_src, read_locks=1, l1_owners=task.l1_owners)
        return nbytes


@dataclass(frozen=True)
class SourceStage:
    """A prefetch at another TP size that a lookup submits as one stage.

    Attributes:
        key: The lookup's key at the source TP size.
        group_layout_descs: Per-object-group layouts at the source TP size.
        attn_desc: Attention windows at the source TP size.
        num_kv_readers: Read locks to take per object; only the reshard reads
            the source objects, so this is 1.
    """

    key: IPCCacheServerKey
    group_layout_descs: dict[int, MemoryLayoutDesc]
    attn_desc: AttnWindowDesc
    num_kv_readers: int


class HeteroTPPlan:
    """The cross-TP fallback of one lookup: sources to try, and the reshard.

    Created by :meth:`HeteroTPFallback.plan`. The lookup calls
    :meth:`next_source` for each source stage it submits. When a source stage
    hits, the lookup either calls :meth:`convert` to reshard the hit into its
    own TP size, or, when it stops early, :meth:`release` to drop the source
    stage's read locks. Not thread-safe; the lookup serializes its calls.

    Args:
        key: The lookup's own key.
        suffix_hashes: Chunk hashes of the looked-up range.
        attn_desc: Attention windows at the lookup's TP size.
        group_layout_descs: Per-object-group layouts at the lookup's TP size.
        num_kv_heads: KV heads of the whole model.
        candidates: Source TP sizes to try, in order.
        storage_manager: Storage manager the source stages lock in.
        resharder: Runs the reshard of a source hit.
    """

    def __init__(
        self,
        key: IPCCacheServerKey,
        suffix_hashes: list[bytes],
        attn_desc: AttnWindowDesc,
        group_layout_descs: dict[int, MemoryLayoutDesc],
        num_kv_heads: int,
        candidates: list[int],
        storage_manager: StorageManager,
        resharder: L1Resharder,
    ) -> None:
        self._key = key
        self._suffix_hashes = suffix_hashes
        self._attn_desc = attn_desc
        self._group_layout_descs = group_layout_descs
        self._num_kv_heads = num_kv_heads
        self._candidates = candidates
        self._sm = storage_manager
        self._resharder = resharder
        self._source: int | None = None

    def next_source(self) -> SourceStage | None:
        """Move to the next source TP size and describe its prefetch.

        Returns:
            The stage to submit, or ``None`` if no source TP size is left.
        """
        if not self._candidates:
            return None
        source = self._candidates.pop(0)
        self._source = source
        return SourceStage(
            key=replace(self._key, world_size=source),
            group_layout_descs={
                gid: reshard_layout_desc(
                    desc, self._num_kv_heads, self._key.world_size, source
                )
                for gid, desc in self._group_layout_descs.items()
            },
            attn_desc=replace(self._attn_desc, world_size=source),
            num_kv_readers=1,
        )

    def convert(
        self, found: int, result: PrefetchResult, then: Callable[[], _T]
    ) -> Future[_T]:
        """Reshard the current source stage's hit into the lookup's TP size.

        Runs on a reshard thread and releases the source stage's read locks;
        see :meth:`L1Resharder.submit`.

        Args:
            found: Leading chunks the current source stage hit.
            result: The current source stage's result.
            then: Called on the reshard thread once the reshard is done.

        Returns:
            A future resolving to ``then``'s result.
        """
        source = cast(int, self._source)
        hashes = self._suffix_hashes[:found]
        groups = list(range(self._attn_desc.num_object_groups))
        return self._resharder.submit(
            ReshardTask(
                src_keys=ipc_key_to_object_keys(
                    replace(self._key, world_size=source), hashes, groups
                ),
                dst_keys=ipc_key_to_object_keys(self._key, hashes, groups),
                dst_layout_descs=self._group_layout_descs,
                num_kv_heads=self._num_kv_heads,
                src_tp=source,
                dst_tp=self._key.world_size,
                l1_owners=result.l1_owners,
                request_id=self._key.request_id,
            ),
            then,
        )

    def release(self, found: int, result: PrefetchResult) -> None:
        """Drop the read locks of the current source stage's hit.

        Args:
            found: Leading chunks the current source stage hit.
            result: The current source stage's result.
        """
        self._sm.finish_read_prefetched(
            [
                k
                for group in ipc_key_to_object_keys(
                    replace(self._key, world_size=cast(int, self._source)),
                    self._suffix_hashes[:found],
                    list(range(self._attn_desc.num_object_groups)),
                )
                for k in group
            ],
            read_locks=1,
            l1_owners=result.l1_owners,
        )


class HeteroTPFallback:
    """Decides which lookups may fall back to other TP sizes, and builds plans.

    Args:
        registry: Layout registry holding the KV head count engines report.
        storage_manager: Storage manager the reshards read and write.
        source_world_sizes: TP sizes other engines store with, tried in order.
    """

    def __init__(
        self,
        registry: LayoutDescRegistry,
        storage_manager: StorageManager,
        source_world_sizes: Sequence[int],
    ) -> None:
        self._registry = registry
        self._sm = storage_manager
        self._source_world_sizes = list(source_world_sizes)
        self._resharder = L1Resharder(storage_manager)
        # (KV heads, source world size) pairs already warned about.
        self._warned_invalid_sources: set[tuple[int, int]] = set()

    def plan(
        self,
        key: IPCCacheServerKey,
        suffix_hashes: list[bytes],
        attn_desc: AttnWindowDesc,
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> HeteroTPPlan | None:
        """Build the cross-TP fallback for a lookup, or ``None`` if it has none.

        A lookup qualifies when source world sizes are configured, the engine
        reported the model-wide KV head count (it does so only for
        attention-only models whose world size equals their TP size), the
        lookup's world size divides that head count, and no object group has
        a sliding window. Candidates that do not divide the head count are
        dropped with a warning.

        Args:
            key: The lookup's key.
            suffix_hashes: Chunk hashes of the looked-up range.
            attn_desc: Attention windows at the lookup's world size.
            group_layout_descs: Per-object-group layouts at the lookup's
                world size.

        Returns:
            The fallback plan, or ``None`` if the lookup does not qualify.
        """
        if not self._source_world_sizes:
            return None
        num_heads = self._registry.find_total_num_kv_heads(
            key.model_name, key.world_size
        )
        if (
            num_heads is None
            or not is_valid_tp(num_heads, key.world_size)
            or any(w >= 0 for w in attn_desc.num_chunks_in_sw)
        ):
            return None
        candidates = []
        for world_size in self._source_world_sizes:
            if world_size == key.world_size:
                continue
            if not is_valid_tp(num_heads, world_size):
                if (num_heads, world_size) not in self._warned_invalid_sources:
                    self._warned_invalid_sources.add((num_heads, world_size))
                    logger.warning(
                        "Skipping hetero TP source world size %d: %d KV heads",
                        world_size,
                        num_heads,
                    )
                continue
            candidates.append(world_size)
        if not candidates:
            return None
        return HeteroTPPlan(
            key,
            suffix_hashes,
            attn_desc,
            group_layout_descs,
            num_heads,
            candidates,
            self._sm,
            self._resharder,
        )

    def close(self) -> None:
        """Stop the reshard threads after every submitted reshard finishes.

        Queued reshards still run: cancelling one would leave its source
        read locks held until their TTL.
        """
        self._resharder.close()

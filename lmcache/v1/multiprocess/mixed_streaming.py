# SPDX-License-Identifier: Apache-2.0
"""Plan a contiguous prefix across locked L1 objects and FS streaming."""

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey, PrefetchResult, PrefetchTaskSpec
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.multiprocess.streaming import FSStreamingReader


def plan_mixed_streaming(
    spec: PrefetchTaskSpec,
    managers: list[L1ManagerInterface],
    reader: FSStreamingReader,
) -> PrefetchResult | None:
    """Reserve L1 hits and find the prefix that can be served with FS windows.

    The caller owns returned L1 read locks. Returns None after releasing all
    locks if no FS object participates, leaving the normal prefetch path to
    handle the request. On an exception every acquired L1 lock is released.
    """
    held: dict[ObjectKey, tuple[L1ManagerInterface, int]] = {}
    l1_rows = [Bitmap(spec.group_size) for _ in spec.key_groups]
    l2_rows = [Bitmap(spec.group_size) for _ in spec.key_groups]
    accepted = False
    try:
        for row_index, row in enumerate(spec.key_groups):
            for manager in managers:
                pending = [key for key in row.keys if key not in held]
                if not pending:
                    break
                for key, (error, _) in manager.reserve_read(
                    pending, row.num_kv_readers
                ).items():
                    if error == L1Error.SUCCESS:
                        held[key] = (manager, row.num_kv_readers)
            for index, key in enumerate(row.keys):
                if key in held:
                    l1_rows[row_index].set(index)
        missing = [
            (row_index, index, key)
            for row_index, row in enumerate(spec.key_groups)
            for index, key in enumerate(row.keys)
            if key not in held
        ]
        if missing:
            disk = reader.lookup(
                [key for _, _, key in missing], spec.group_layout_descs
            )
            for index, (row_index, column, _) in enumerate(missing):
                if disk.test(index):
                    l2_rows[row_index].set(column)
        hits = []
        for l1, l2 in zip(l1_rows, l2_rows, strict=True):
            row_hits = Bitmap(spec.group_size)
            for index in range(spec.group_size):
                if l1.test(index) or l2.test(index):
                    row_hits.set(index)
            hits.append(row_hits)
        _, retained = fold_unfold_grouped(
            hits, [row.sliding_window_size for row in spec.key_groups]
        )
        l1_retained = [Bitmap(spec.group_size) for _ in spec.key_groups]
        l2_retained = [Bitmap(spec.group_size) for _ in spec.key_groups]
        hit_retained = [Bitmap(spec.group_size) for _ in spec.key_groups]
        retained_keys: set[ObjectKey] = set()
        for row_index, row in enumerate(spec.key_groups):
            for index, key in enumerate(row.keys):
                if not retained[row_index].test(index):
                    continue
                retained_keys.add(key)
                hit_retained[row_index].set(index)
                if l1_rows[row_index].test(index):
                    l1_retained[row_index].set(index)
                else:
                    l2_retained[row_index].set(index)
        if not any(row.popcount() for row in l2_retained):
            return None
        for key, (manager, count) in list(held.items()):
            if key not in retained_keys:
                manager.finish_read([key], count)
                del held[key]
        result = PrefetchResult(
            hit_cells=hit_retained,
            l1_hit_cells=l1_retained,
            l2_hit_cells=l2_retained,
            l1_owners={
                key: manager.l1_manager_id for key, (manager, _) in held.items()
            },
        )
        accepted = True
        return result
    finally:
        if not accepted:
            for key, (manager, count) in held.items():
                manager.finish_read([key], count)

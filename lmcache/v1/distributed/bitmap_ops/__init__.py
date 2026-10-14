# SPDX-License-Identifier: Apache-2.0
"""Bitmap arithmetic for multi-object-group prefix-cache hit computation.

See :mod:`~lmcache.v1.distributed.bitmap_ops.fold` and the
package ``README.md`` for the fold -> highest-set-bit -> unfold design.
"""

# First Party
from lmcache.v1.distributed.bitmap_ops.fold import (
    FULL_ATTENTION_WINDOW,
    fold,
    fold_grouped,
    fold_unfold_grouped,
    highest_set_bit,
    merge_bitmaps,
    unfold,
    unfold_grouped,
    unfold_range,
)

__all__ = [
    "FULL_ATTENTION_WINDOW",
    "highest_set_bit",
    "fold",
    "fold_grouped",
    "fold_unfold_grouped",
    "merge_bitmaps",
    "unfold",
    "unfold_grouped",
    "unfold_range",
]

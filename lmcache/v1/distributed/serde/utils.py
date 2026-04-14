# SPDX-License-Identifier: Apache-2.0
"""
Serde helper utilities for the distributed storage controllers.
"""

# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import SerdeProcessor

# Fixed multiplier on top of estimate_serialized_size() for buffer allocation.
# Can be replaced with custom logic later.
SERDE_BUFFER_FACTOR = 1.5


def serialized_layout_desc(
    layout_desc: MemoryLayoutDesc,
    serde: SerdeProcessor,
) -> MemoryLayoutDesc:
    """Compute a flat byte-buffer MemoryLayoutDesc for the serialized output.

    Returns a single-group uint8 layout sized at SERDE_BUFFER_FACTOR *
    the processor's estimated size.
    """
    estimated = serde.estimate_serialized_size(layout_desc)
    buffer_size = int(estimated * SERDE_BUFFER_FACTOR)
    return MemoryLayoutDesc(
        shapes=[torch.Size([buffer_size])],
        dtypes=[torch.uint8],
    )


def make_temp_key(original_key: ObjectKey, purpose: str) -> ObjectKey:
    """Create a unique temporary key derived from the original.

    Uses a random suffix so it never collides with real keys.

    Args:
        original_key: The original ObjectKey to derive from.
        purpose: "ser" or "deser", for debugging only.
    """
    random_suffix = os.urandom(8)
    temp_hash = original_key.chunk_hash + random_suffix
    return ObjectKey(
        chunk_hash=temp_hash,
        model_name=original_key.model_name,
        kv_rank=original_key.kv_rank,
    )

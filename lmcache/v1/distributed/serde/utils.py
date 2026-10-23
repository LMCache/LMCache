# SPDX-License-Identifier: Apache-2.0
"""
Serde helper utilities for the distributed storage controllers.
"""

# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.cache_identity import ObjectKeyIdentity
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import SerdeProcessor


def serialized_layout_desc(
    layout_desc: MemoryLayoutDesc,
    serde: SerdeProcessor,
) -> MemoryLayoutDesc:
    """Compute a flat byte-buffer MemoryLayoutDesc for the serialized output.

    Returns a single-group uint8 layout whose size is determined by the
    serde processor's ``estimate_serialized_size`` (which already includes
    any safety margin).
    """
    buffer_size = serde.estimate_serialized_size(layout_desc)
    return MemoryLayoutDesc(
        shapes=[torch.Size([buffer_size])],
        dtypes=[torch.uint8],
    )


def make_temp_key(original_key: ObjectKey) -> ObjectKey:
    """Create a unique temporary key derived from the original.

    Appends 16 random bytes (128 bits of entropy) to the original
    chunk hash. Birthday-collision risk is ~1 in 2**64 keys, which is
    effectively zero at any realistic scale, so the same original key
    can be serde'd repeatedly without practical concern.

    ``cache_salt`` are propagated so per-tenant L1 byte accounting and
    quota / eviction logic continue to attribute temp buffers to the
    same bucket as the originals.

    Args:
        original_key: The original ObjectKey to derive from.
    """
    return ObjectKey(
        identity=ObjectKeyIdentity(
            chunk_hash=original_key.identity.chunk_hash + os.urandom(16),
            model_name=original_key.identity.model_name,
            kv_rank=original_key.identity.kv_rank,
            object_group_id=original_key.identity.object_group_id,
            cache_salt=original_key.identity.cache_salt,
        )
    )

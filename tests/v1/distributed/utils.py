# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for distributed tests."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.platform import current_device_spec
import lmcache.v1.memory_management as memory_management

OBJECT_BYTES = 4096
_OBJECT_LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])


def should_use_lazy_alloc() -> bool:
    """Return whether the current platform supports lazy L1 allocation."""
    return current_device_spec.is_pin_supported


def single_row_spec(
    keys: list[ObjectKey], layout_desc: MemoryLayoutDesc, **spec_kwargs: Any
) -> PrefetchTaskSpec:
    """A prefetch spec with one key row: object group 0, a single kv rank.

    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec` (e.g.
    ``num_kv_readers``, ``fetching_policy``, ``lock_mode``).
    """
    return PrefetchTaskSpec(
        key_groups=[
            GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)
        ],
        **spec_kwargs,
    )


def ranked_spec(
    keys_by_rank: list[list[ObjectKey]],
    layout_desc: MemoryLayoutDesc,
    **spec_kwargs: Any,
) -> PrefetchTaskSpec:
    """A single-object-group prefetch spec with one key row per kv rank.

    ``keys_by_rank[r]`` is the chunk-ordered key list of kv rank ``r``.
    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec`.
    """
    return PrefetchTaskSpec(
        key_groups=[
            GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)
            for keys in keys_by_rank
        ],
        **spec_kwargs,
    )


def cpu_storage(monkeypatch: pytest.MonkeyPatch) -> tuple[L1Manager, StorageManager]:
    """A real CPU L1 of 32 objects behind a real storage manager.

    Pinned allocation and free are replaced through
    ``memory_management._allocate_cpu_memory`` / ``_free_cpu_memory``; the
    manager is injected through the ``_l1_managers`` seam, as test_multi_l1.py
    does. The caller closes the storage manager.
    """
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    config = L1ManagerConfig(
        L1MemoryManagerConfig(
            size_in_bytes=32 * OBJECT_BYTES, use_lazy=False, shm_name=""
        )
    )
    manager = L1Manager(config)
    store = StorageManager(
        StorageManagerConfig(config, EvictionConfig("noop")), _l1_managers=(manager,)
    )
    return manager, store


def object_key(
    chunk_hash: bytes,
    model_name: str = "m",
    rank: int = 0,
    world_size: int = 1,
    cache_salt: str = "",
) -> ObjectKey:
    """The key of ``chunk_hash`` for kv rank ``rank`` of ``world_size``."""
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=ObjectKey.ComputeKVRank(world_size, rank, world_size, rank),
        cache_salt=cache_salt,
    )


def commit(l1: L1Manager, *keys: ObjectKey, temporary: bool = False) -> None:
    """Write ``OBJECT_BYTES``-byte objects under ``keys`` and finish the write."""
    l1.reserve_write(list(keys), [temporary] * len(keys), _OBJECT_LAYOUT)
    l1.finish_write(list(keys))


def stage(l1: L1Manager, *keys: ObjectKey) -> None:
    """Reserve a write under ``keys`` without finishing it."""
    l1.reserve_write(list(keys), [False] * len(keys), _OBJECT_LAYOUT)

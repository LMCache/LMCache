# SPDX-License-Identifier: Apache-2.0
"""Child process that owns shared-L1 allocation metadata, never KV payload."""

# Standard
from multiprocessing import get_context
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

# First Party
from lmcache.v1.distributed.shared_l1.pool import SharedL1Pool

_EXPOSED_METHODS = (
    "region_contract",
    "reserve_writes",
    "finish_writes",
    "abort_writes",
    "reserve_reads",
    "finish_reads",
    "abort_reads",
    "snapshot",
)
_pool: SharedL1Pool | None = None


def read_shared_l1_authkey(authkey_file: str) -> bytes:
    """Read a nonempty manager key without putting its bytes in configuration."""
    path = Path(authkey_file)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("shared-L1 authkey must be an absolute regular file")
    authkey = path.read_bytes()
    if not authkey:
        raise ValueError("shared-L1 authkey file must not be empty")
    return authkey


def _initialize_pool(
    region_id: str,
    capacity: int,
    alignment: int,
    layout_id: str,
) -> None:
    """Construct the sole allocator inside the child process."""
    global _pool
    _pool = SharedL1Pool(region_id, capacity, alignment, layout_id)


def _get_pool() -> SharedL1Pool:
    if _pool is None:
        raise RuntimeError("shared-L1 pool is not initialized")
    return _pool


class SharedL1Manager(BaseManager):
    """Manager type shared by the coordinator child and its clients."""

    def get_pool(self) -> Any:
        """Return the coordinator-owned metadata proxy."""
        raise NotImplementedError


SharedL1Manager.register(
    "get_pool",
    callable=_get_pool,
    exposed=_EXPOSED_METHODS,
)


def connect_shared_l1_manager(
    host: str,
    port: int,
    authkey: bytes,
) -> SharedL1Manager:
    """Connect an MP server to the coordinator child."""
    if not authkey:
        raise ValueError("shared-L1 authkey must not be empty")
    manager = SharedL1Manager(address=(host, port), authkey=authkey)
    manager.connect()
    return manager


def start_shared_l1_manager(
    *,
    host: str,
    port: int,
    authkey: bytes,
    region_id: str,
    capacity: int,
    alignment: int,
    layout_id: str,
) -> SharedL1Manager:
    """Start the coordinator child and return its lifecycle handle."""
    manager = SharedL1Manager(
        address=(host, port),
        authkey=authkey,
        ctx=get_context("spawn"),
    )
    manager.start(
        initializer=_initialize_pool,
        initargs=(region_id, capacity, alignment, layout_id),
    )
    return manager

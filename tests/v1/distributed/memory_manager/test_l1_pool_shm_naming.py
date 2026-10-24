# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the L1 pool SHM segment naming contract."""

# Standard
from unittest import mock
import re
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    L1_POOL_SHM_NAME_ATTEMPTS,
    L1_POOL_SHM_NAME_TOKEN_HEX_LEN,
    L1MemoryManagerConfig,
    generated_l1_pool_shm_name,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.multiprocess.posix_shm import POSIX_SHM_MAX_NAME_LEN

_NAME_PATTERN = re.compile(
    r"lmcache_l1_pool_[0-9a-f]{%d}\Z" % L1_POOL_SHM_NAME_TOKEN_HEX_LEN
)
_FAKE_PID = 424_242
_POOL_SIZE = 64 * 1024


def test_default_shm_name_is_uuid_derived():
    """Generated L1 pool names are UUID tokens, not PID-derived.

    POSIX SHM is host-global and containers with independent PID
    namespaces recycle the same low PIDs, so a PID-derived default
    would collide on ``shm_open``. The token length is derived from
    ``POSIX_SHM_MAX_NAME_LEN`` (30 on macOS, 255 on Linux) so generated
    names fit on every supported POSIX target.
    """
    names = set()
    for _ in range(4):
        config = L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False)
        assert _NAME_PATTERN.fullmatch(config.shm_name), (
            f"Expected pattern {_NAME_PATTERN.pattern!r}, got {config.shm_name!r}"
        )
        names.add(config.shm_name)
    assert len(names) == 4


def test_generated_l1_pool_shm_name_fits_posix_limit():
    """A generated L1 pool ``shm_open`` name must fit the platform limit."""
    bare = generated_l1_pool_shm_name()
    shm_open_name = "/" + bare
    assert len(shm_open_name) <= POSIX_SHM_MAX_NAME_LEN
    assert (
        len(shm_open_name) == len("/lmcache_l1_pool_") + L1_POOL_SHM_NAME_TOKEN_HEX_LEN
    )


def test_l1_pool_token_hex_len_matches_platform_limit():
    """Token hex length is derived from ``POSIX_SHM_MAX_NAME_LEN``."""
    expected = min(16, POSIX_SHM_MAX_NAME_LEN - len("/lmcache_l1_pool_"))
    assert L1_POOL_SHM_NAME_TOKEN_HEX_LEN == expected
    if sys.platform == "darwin":
        assert L1_POOL_SHM_NAME_TOKEN_HEX_LEN == 13


def test_l1_pools_with_same_pid_do_not_collide():
    """Two pools configured with the same logical PID stay independent.

    Regression for PID-namespace collisions: a PID-derived default in
    two containers sharing ``/dev/shm`` produced the same
    ``lmcache_l1_pool_<N>`` name, so the second pool unlinked the
    first pool's live segment and silently re-created the name. The
    ``os.getpid`` patch is only a regression trap -- the generated
    name must not depend on it.
    """
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([8, 2, 64])],
        dtypes=[torch.bfloat16],
    )
    with mock.patch("os.getpid", return_value=_FAKE_PID):
        config1 = L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False)
        config2 = L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False)
    manager1 = L1MemoryManager(config1)
    manager2 = L1MemoryManager(config2)
    try:
        assert config1.shm_name != config2.shm_name
        # Both pools are independently usable after coexisting.
        error1, objects1 = manager1.allocate(layout, 1)
        error2, objects2 = manager2.allocate(layout, 1)
        assert error1 == error2 == L1Error.SUCCESS
        manager1.free(objects1)
        manager2.free(objects2)
    finally:
        manager1.close()
        manager2.close()


def test_unlink_stale_shm_runs_only_for_pinned_names(monkeypatch):
    """Generated pool names must not call ``_unlink_stale_shm`` before create."""
    # First Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager as l1mm

    unlinked: list[str] = []
    monkeypatch.setattr(
        l1mm,
        "_unlink_stale_shm",
        lambda name: unlinked.append(name),
    )

    generated_manager = L1MemoryManager(
        L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False),
    )
    try:
        assert unlinked == []
    finally:
        generated_manager.close()

    pinned_manager = L1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=_POOL_SIZE,
            use_lazy=False,
            shm_name="lmcache_l1_pool_pinned_test",
        ),
    )
    try:
        assert unlinked == ["lmcache_l1_pool_pinned_test"]
    finally:
        pinned_manager.close()


def test_l1_pool_shm_name_collision_retries_with_fresh_name(monkeypatch):
    """Generated pool names retry on ``O_EXCL`` collisions with a new UUID token."""
    # First Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager as l1mm

    real_mixed = l1mm.MixedMemoryAllocator
    attempted: list[str] = []

    def flaky_mixed(*args, **kwargs):
        name = kwargs.get("shm_name")
        if name is not None:
            attempted.append(name)
        if len(attempted) <= 2:
            raise FileExistsError("simulated name collision")
        return real_mixed(*args, **kwargs)

    monkeypatch.setattr(l1mm, "MixedMemoryAllocator", flaky_mixed)

    config = L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False)
    manager = L1MemoryManager(config)
    try:
        assert len(attempted) == 3
        assert len(set(attempted)) == 3
        assert config.shm_name == attempted[-1], (
            "config.shm_name must reflect the name actually used by "
            "MixedMemoryAllocator"
        )
    finally:
        manager.close()


def test_l1_pool_shm_name_collision_exhausts_retries(monkeypatch):
    """Pinned names do not mask a live collision by switching to a new UUID."""
    # First Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager as l1mm

    def always_collides(*args, **kwargs):
        raise FileExistsError("simulated persistent collision")

    monkeypatch.setattr(l1mm, "MixedMemoryAllocator", always_collides)

    config = L1MemoryManagerConfig(
        size_in_bytes=_POOL_SIZE,
        use_lazy=False,
        shm_name="lmcache_l1_pool_pinned_test",
    )
    with pytest.raises(FileExistsError, match="simulated persistent collision"):
        L1MemoryManager(config)


def test_l1_pool_generated_name_collision_exhausts_retry_budget(monkeypatch):
    """Generated names propagate ``FileExistsError`` after exhausting retries."""
    # First Party
    from lmcache.v1.distributed.memory_manager import l1_memory_manager as l1mm

    attempted: list[str] = []

    def always_collides(*args, **kwargs):
        name = kwargs.get("shm_name")
        if name is not None:
            attempted.append(name)
        raise FileExistsError("simulated persistent collision")

    monkeypatch.setattr(l1mm, "MixedMemoryAllocator", always_collides)

    config = L1MemoryManagerConfig(size_in_bytes=_POOL_SIZE, use_lazy=False)
    with pytest.raises(FileExistsError, match="simulated persistent collision"):
        L1MemoryManager(config)
    assert len(attempted) == L1_POOL_SHM_NAME_ATTEMPTS
    assert len(set(attempted)) == len(attempted)

# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.platform.cpu.shm``.

Validates that the POSIX-SHM-backed wrapper can round-trip a CPU
tensor in-process: the constructed wrapper's ``to_tensor()`` view
sees writes made through the original tensor.
"""

# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.cpu.shm import (
    CpuShmTensorWrapper,
    migrate_to_shm_and_wrap,
    shm_create_readwrite,
    shm_unlink,
)


def test_shm_create_unlink_roundtrip(tmp_path):
    """``shm_create_readwrite`` succeeds and ``shm_unlink`` cleans up."""
    name = "/lmcache_test_%d" % os.getpid()
    addr = shm_create_readwrite(name, 4096)
    try:
        assert addr not in (0, None)
    finally:
        shm_unlink(name)


def test_migrate_to_shm_and_wrap_zero_copy_view():
    """After migrate, writes via the original tensor are visible via wrapper."""
    src = torch.zeros((2, 4, 4), dtype=torch.float32)
    wrapper = migrate_to_shm_and_wrap(src)
    try:
        assert isinstance(wrapper, CpuShmTensorWrapper)
        assert wrapper.shape == (2, 4, 4)
        assert wrapper.dtype == torch.float32
        # Mutate via the migrated source tensor; its storage is now the
        # SHM segment, so the wrapper's reconstructed view must see it.
        src.add_(7.0)
        view = wrapper.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(wrapper.shm_name)


def test_migrate_is_idempotent_on_same_tensor():
    """Re-wrapping the same tensor reuses the existing SHM segment."""
    src = torch.zeros((3, 5), dtype=torch.float32)
    w1 = migrate_to_shm_and_wrap(src)
    try:
        w2 = migrate_to_shm_and_wrap(src)
        assert w1.shm_name == w2.shm_name
    finally:
        shm_unlink(w1.shm_name)


def test_rejects_non_cpu_tensor():
    """Construction rejects tensors that are not on CPU."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available; cannot synthesize a non-cpu tensor")
    src = torch.zeros((2, 2), device="mps")
    with pytest.raises(ValueError, match="CPU tensor"):
        CpuShmTensorWrapper(src, "/lmcache_test_should_not_exist")


def test_migrate_finalizer_unlinks_on_gc():
    """Once the migrated tensor is GC-ed, its SHM segment is unlinked."""
    # Standard
    import gc

    # First Party
    from lmcache.v1.platform.cpu.shm import shm_map_readwrite

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    name = w.shm_name
    nbytes = w.nbytes
    # Drop both references; the weakref.finalize hook should unlink.
    del src, w
    gc.collect()
    with pytest.raises(OSError):
        shm_map_readwrite(name, nbytes)


def test_shm_create_cleans_up_on_existing_name():
    """If ``shm_open(O_EXCL)`` fails the helper must not leave the fd open.

    We exercise the failure path by creating a segment, then asking
    ``shm_create_readwrite`` to recreate the same name -- it must
    raise without leaking the file descriptor it briefly held.
    """
    name = "/lmcache_test_excl_%d" % os.getpid()
    addr = shm_create_readwrite(name, 4096)
    try:
        with pytest.raises(OSError):
            shm_create_readwrite(name, 4096)
    finally:
        shm_unlink(name)
    # And after unlink, the name is reusable again.
    addr2 = shm_create_readwrite(name, 4096)
    assert addr2 not in (0, None)
    shm_unlink(name)
    _ = addr  # silence unused-variable hint


def test_to_tensor_view_carries_munmap_finalizer():
    """``to_tensor`` returns a tensor that releases its mmap on GC."""
    # Standard
    import gc
    import weakref

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        view = w.to_tensor()
        # The view must keep ``flat`` alive so its mmap stays valid.
        assert hasattr(view, "_lmcache_shm_buf")
        ref = weakref.ref(view)
        del view
        gc.collect()
        assert ref() is None
    finally:
        del src
        gc.collect()
        shm_unlink(w.shm_name)

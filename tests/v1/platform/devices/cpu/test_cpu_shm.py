# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.platform.devices.cpu.shm``.

Validates that the POSIX-SHM-backed wrapper can round-trip a CPU
tensor in-process: the constructed wrapper's ``to_tensor()`` view
sees writes made through the original tensor.
"""

# Standard
import errno
import multiprocessing
import os
import re

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.posix_shm import POSIX_SHM_MAX_NAME_LEN, shm_unlink
from lmcache.v1.platform.devices.cpu.shm import (
    SHM_KV_NAME_TOKEN_HEX_LEN,
    SHM_SEGMENT_NAME_ATTEMPTS,
    CpuShmTensorWrapper,
    migrate_to_shm_and_wrap,
    shm_create_readwrite,
)

# Legacy pid+counter names collided across host-global POSIX SHM namespaces.
_LEGACY_PID_COUNTER_SHM_NAME = re.compile(r"/lmcache_kv_\d+_\d+\Z")


def _assert_not_legacy_pid_counter_name(name: str) -> None:
    assert not _LEGACY_PID_COUNTER_SHM_NAME.fullmatch(name), (
        f"SHM name {name!r} uses legacy pid+counter naming"
    )


def test_shm_create_unlink_roundtrip():
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


def test_migrate_normalizes_layout_in_place_on_caller_tensor():
    """Layout normalization must not detach migration from the caller's tensor.

    ``attempt_permute_to_contiguous_view`` returns a *new view* object; the
    migration must still re-point the caller's tensor (its storage, and its
    shape/stride to the stride-truthful layout) and key the unlink finalizer
    on it. Regression for the premature-unlink bug where the finalizer was
    attached to the temporary view and the segment vanished on return.
    """
    # Torch-contiguous, but the size-1 dims hide the true inner extent:
    # the stride-truthful layout is (8, 32, 1, 1).
    src = torch.zeros(8 * 32, dtype=torch.float32).as_strided(
        (8, 1, 1, 32), (32, 1, 1, 1)
    )
    wrapper = migrate_to_shm_and_wrap(src)
    try:
        assert tuple(src.shape) == (8, 32, 1, 1)
        assert wrapper.shape == (8, 32, 1, 1)
        # Writes via the caller's tensor land in the SHM segment.
        src.add_(3.0)
        view = wrapper.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(wrapper.shm_name)


def test_migrate_handles_empty_tensor():
    """Empty tensors must not call ``mmap`` (length 0 is EINVAL).

    Regression for the case where ``nbytes == 0``: the wrapper carries
    an empty ``shm_name`` and ``to_tensor`` rebuilds the empty view in
    process without touching POSIX shared memory.
    """
    src = torch.empty((0, 4), dtype=torch.float32)
    wrapper = migrate_to_shm_and_wrap(src)
    assert isinstance(wrapper, CpuShmTensorWrapper)
    assert wrapper.shm_name == ""
    assert wrapper.nbytes == 0
    view = wrapper.to_tensor()
    assert view.shape == (0, 4)
    assert view.dtype == torch.float32


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
    from lmcache.v1.platform.devices.cpu.shm import shm_map_readwrite

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
    """``to_tensor`` returns a tensor whose storage releases its mmap on GC."""
    # Standard
    import gc
    import weakref

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        view = w.to_tensor()
        ref = weakref.ref(view)
        del view
        gc.collect()
        assert ref() is None
    finally:
        del src
        gc.collect()
        shm_unlink(w.shm_name)


def test_to_tensor_view_survives_reshape_after_original_is_gced():
    """A reshape view must keep working after the original tensor is GC-ed.

    Regression for the CI ``cpu_e2e_validation (server-side copy)`` segfault
    where ``normalize_kv_and_discover_format`` reshapes the unwrapped 4D
    tensor into the canonical 5D ``[NB, NH, BS, 2, HS]`` and the original
    4D tensor goes out of scope. The reshape view shares the same storage
    but used to be left without a lifetime hook, so the next read after GC
    landed on an already-``munmap``-ed page and crashed the LMCache server.
    """
    # Standard
    import gc

    NB, NH, BS, HS = 8, 4, 16, 8
    src = torch.zeros((NB, NH, BS, 2 * HS), dtype=torch.bfloat16)
    w = migrate_to_shm_and_wrap(src)
    try:
        unwrapped = [w.to_tensor()]
        normalized = [
            layer.reshape(*layer.shape[:3], 2, layer.shape[3] // 2)
            for layer in unwrapped
        ]
        # Drop every reference to the 4D unwrapped tensor; only the 5D
        # reshape view keeps the storage alive now.
        del unwrapped
        gc.collect()

        # These ops would segfault before the fix because the SHM mapping
        # had been munmap-ed when the 4D tensor's finalizer ran.
        idx = torch.tensor([0, 3, 5], dtype=torch.long)
        gathered = normalized[0].index_select(0, idx)
        assert tuple(gathered.shape) == (3, NH, BS, 2, HS)
        # The SHM segment is freshly mmap'd zeros; just make sure the
        # bytes are addressable end-to-end so the kernel does not fault.
        assert gathered.float().abs().sum().item() == 0.0
    finally:
        del src
        gc.collect()
        shm_unlink(w.shm_name)


def test_to_tensor_replays_stride_and_storage_offset():
    """``to_tensor`` rebuilds the view via stride+offset (not reshape)."""
    src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).contiguous()
    w = migrate_to_shm_and_wrap(src)
    try:
        view = w.to_tensor()
        assert tuple(view.stride()) == w.stride
        assert int(view.storage_offset()) == w.storage_offset
        assert torch.equal(view, src)
    finally:
        del src, view
        shm_unlink(w.shm_name)


def test_wrap_kv_caches_unlinks_partial_batch_on_failure(monkeypatch):
    """If wrapping the N-th tensor raises, earlier SHM names are unlinked.

    Drives :func:`wrap_kv_caches` with two CPU tensors and forces the
    second factory call to raise; the first iteration's SHM segment
    must be ``shm_unlink``-ed so the named segment does not outlive
    the failed batch.
    """
    # First Party
    from lmcache.v1.platform import kv_wrap
    from lmcache.v1.platform.devices.cpu.shm import shm_map_readwrite

    real_wrap = kv_wrap.wrap_one_kv_cache
    state = {"n": 0, "first_name": None}

    def flaky_wrap(tensor):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("simulated migration failure")
        w = real_wrap(tensor)
        state["first_name"] = w.shm_name
        return w

    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", flaky_wrap)

    t1 = torch.zeros((2, 2), dtype=torch.float32)
    t2 = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        kv_wrap.wrap_kv_caches({"a": t1, "b": t2})

    # The first iteration's SHM segment must no longer be openable.
    nbytes = t1.numel() * t1.element_size()
    with pytest.raises(OSError):
        shm_map_readwrite(state["first_name"], nbytes)


def test_migrate_shares_one_segment_across_views_of_one_storage():
    """Views of one backing buffer share a single SHM segment.

    Engines may register every KV layer as a strided view into one
    shared allocation (nonzero ``storage_offset``): migration must key
    on the backing storage so every view lands in one segment, keeps
    its own offset, and stays aliased to the same bytes. Regression
    for the storage_offset assert crash in ``cpu_e2e_validation
    (server_copy)``.
    """
    layers, layer_numel = 4, 1024
    base = torch.zeros(layers * layer_numel, dtype=torch.bfloat16)
    views = [
        base[i * layer_numel : (i + 1) * layer_numel].view(4, 16, 16)
        for i in range(layers)
    ]

    wrappers = [migrate_to_shm_and_wrap(v) for v in views]
    try:
        # One segment for all views, each keeping its own offset.
        names = {w.shm_name for w in wrappers}
        assert len(names) == 1
        segment_nbytes = layers * layer_numel * base.element_size()
        for i, w in enumerate(wrappers):
            assert w.storage_offset == i * layer_numel
            assert w.nbytes == segment_nbytes

        # All migrated views share the SHM-backed storage.
        storages = {id(v.untyped_storage()) for v in views}
        assert len(storages) == 1

        # Aliasing survives the round-trip in both directions.
        views[2].fill_(3.0)
        rebuilt = wrappers[2].to_tensor()
        assert torch.equal(rebuilt, views[2])
        rebuilt.add_(1.0)
        assert views[2].float().max().item() == 4.0
        # Sibling views are untouched (offsets do not overlap).
        assert views[1].float().abs().sum().item() == 0.0
    finally:
        shm_unlink(wrappers[0].shm_name)


def test_migrate_preserves_nonzero_storage_offset_roundtrip():
    """A lone offset view wraps, ships its offset, and round-trips data."""
    base = torch.arange(300, dtype=torch.float32)
    view = base[100:200]
    w = migrate_to_shm_and_wrap(view)
    try:
        assert w.storage_offset == 100
        assert w.nbytes == 300 * 4  # whole backing storage
        rebuilt = w.to_tensor()
        assert torch.equal(rebuilt, torch.arange(100, 200, dtype=torch.float32))
    finally:
        shm_unlink(w.shm_name)


def test_migrate_is_idempotent_across_sibling_views():
    """Wrapping sibling views (or re-wrapping one) reuses the segment."""
    base = torch.zeros(2 * 64, dtype=torch.float32)
    v1, v2 = base[:64], base[64:]
    w1 = migrate_to_shm_and_wrap(v1)
    try:
        w2 = migrate_to_shm_and_wrap(v2)
        w1_again = migrate_to_shm_and_wrap(v1)
        assert w1.shm_name == w2.shm_name == w1_again.shm_name
    finally:
        shm_unlink(w1.shm_name)


def test_migrate_segment_unlinks_only_after_last_view_dies():
    """The shared segment must outlive every view, not just the first."""
    # Standard
    import gc

    # First Party
    from lmcache.v1.platform.devices.cpu.shm import shm_map_readwrite, shm_munmap

    base = torch.zeros(2 * 64, dtype=torch.float32)
    v1, v2 = base[:64], base[64:]
    w1 = migrate_to_shm_and_wrap(v1)
    w2 = migrate_to_shm_and_wrap(v2)
    name, nbytes = w1.shm_name, w1.nbytes

    del base, v1, w1, w2
    gc.collect()
    # v2 still holds the SHM-backed storage: the segment must survive.
    addr = shm_map_readwrite(name, nbytes)
    shm_munmap(addr, nbytes)

    del v2
    gc.collect()
    with pytest.raises(OSError):
        shm_map_readwrite(name, nbytes)


def test_wrapper_rejects_view_larger_than_segment():
    """A view whose end lies past the declared segment must be refused."""
    base = torch.zeros(200, dtype=torch.float32)
    view = base[100:200]
    with pytest.raises(ValueError, match="segment"):
        CpuShmTensorWrapper(view, "/lmcache_test_never_created", segment_nbytes=100 * 4)


def test_kv_shm_name_fits_posix_limit():
    """CPU KV ``shm_open`` names must fit within ``POSIX_SHM_MAX_NAME_LEN``."""
    prefix_len = len(CpuShmTensorWrapper.SHM_NAME_PREFIX)
    assert prefix_len + SHM_KV_NAME_TOKEN_HEX_LEN <= POSIX_SHM_MAX_NAME_LEN


def test_shm_name_format():
    """SHM names must match /lmcache_kv_<N hex> (platform-derived N).

    A UUID-per-segment name avoids PID collisions across containers with
    independent PID namespaces.  We validate the format directly rather
    than checking for the absence of the PID decimal string, which could
    yield a false positive when the UUID hex coincidentally contains the
    PID digits (e.g. PID 1 appears in almost any hex string).
    """
    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        pattern = re.compile(r"/lmcache_kv_[0-9a-f]{%d}\Z" % SHM_KV_NAME_TOKEN_HEX_LEN)
        assert pattern.fullmatch(w.shm_name), (
            f"SHM name {w.shm_name!r} does not match expected format "
            f"'/lmcache_kv_<{SHM_KV_NAME_TOKEN_HEX_LEN} hex chars>'"
        )
    finally:
        shm_unlink(w.shm_name)


def _worker_allocate_and_report(queue: "multiprocessing.Queue[str | None]") -> None:
    """Allocate one SHM segment and send its name back through *queue*."""
    # First Party
    from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap

    t = torch.zeros((4, 4), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(t)
    queue.put(w.shm_name)
    # Block until the parent signals it is safe to exit (segment still mapped).
    queue.get()


def test_shm_segment_names_ignore_monkeypatched_getpid(monkeypatch):
    """Segment names must not revert to ``/lmcache_kv_<pid>_<n>`` naming.

    If ``migrate_to_shm_and_wrap`` called ``os.getpid()`` again, a fixed
    monkeypatched PID would produce predictable legacy names; UUID naming
    must stay independent of process id.
    """
    fake_pid = 12_345
    monkeypatch.setattr(os, "getpid", lambda: fake_pid)
    wa = migrate_to_shm_and_wrap(torch.zeros((2, 2), dtype=torch.float32))
    wb = migrate_to_shm_and_wrap(torch.zeros((3, 3), dtype=torch.float32))
    try:
        _assert_not_legacy_pid_counter_name(wa.shm_name)
        _assert_not_legacy_pid_counter_name(wb.shm_name)
        assert wa.shm_name != f"/lmcache_kv_{fake_pid}_0"
        assert wb.shm_name != f"/lmcache_kv_{fake_pid}_1"
    finally:
        shm_unlink(wa.shm_name)
        shm_unlink(wb.shm_name)


def test_shm_names_are_unique_across_spawned_workers():
    """Spawned workers must allocate distinct host-global SHM segment names.

    Regression for the original ``EEXIST`` crash when legacy pid+counter
    naming reused ``/lmcache_kv_<N>_0`` across containers.  The second
    worker starts while the first segment is still mapped so a duplicate
    name would fail ``shm_open(O_CREAT|O_EXCL)``.
    """
    ctx = multiprocessing.get_context("spawn")
    q1: multiprocessing.Queue[str | None] = ctx.Queue()
    q2: multiprocessing.Queue[str | None] = ctx.Queue()
    p1 = ctx.Process(target=_worker_allocate_and_report, args=(q1,))
    p2 = ctx.Process(target=_worker_allocate_and_report, args=(q2,))
    name1: str | None = None
    name2: str | None = None
    try:
        p1.start()
        name1 = q1.get(timeout=30)
        # Second worker starts while the first segment is still mapped.
        p2.start()
        name2 = q2.get(timeout=30)
        assert name1 != name2, (
            f"Both workers allocated the same SHM name {name1!r}; "
            "per-segment UUID naming must be unique on the host"
        )
        _assert_not_legacy_pid_counter_name(name1)
        _assert_not_legacy_pid_counter_name(name2)
    finally:
        # Unblock workers so they exit cleanly before we unlink.
        q1.put(None)
        q2.put(None)
        p1.join(timeout=10)
        p2.join(timeout=10)
        for name in (name1, name2):
            if name is None:
                continue
            try:
                shm_unlink(name)
            except OSError:
                pass


def test_shm_name_collision_retries_with_fresh_name(monkeypatch):
    """A rare SHM name collision is retried with a fresh name.

    ``shm_create_readwrite`` opens with ``O_CREAT|O_EXCL``, so an
    already-existing segment name raises ``FileExistsError``: migration
    must draw a new UUID name and succeed instead of propagating.  The
    fake creator below collides twice before delegating to the real one.
    """
    # First Party
    from lmcache.v1.platform.devices.cpu import shm as shm_module

    real_create = shm_create_readwrite
    attempted: list[str] = []

    def flaky_create(name: str, nbytes: int) -> int:
        attempted.append(name)
        if len(attempted) <= 2:
            raise FileExistsError("simulated name collision")
        return real_create(name, nbytes)

    monkeypatch.setattr(shm_module, "shm_create_readwrite", flaky_create)

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        assert len(attempted) == 3
        # Every attempt drew a distinct name before one succeeded.
        assert len(set(attempted)) == 3
        src.add_(1.0)
        view = w.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(w.shm_name)


def test_shm_name_collision_retries_on_oserror_eexist(monkeypatch):
    """OSError(EEXIST) is retried the same way as FileExistsError.

    Some POSIX implementations raise ``OSError`` with ``errno.EEXIST``
    directly rather than ``FileExistsError``.  The retry loop must treat
    both identically.
    """
    # First Party
    from lmcache.v1.platform.devices.cpu import shm as shm_module

    real_create = shm_create_readwrite
    attempted: list[str] = []

    def flaky_create(name: str, nbytes: int) -> int:
        attempted.append(name)
        if len(attempted) <= 2:
            raise OSError(errno.EEXIST, "simulated EEXIST")
        return real_create(name, nbytes)

    monkeypatch.setattr(shm_module, "shm_create_readwrite", flaky_create)

    src = torch.zeros((2, 2), dtype=torch.float32)
    w = migrate_to_shm_and_wrap(src)
    try:
        assert len(attempted) == 3
        assert len(set(attempted)) == 3
        src.add_(1.0)
        view = w.to_tensor()
        assert torch.equal(view, src)
    finally:
        shm_unlink(w.shm_name)


def test_shm_name_collision_exhausts_retries_and_raises(monkeypatch):
    """Persistent name collisions propagate after exhausting retries.

    When every attempt fails with ``FileExistsError`` the error must
    reach the caller (after retrying with fresh names) and leave the
    caller's tensor untouched -- name allocation happens before any
    storage mutation.
    """
    # First Party
    from lmcache.v1.platform.devices.cpu import shm as shm_module

    attempted: list[str] = []

    def always_collides(name: str, nbytes: int) -> int:
        attempted.append(name)
        raise FileExistsError("simulated persistent collision")

    monkeypatch.setattr(shm_module, "shm_create_readwrite", always_collides)

    src = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(FileExistsError, match="simulated persistent collision"):
        migrate_to_shm_and_wrap(src)
    assert len(attempted) == SHM_SEGMENT_NAME_ATTEMPTS
    assert len(set(attempted)) == len(attempted)
    # ...and the failed migration did not touch the caller's data.
    assert torch.equal(src, torch.zeros_like(src))


def test_migrate_ignores_stale_entry_from_id_reuse():
    """A cached entry whose weakref is dead must not be reused.

    Simulates CPython recycling an object id by injecting a stale
    ``(dead_ref, old_name)`` tuple keyed by the live tensor's id,
    then calling :func:`migrate_to_shm_and_wrap`. The factory must
    treat the dead entry as a miss and allocate a fresh SHM segment
    -- if it blindly reused the cached name, ``shm_create_readwrite``
    would crash with ``EEXIST`` (and even worse, the fresh tensor
    would be silently bound to the wrong SHM name).
    """
    # Standard
    import gc
    import weakref as _wr

    # First Party
    from lmcache.v1.platform.devices.cpu.shm import inject_stale_cache_entry_for_test

    # Build a tensor we will let die so we have a guaranteed-dead ref.
    ghost = torch.zeros((1,), dtype=torch.float32)
    dead_ref = _wr.ref(ghost)
    del ghost
    gc.collect()
    assert dead_ref() is None

    live = torch.zeros((2, 2), dtype=torch.float32)
    stale_name = "/lmcache_test_stale_%d" % os.getpid()
    inject_stale_cache_entry_for_test(live, dead_ref, stale_name)

    w = migrate_to_shm_and_wrap(live)
    try:
        assert w.shm_name != stale_name
    finally:
        shm_unlink(w.shm_name)

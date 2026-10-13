# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GDS L1 context (``GDSContext``).

Most tests are pure (no GPU storage library): they exercise the public interface
(singleton/no-op semantics, the <=16 MiB region split through the backend object,
and the registered-region mapping driven through
:meth:`GDSContext.transfer_async`). Round-trip tests cover cuFile on NVIDIA,
hipFile on AMD ROCm, or an explicitly configured uGDS device, and are skipped
unless the corresponding stack is available.
"""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.memory_manager import GDSL1MemoryManager
from lmcache.v1.gpu_connector import gds_context
from lmcache.v1.gpu_connector._gds_backends import create_backend
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_context import (
    GDSContext,
    SlabDirection,
    get_gds_context,
    initialize_gds_context,
)
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObjMetadata


def _fake_stream(handle: int):
    """A stand-in for a platform stream (no accelerator needed)."""
    return SimpleNamespace(raw_handle=handle, synchronize=lambda: None)


def _use_fake_stream(monkeypatch: pytest.MonkeyPatch, handle: int) -> None:
    """Route GDSContext's platform stream operations to a fake stream."""
    monkeypatch.setattr(
        gds_context.platform_stream,
        "current_stream",
        lambda device: _fake_stream(handle),
    )
    monkeypatch.setattr(
        gds_context.platform_stream,
        "stream_handle",
        lambda device, stream: stream.raw_handle,
    )


def _gds_available() -> bool:
    """Whether a real GPUDirect Storage stack is present for the roundtrip tests.

    NVIDIA: CUDA plus the nvidia-fs kernel module (``/proc/driver/nvidia-fs``).
    AMD ROCm: a loadable ``libhipfile.so`` (the hipFile GPU IO library). When
    the GDS-capable driver/filesystem is absent hipFile still round-trips
    correctly via its host-bounce fallback, so library loadability is a
    sufficient gate for the correctness checks below.
    """
    if torch_device_type != "cuda" or not torch_dev.is_available():
        return False
    if torch.version.hip is not None:
        # Standard
        import ctypes

        try:
            ctypes.CDLL("libhipfile.so")
        except OSError:
            return False
        return True
    return os.path.exists("/proc/driver/nvidia-fs/stats")


def _ugds_device() -> str:
    """Return the opt-in uGDS test device, or an empty string when unset.

    These tests write to the raw device, destroying whatever occupies the slab
    range. They are therefore opt-in via ``UGDS_TEST_DEVICE`` rather than
    probing for any ``/dev/ugds_drv*``: on a host with several bound devices,
    probing could overwrite one the caller did not intend to sacrifice.
    """
    device = os.environ.get("UGDS_TEST_DEVICE", "")
    if not device or not os.path.exists(device):
        return ""
    if not torch.cuda.is_available():
        return ""
    # Standard
    import ctypes

    try:
        ctypes.CDLL("libugds.so")
    except OSError:
        return ""
    return device


requires_gds = pytest.mark.skipif(
    not _gds_available(),
    reason="needs CUDA + nvidia-fs or ROCm + libhipfile.so (real GPUDirect Storage)",
)
requires_ugds = pytest.mark.skipif(
    not _ugds_device(),
    reason=(
        "set UGDS_TEST_DEVICE to a scratch uGDS device (e.g. /dev/ugds_drv0); "
        "also needs CUDA or ROCm and a matching libugds.so. These tests "
        "overwrite the device."
    ),
)


def _skip_unless_gds_registrable(directory: Path) -> None:
    """Skip the calling test when ``directory`` cannot back a GDS slab file.

    Creates a small preallocated file in ``directory``, opens it with
    ``O_DIRECT``, and registers it with the GDS driver, mirroring how
    ``GDSContext`` opens its slab. Registration fails on filesystems the
    driver does not support; on NVIDIA, nvidia-fs rejects files on
    device-mapper volumes (LVM) with cuFile error 5008 whenever cuFile compat
    mode is disabled.

    Args:
        directory: The directory in which the roundtrip test would place the
            GDS slab file.
    """
    try:
        with tempfile.TemporaryDirectory(
            dir=directory, prefix=".gds_registration_probe_"
        ) as probe_dir:
            backend = create_backend("auto")
            try:
                with backend.open_slab(probe_dir, 4096, direct_io=True):
                    pass
            finally:
                backend.close_driver()
    except (OSError, RuntimeError) as exc:
        # OSError: open/fallocate; RuntimeError: cufile/hipfile registration.
        pytest.skip(
            f"GDS driver cannot register files under {directory} ({exc}); "
            "point LMCACHE_GDS_TEST_DIR at a GDS-capable filesystem"
        )


@pytest.fixture
def gds_slab_dir(tmp_path: Path) -> Iterator[Path]:
    """Directory holding the GDS slab file for the real-DMA roundtrip tests.

    On NVIDIA the directory must be named explicitly via the
    ``LMCACHE_GDS_TEST_DIR`` environment variable, pointing at a GDS-capable
    filesystem (e.g. ext4 on a directly attached NVMe device); otherwise the
    roundtrip tests are skipped. pytest's ``tmp_path`` is deliberately not
    used as a fallback: it lives on the OS disk rather than a data disk set
    up for GDS, and OS disks commonly use LVM/device-mapper volumes that
    nvidia-fs cannot register, where cuFile either fails with error 5008 or,
    with compat mode enabled, silently falls back to POSIX IO, so a passing
    test would not have exercised the DMA path it claims to.

    On AMD ROCm ``tmp_path`` is used when the variable is unset: per
    :func:`_gds_available`, library loadability is a sufficient gate for
    these correctness checks because the hipFile host-bounce fallback still
    round-trips correctly, so no GDS-capable filesystem is required.

    The resolved directory is probed with a small registration first and the
    test is skipped, rather than failed, when the driver rejects it.
    """
    override = os.environ.get("LMCACHE_GDS_TEST_DIR", "")
    if not override:
        if torch.version.hip is None:
            pytest.skip(
                "set LMCACHE_GDS_TEST_DIR to a directory on a GDS-capable "
                "filesystem to run the real-DMA roundtrip tests (pytest's "
                "tmp_path may silently use the POSIX compat fallback)"
            )
        _skip_unless_gds_registrable(tmp_path)
        yield tmp_path
        return
    with tempfile.TemporaryDirectory(dir=override, prefix="lmcache_gds_test_") as d:
        slab_dir = Path(d)
        _skip_unless_gds_registrable(slab_dir)
        yield slab_dir


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Drop the process-global GDSContext between tests."""
    get_gds_context.cache_clear()
    yield
    get_gds_context.cache_clear()


class TestSingleton:
    def test_singleton_identity(self):
        assert get_gds_context() is get_gds_context()

    def test_fresh_context_is_off(self):
        assert GDSContext().initialized is False

    def test_initialize_with_none_is_noop(self):
        ctx = initialize_gds_context(None)
        assert ctx is get_gds_context()
        assert ctx.initialized is False


@pytest.fixture
def backend() -> Mock:
    return Mock(spec=GDSBackend)


class TestRegisterGpuBuffer:
    def test_noop_when_uninitialized(self, monkeypatch, backend):
        ctx = GDSContext(backend)
        registered = []
        monkeypatch.setattr(backend, "register_buffer", registered.append)
        # GDS off -> registers nothing, makes no cuFile calls.
        ctx.register_gpu_buffer(torch.empty(4096, dtype=torch.uint8))
        assert registered == []

    def test_splits_buffer_into_regions(self, monkeypatch, backend):
        ctx = GDSContext(backend)
        ctx.initialized = True
        # Record each cuFile registration's byte size through the backend.
        sizes = []
        monkeypatch.setattr(
            backend,
            "register_buffer",
            lambda buf: sizes.append(buf.numel() * buf.element_size()),
        )
        monkeypatch.setattr(backend, "register_stream", lambda raw: None)
        _use_fake_stream(monkeypatch, 0)

        # The whole buffer is registered in <=16 MiB regions, irrespective of
        # any chunk/slot layout. A 40 MiB buffer -> 16 + 16 + 8 MiB.
        # A CPU tensor is fine: the cuFile calls are mocked.
        buf = torch.empty(40 << 20, dtype=torch.uint8)
        ctx.register_gpu_buffer(buf)

        assert sizes == [16 << 20, 16 << 20, 8 << 20]

    def test_rolls_back_regions_when_registration_fails(self, monkeypatch):
        backend = Mock(spec=GDSBackend)
        ctx = GDSContext(backend)
        ctx.initialized = True
        registered_sizes: list[int] = []
        deregistered_sizes: list[int] = []
        stream_registrations: list[int] = []
        stream_deregistrations: list[int] = []

        def register_buffer(region):
            registered_sizes.append(region.numel() * region.element_size())
            if len(registered_sizes) == 2:
                raise RuntimeError("registration failed")

        monkeypatch.setattr(backend, "register_buffer", register_buffer)
        monkeypatch.setattr(
            backend,
            "deregister_buffer",
            lambda region: deregistered_sizes.append(
                region.numel() * region.element_size()
            ),
        )
        monkeypatch.setattr(backend, "register_stream", stream_registrations.append)
        monkeypatch.setattr(backend, "deregister_stream", stream_deregistrations.append)
        _use_fake_stream(monkeypatch, 7)

        with pytest.raises(RuntimeError, match="registration failed"):
            ctx.register_gpu_buffer(torch.empty(40 << 20, dtype=torch.uint8))

        assert registered_sizes == [16 << 20, 16 << 20]
        assert deregistered_sizes == [16 << 20]
        assert stream_registrations == [7]
        assert stream_deregistrations == [7]
        assert ctx._base_ptrs == []


@pytest.mark.parametrize("concurrency", [0, 4])
def test_concurrent_slots_own_disjoint_native_registrations(
    backend: Mock,
    monkeypatch: pytest.MonkeyPatch,
    concurrency: int,
) -> None:
    """No native region may be shared by two concurrently used 18 MiB slots."""
    backend.name = "cufile"
    backend.open_slab.return_value.path = "/test"
    _use_fake_stream(monkeypatch, 7)
    monkeypatch.setattr(
        gds_context.platform_stream, "synchronize_stream", lambda *_: None
    )
    monkeypatch.setattr(
        gds_context.platform_stream, "synchronize_device", lambda _: None
    )
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 80 << 20, io_concurrency=concurrency))
    buffer = torch.empty(72 << 20, dtype=torch.uint8)
    try:
        context.register_gpu_buffer(buffer, slot_bytes=18 << 20)
        regions = [
            (call.args[0].data_ptr() - buffer.data_ptr(), call.args[0].numel())
            for call in backend.register_buffer.call_args_list
        ]
        if concurrency:
            assert regions == [
                (offset << 20, size << 20)
                for offset, size in [
                    (0, 16),
                    (16, 2),
                    (18, 16),
                    (34, 2),
                    (36, 16),
                    (52, 2),
                    (54, 16),
                    (70, 2),
                ]
            ]
            assert all(
                start // (18 << 20) == (start + size - 1) // (18 << 20)
                for start, size in regions
            )
        else:
            assert regions == [
                (offset << 20, size << 20)
                for offset, size in [(0, 16), (16, 16), (32, 16), (48, 16), (64, 8)]
            ]
        context.deregister_gpu_buffer(buffer, slot_bytes=18 << 20)
        retired = [
            (call.args[0].data_ptr() - buffer.data_ptr(), call.args[0].numel())
            for call in backend.deregister_buffer.call_args_list
        ]
        assert retired == regions
    finally:
        context.close()


def test_failed_slot_registration_rolls_back_exact_native_regions(
    backend: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure entering slot two releases both differently sized slot-one regions."""
    backend.name = "cufile"
    backend.open_slab.return_value.path = "/test"
    backend.register_buffer.side_effect = [None, None, OSError("registration")]
    _use_fake_stream(monkeypatch, 7)
    monkeypatch.setattr(
        gds_context.platform_stream, "synchronize_device", lambda _: None
    )
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 80 << 20, io_concurrency=4))
    buffer = torch.empty(72 << 20, dtype=torch.uint8)
    try:
        with pytest.raises(OSError, match="registration"):
            context.register_gpu_buffer(buffer, slot_bytes=18 << 20)
        retired = [
            (call.args[0].data_ptr() - buffer.data_ptr(), call.args[0].numel())
            for call in backend.deregister_buffer.call_args_list
        ]
        assert retired == [(16 << 20, 2 << 20), (0, 16 << 20)]
        backend.deregister_stream.assert_called_once_with(7)
    finally:
        context.close()
    assert backend.deregister_buffer.call_count == 2


class TestResolveBuffer:
    """Region mapping: a buffer slice resolves to ``(region base, offset)``,
    exercised through the public ``transfer_async`` path."""

    def _registered_ctx(self, monkeypatch, buf: torch.Tensor, backend):
        """Register ``buf``; capture the ``(base, offset)`` that
        ``transfer_async`` resolves a slice to before handing it to the slab."""
        ctx = GDSContext(backend)
        ctx.initialized = True
        monkeypatch.setattr(backend, "register_buffer", lambda b: None)
        monkeypatch.setattr(backend, "register_stream", lambda raw: None)
        _use_fake_stream(monkeypatch, 0)
        ctx.register_gpu_buffer(buf)
        resolved: list[tuple[int, int]] = []

        def record_resolved(
            slab_offset: int,
            size: int,
            dev_offset: int,
            buf_base: int,
            stream_handle: int,
        ) -> object:
            resolved.append((buf_base, dev_offset))
            return object()

        monkeypatch.setattr(
            ctx,
            "_slab_write",
            record_resolved,
        )
        return ctx, resolved

    def test_maps_slice_to_base_and_offset(self, monkeypatch, backend):
        buf = torch.empty(8192, dtype=torch.uint8)
        ctx, resolved = self._registered_ctx(monkeypatch, buf, backend)
        mem_obj = SimpleNamespace(get_size=lambda: 4096, slab_offset=0)
        # A slice 4 KiB into the region must map to (region base, offset 4096).
        ctx.transfer_async(mem_obj, buf[4096:], SlabDirection.WRITE)
        assert resolved == [(buf.data_ptr(), 4096)]


class TestPerStreamRegistration:
    """Each distinct stream is cuFile-registered once and deregistered once its
    last region is gone -- observed at the backend interface (no private state)."""

    def test_register_and_deregister_per_stream(self, monkeypatch, backend):
        ctx = GDSContext(backend)
        ctx.initialized = True
        reg_str: list[int] = []
        dereg_str: list[int] = []
        dereg_buf: list[int] = []
        monkeypatch.setattr(backend, "register_buffer", lambda b: None)
        monkeypatch.setattr(
            backend, "deregister_buffer", lambda b: dereg_buf.append(b.data_ptr())
        )
        monkeypatch.setattr(backend, "register_stream", reg_str.append)
        monkeypatch.setattr(backend, "deregister_stream", dereg_str.append)

        def use_stream(handle: int):
            _use_fake_stream(monkeypatch, handle)

        buf_a = torch.empty(24 << 20, dtype=torch.uint8)  # 2 regions on stream 11
        buf_b = torch.empty(4096, dtype=torch.uint8)  # 1 region on stream 22
        use_stream(11)
        ctx.register_gpu_buffer(buf_a)
        use_stream(22)
        ctx.register_gpu_buffer(buf_b)
        # Each distinct stream registered exactly once.
        assert reg_str == [11, 22]

        # Deregistering buf_b frees stream 22's only region -> stream 22 dropped.
        use_stream(22)
        ctx.deregister_gpu_buffer(buf_b)
        assert dereg_str == [22]
        # Stream 11 still has 2 regions (24 MiB -> 16 + 8), so not yet dropped.
        use_stream(11)
        ctx.deregister_gpu_buffer(buf_a)
        assert dereg_str == [22, 11]
        assert len(dereg_buf) == 3  # all three slots deregistered


@requires_ugds
def test_ugds_context_roundtrip():
    """Round-trip a multi-region chunk through the LMCache uGDS L1 path."""
    cfg = GdsL1Config(
        file_location=_ugds_device(),
        size_in_bytes=64 << 20,
        backend="ugds",
    )
    chunk_bytes = 24 << 20
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        buffer = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
        ctx.register_gpu_buffer(buffer)
        manager = GDSL1MemoryManager(cfg)
        error, objects = manager.allocate(
            MemoryLayoutDesc(
                shapes=[torch.Size([chunk_bytes])],
                dtypes=[torch.uint8],
            ),
            1,
        )
        assert error == L1Error.SUCCESS

        expected = (torch.arange(chunk_bytes, dtype=torch.int64) % 251).to(torch.uint8)
        buffer.copy_(expected.cuda())
        torch.cuda.synchronize()
        ctx.transfer_async(objects[0], buffer, SlabDirection.WRITE)
        torch.cuda.synchronize()

        buffer.zero_()
        torch.cuda.synchronize()
        ctx.transfer_async(objects[0], buffer, SlabDirection.READ)
        torch.cuda.synchronize()

        assert torch.equal(buffer.cpu(), expected)
        ctx.deregister_gpu_buffer(buffer)
    finally:
        ctx.close()


@requires_gds
def test_gds_two_stream_write_read(gds_slab_dir: Path):
    """Verify independent chunk writes and reads on two GPU streams."""
    cfg = GdsL1Config(file_location=str(gds_slab_dir), size_in_bytes=64 << 20)
    chunk_bytes = 8 << 20
    ctx = GDSContext()
    ctx.initialize(cfg)
    mgr = GDSL1MemoryManager(cfg)

    def register_and_write(stream, pattern):
        """Register a buffer on ``stream`` and write ``pattern`` to a chunk."""
        with torch_dev.stream(stream):
            buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=torch_device_type)
            ctx.register_gpu_buffer(buf)
            err, objs = mgr.allocate(
                MemoryLayoutDesc(
                    shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]
                ),
                1,
            )
            assert err == L1Error.SUCCESS
            buf.fill_(pattern)
            torch_dev.synchronize()
            ctx.transfer_async(objs[0], buf, SlabDirection.WRITE)
            torch_dev.synchronize()
        return buf, objs[0]

    stream_a = torch_dev.Stream()
    stream_b = torch_dev.Stream()
    try:
        buf_a, mem_a = register_and_write(stream_a, 0xA1)
        buf_b, mem_b = register_and_write(stream_b, 0xB2)

        # Read each chunk back on its own stream; each must see its own pattern,
        # confirming the two streams' buffers/regions don't clobber each other.
        for stream, buf, mem, pattern in (
            (stream_a, buf_a, mem_a, 0xA1),
            (stream_b, buf_b, mem_b, 0xB2),
        ):
            with torch_dev.stream(stream):
                buf.zero_()
                torch_dev.synchronize()
                ctx.transfer_async(mem, buf, SlabDirection.READ)
                torch_dev.synchronize()
                expected = torch.full((chunk_bytes,), pattern, dtype=torch.uint8)
                assert torch.equal(buf.cpu(), expected)

        # Deregister each buffer on its own stream.
        for stream, buf in ((stream_a, buf_a), (stream_b, buf_b)):
            with torch_dev.stream(stream):
                ctx.deregister_gpu_buffer(buf)
    finally:
        ctx.close()


@requires_gds
def test_gds_write_read_roundtrip(gds_slab_dir: Path):
    """Cold write then read of a chunk through the real cuFile DMA path."""
    cfg = GdsL1Config(file_location=str(gds_slab_dir), size_in_bytes=64 << 20)
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        chunk_bytes = 8 << 20
        buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=torch_device_type)
        ctx.register_gpu_buffer(buf)

        mgr = GDSL1MemoryManager(cfg)
        err, objs = mgr.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]),
            1,
        )
        assert err == L1Error.SUCCESS
        mem_obj = objs[0]
        assert isinstance(mem_obj, GDSMemoryObject)

        buf.fill_(0xAB)
        torch_dev.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.WRITE)

        buf.zero_()
        torch_dev.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.READ)
        torch_dev.synchronize()

        expected = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8)
        assert torch.equal(buf.cpu(), expected)
    finally:
        ctx.close()


@requires_gds
def test_gds_chunk_larger_than_region_roundtrip(gds_slab_dir: Path):
    """A chunk larger than the 16 MiB cuFile region cap round-trips correctly.

    Exercises the multi-region registration and the split (per-segment) DMA
    path: a 24 MiB chunk is registered/transferred as a 16 MiB + 8 MiB pair.
    """
    cfg = GdsL1Config(file_location=str(gds_slab_dir), size_in_bytes=64 << 20)
    ctx = GDSContext()
    ctx.initialize(cfg)
    try:
        chunk_bytes = 24 << 20  # > 16 MiB -> two registered regions / two DMAs
        buf = torch.empty(chunk_bytes, dtype=torch.uint8, device=torch_device_type)
        ctx.register_gpu_buffer(buf)

        mgr = GDSL1MemoryManager(cfg)
        err, objs = mgr.allocate(
            MemoryLayoutDesc(shapes=[torch.Size([chunk_bytes])], dtypes=[torch.uint8]),
            1,
        )
        assert err == L1Error.SUCCESS
        mem_obj = objs[0]
        assert isinstance(mem_obj, GDSMemoryObject)

        # Position-dependent pattern: a mis-offset or swapped segment (e.g. the
        # second segment using the wrong slab offset) would corrupt the bytes
        # around the 16 MiB boundary, which a uniform fill would not catch.
        pattern = (torch.arange(chunk_bytes, dtype=torch.int64) % 251).to(torch.uint8)
        buf.copy_(pattern.to(torch_device_type))
        torch_dev.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.WRITE)

        buf.zero_()
        torch_dev.synchronize()
        ctx.transfer_async(mem_obj, buf, SlabDirection.READ)
        torch_dev.synchronize()

        assert torch.equal(buf.cpu(), pattern)
    finally:
        ctx.close()


def test_l1_context_routes_by_owner_and_limits_process_slabs(monkeypatch, backend):
    """Owner routing must reject a second slab before native registration."""
    monkeypatch.setattr(gds_context, "create_backend", lambda name: backend)
    backend.name = "test"
    backend.open_slab.return_value.path = "/test/slab"
    config = GdsL1Config("/test", 4096)
    owner = 9001
    gds_context.initialize_l1_gds_context(owner, config)
    try:
        context = gds_context.get_l1_gds_context(owner)
        register = Mock()
        deregister = Mock()
        monkeypatch.setattr(context, "register_gpu_buffer", register)
        monkeypatch.setattr(context, "deregister_gpu_buffer", deregister)
        tensor = torch.empty(4096, dtype=torch.uint8)
        gds_context.register_gds_gpu_buffer(tensor)
        gds_context.deregister_gds_gpu_buffer(tensor)
        register.assert_called_once_with(tensor, slot_bytes=None)
        deregister.assert_called_once_with(tensor, slot_bytes=None)
        with pytest.raises(ValueError, match="one GDS"):
            gds_context.initialize_l1_gds_context(owner + 1, config)
        with pytest.raises(ValueError, match="one GDS"):
            initialize_gds_context(config)
        with pytest.raises(ValueError, match="Unknown GDS L1 owner"):
            gds_context.get_l1_gds_context(owner + 1)
    finally:
        gds_context.close_l1_gds_context(owner)
    backend.close_driver.assert_called_once()
    with pytest.raises(ValueError, match="Unknown GDS L1 owner"):
        gds_context.get_l1_gds_context(owner)
    gds_context.initialize_l1_gds_context(owner + 1, config)
    gds_context.close_l1_gds_context(owner + 1)


@pytest.fixture
def registered_context(
    monkeypatch: pytest.MonkeyPatch, backend: Mock
) -> Iterator[tuple[GDSContext, Mock, torch.Tensor]]:
    """Use real async descriptors with completion driven by each fake stream."""
    backend.name = "cufile"
    handle = Mock(spec=GDSHandle, path="/test/slab")
    pending: dict[int, list[Submission]] = {}
    handle.submissions = []

    def submit(base: int, size: int, file: int, offset: int, stream: int) -> Submission:
        submission = Submission(size, file, offset)
        pending.setdefault(stream, []).append(submission)
        handle.submissions.append(submission)
        return submission

    def complete(device: object, stream: SimpleNamespace) -> None:
        for submission in pending.pop(stream.raw_handle, []):
            submission.result.value = submission.size.value

    handle.read_async.side_effect = handle.write_async.side_effect = submit
    backend.open_slab.return_value = handle
    _use_fake_stream(monkeypatch, 1)
    monkeypatch.setattr(gds_context.platform_stream, "synchronize_stream", complete)
    monkeypatch.setattr(
        gds_context.platform_stream, "synchronize_device", lambda _: None
    )
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 64 << 20, io_concurrency=4))
    buffer = torch.empty(32 << 20, dtype=torch.uint8)
    context.register_gpu_buffer(buffer)
    yield context, handle, buffer
    context.close()


def _memory(size: int, offset: int = 0) -> GDSMemoryObject:
    """Build an object whose logical bytes and slab offset are independently checked."""
    return GDSMemoryObject(
        MemoryObjMetadata(torch.Size([size]), torch.uint8, offset, size, 0)
    )


@pytest.mark.parametrize("direction", list(SlabDirection))
def test_async_region_boundary_and_tail(
    registered_context: tuple[GDSContext, Mock, torch.Tensor], direction: SlabDirection
) -> None:
    """A crossing transfer preserves logical tail, file offset and registered base."""
    context, handle, buffer = registered_context
    start = (16 << 20) - 4096
    context.transfer_async(_memory(8193, 12288), buffer[start:], direction)
    operation = (
        handle.read_async if direction is SlabDirection.READ else handle.write_async
    )
    assert [call.args for call in operation.call_args_list] == [
        (buffer.data_ptr(), 4096, 12288, start, 1),
        (buffer.data_ptr() + (16 << 20), 4097, 16384, 0, 1),
    ]
    context.synchronize_and_check(buffer.device, _fake_stream(1))


@pytest.mark.parametrize("completed", [4096, 2048, -1])
def test_async_validation_waits_and_retires_results(
    registered_context: tuple[GDSContext, Mock, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
    completed: int,
) -> None:
    """Inspect bytes after completion and retire failures before later I/O."""
    context, handle, buffer = registered_context
    submission = Submission(4096, 8192, 0)
    handle.read_async.side_effect = None
    handle.read_async.return_value = submission
    context.transfer_async(_memory(4096, 8192), buffer, SlabDirection.READ)
    monkeypatch.setattr(
        gds_context.platform_stream,
        "synchronize_stream",
        lambda *_: setattr(submission.result, "value", completed),
    )
    if completed == 4096:
        context.synchronize_and_check(buffer.device, _fake_stream(1))
    else:
        with pytest.raises(RuntimeError, match=f"completed {completed}"):
            context.synchronize_and_check(buffer.device, _fake_stream(1))
    context.synchronize_and_check(buffer.device, _fake_stream(1))


def test_auxiliary_streams_register_once_and_complete_independently(
    registered_context: tuple[GDSContext, Mock, torch.Tensor],
    backend: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four extra streams share registrations but retain independent result queues."""
    context, handle, buffer = registered_context
    region_count = backend.register_buffer.call_count
    with context.io_scope():
        for raw in range(2, 6):
            context.register_stream(buffer.device, _fake_stream(raw))
            context.register_stream(buffer.device, _fake_stream(raw))
        for raw in (2, 3):
            _use_fake_stream(monkeypatch, raw)
            context.transfer_async(_memory(4096), buffer, SlabDirection.READ)
        context.synchronize_and_check(buffer.device, _fake_stream(2))
        assert [sub.bytes_done for sub in handle.submissions] == [4096, 0]
        context.synchronize_and_check(buffer.device, _fake_stream(3))
        assert [sub.bytes_done for sub in handle.submissions] == [4096, 4096]
        context.deregister_stream(buffer.device, _fake_stream(2))
        context.deregister_stream(buffer.device, _fake_stream(2))
    assert [call.args[0] for call in backend.register_stream.call_args_list] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert backend.register_buffer.call_count == region_count
    backend.deregister_stream.assert_called_once_with(2)
    assert [call.args[-1] for call in handle.read_async.call_args_list] == [2, 3]


def test_staging_budget_uses_total_allocation(
    registered_context: tuple[GDSContext, Mock, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured limit covers all four slots and every object group."""
    context, _, _ = registered_context
    monkeypatch.setattr(gds_context, "get_gds_context", lambda: context)
    gds_context.validate_gds_staging_size(128 << 20)
    with pytest.raises(ValueError, match="staging needs"):
        gds_context.validate_gds_staging_size((128 << 20) + 1)


def test_async_scope_prevents_slab_teardown(
    registered_context: tuple[GDSContext, Mock, torch.Tensor],
    backend: Mock,
) -> None:
    """An active group keeps the native handle and registrations alive until drain."""
    context, handle, buffer = registered_context
    with ThreadPoolExecutor(max_workers=1) as executor:
        with context.io_scope():
            context.transfer_async(_memory(4096), buffer, SlabDirection.WRITE)
            closing = executor.submit(context.close)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
            handle.close.assert_not_called()
            backend.deregister_buffer.assert_not_called()
            context.synchronize_and_check(buffer.device, _fake_stream(1))
        closing.result(timeout=5)
    handle.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        with context.io_scope():
            pytest.fail("Closed slab admitted another group")


def test_non_cufile_backend_retains_default_async_path(backend: Mock) -> None:
    """The opt-in does not change unrelated backend stream contracts."""
    backend.name = "test"
    backend.open_slab.return_value.path = "/test"
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 4096, io_concurrency=4))
    assert context.io_concurrency == 0
    context.close()

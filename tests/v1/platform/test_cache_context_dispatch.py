# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the device-type-driven dispatch in
``lmcache.v1.platform.cache_context.create_cache_context``.

The facade routes by the ``torch.device.type`` reported by the
wrappers' ``to_tensor()`` output and delegates to the matching
:class:`~lmcache.v1.platform.base.device_spec.DeviceSpec` instance's
``create_cache_context`` hook. These tests exercise that dispatch
without touching CUDA or the real ``GPUCacheContext`` /
``CPUCacheContext`` constructors -- they ``monkeypatch`` the hook on
the registered ``DeviceSpec`` so the test stays platform-agnostic. A separate
CPU-only regression exercises the CUDA context's public GDS lease and event
reuse with device calls mocked.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.v1.platform.devices.cuda.cache_context as cuda_cache_context


class _FakeWrapper:
    """Minimal stand-in for a KV-cache IPC wrapper.

    ``create_cache_context`` only ever reads ``to_tensor().device.type``
    from the wrappers it receives, so a 0-byte tensor on the requested
    device is enough.  For device types torch itself does not know
    about (``"fakedev"``), we return a duck-typed stub instead so the
    unregistered-device branch can still be exercised.
    """

    def __init__(self, device_type: str) -> None:
        self._device_type = device_type

    def to_tensor(self) -> torch.Tensor:
        try:
            return torch.empty(0, device=torch.device(self._device_type))
        except (RuntimeError, TypeError):
            # ``torch.device("fakedev")`` raises -- return a stub that
            # only implements the ``.device.type`` protocol used by
            # ``_detect_device_type``.
            device_type = self._device_type

            class _StubDevice:
                type = device_type

            class _StubTensor:
                device = _StubDevice()

            return _StubTensor()  # type: ignore[return-value]


class _FakeContext(BaseCacheContext):
    """Bare-bones ``BaseCacheContext`` subclass used as a registry stub.

    All abstract members are no-ops: the test never invokes them; it
    only checks that the right class is instantiated with the
    forwarded arguments.
    """

    device_type = "fake"

    def __init__(
        self,
        kv_caches: Any,
        lmcache_tokens_per_chunk: int,
        layout_hints: Any,
        engine_group_infos: Any,
        engine_type: Any,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        # Skip ``BaseCacheContext.__init__`` -- it requires real
        # KVLayerGroupsManager / shape descriptors that are out of
        # scope for the dispatch test.
        self.kv_caches = kv_caches
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self.layout_hints = layout_hints
        self.engine_group_infos = engine_group_infos
        self.engine_type = engine_type
        self.separate_object_groups = separate_object_groups
        self.full_sw_kv = full_sw_kv

    # ------------------------------------------------------------------
    # Abstract stubs -- never called from these tests.
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover - never invoked
        return torch.float32

    @property
    def stream(self) -> Any:  # pragma: no cover - never invoked
        return None

    @property
    def cupy_stream(self) -> Any:  # pragma: no cover - never invoked
        return None

    @property
    def max_batch_size(self) -> int:  # pragma: no cover - never invoked
        return 0

    def close(self) -> None:  # pragma: no cover - never invoked
        return None

    def get_kernel_group_kv_pointers(
        self, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:  # pragma: no cover
        return torch.empty(0)

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> Any:  # pragma: no cover
        return torch.Size(()), torch.float32

    def cache_size_per_token(self) -> int:  # pragma: no cover
        return 0


class _FakeCPUContext(_FakeContext):
    device_type = "cpu"


class _FakeCUDAContext(_FakeContext):
    device_type = torch_device_type


def _patch_hook(
    monkeypatch: pytest.MonkeyPatch, device_type: str, factory: type
) -> None:
    """Install *factory* as the ``create_cache_context`` hook on the
    ``DeviceSpec`` registered for *device_type*."""
    spec = get_device_spec(device_type)
    assert spec is not None, f"DeviceSpec for {device_type!r} not registered"
    monkeypatch.setattr(spec, "create_cache_context", factory)


def test_dispatches_by_cpu_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrappers reporting ``cpu`` tensors must yield the cpu-registered
    class."""
    _patch_hook(monkeypatch, "cpu", _FakeCPUContext)

    wrappers: List[_FakeWrapper] = [_FakeWrapper("cpu"), _FakeWrapper("cpu")]
    ctx = create_cache_context(wrappers)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCPUContext)
    assert ctx.kv_caches is wrappers


@pytest.mark.cuda
def test_dispatches_by_cuda_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrappers reporting a non-cpu device type must route to the
    matching registered class -- no isinstance branching."""
    _patch_hook(monkeypatch, torch_device_type, _FakeCUDAContext)

    wrappers = [_FakeWrapper(torch_device_type)]
    ctx = create_cache_context(wrappers, lmcache_tokens_per_chunk=128)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCUDAContext)
    assert ctx.lmcache_tokens_per_chunk == 128


def test_empty_kv_caches_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        create_cache_context([])


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"Requires available {torch_device_type} runtime",
)
def test_mixed_device_types_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross-device batches are unsupported and must fail loudly."""
    _patch_hook(monkeypatch, "cpu", _FakeCPUContext)
    _patch_hook(monkeypatch, torch_device_type, _FakeCUDAContext)

    wrappers = [_FakeWrapper("cpu"), _FakeWrapper(torch_device_type)]
    with pytest.raises(ValueError, match="share one"):
        create_cache_context(wrappers)  # type: ignore[arg-type]


def test_unregistered_device_type_raises() -> None:
    """An unknown device type is a hard failure with a clear hint."""
    wrappers = [_FakeWrapper("fakedev")]
    with pytest.raises(ValueError, match="No cache-context class"):
        create_cache_context(wrappers)  # type: ignore[arg-type]


def test_gds_context_event_reuse_and_close_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close cannot unmap leased buffers; each transfer re-records one input event."""
    runtime, stream, ready, wrapper = (MagicMock() for _ in range(4))
    stream.cuda_stream = 0
    runtime.Stream.return_value = stream
    cupy = ModuleType("cupy")
    monkeypatch.setattr(
        cupy,
        "cuda",
        SimpleNamespace(ExternalStream=lambda *_: MagicMock()),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(cuda_cache_context, "torch_dev", runtime)
    event_factory = MagicMock(return_value=ready)
    monkeypatch.setattr(cuda_cache_context.torch.cuda, "Event", event_factory)
    slot_streams = [MagicMock() for _ in range(4)]
    monkeypatch.setattr(
        cuda_cache_context.torch.cuda, "Stream", MagicMock(side_effect=slot_streams)
    )
    register = MagicMock()
    monkeypatch.setattr(cuda_cache_context, "register_gds_gpu_buffer", register)
    teardown: list[str] = []

    def drain_stream(device: object, slot_stream: MagicMock) -> None:
        slot_stream.synchronize()
        teardown.append("stream")

    deregister_stream = MagicMock(side_effect=drain_stream)
    monkeypatch.setattr(cuda_cache_context, "deregister_gds_stream", deregister_stream)
    deregister = MagicMock(
        side_effect=lambda _, slot_bytes=None: teardown.append("buffer")
    )
    monkeypatch.setattr(cuda_cache_context, "deregister_gds_gpu_buffer", deregister)
    wrapper.close.side_effect = lambda: teardown.append("wrapper")
    wrapper.to_tensor.return_value = torch.zeros((2, 4, 2, 1, 2), dtype=torch.float16)
    context = cuda_cache_context.GPUCacheContext(
        [wrapper], lmcache_tokens_per_chunk=4, layout_hints={"kv_layout": "NHD"}
    )
    for _ in range(2):
        with context.gds_transfer_scope():
            assert context.record_gds_transfer_ready() is ready
            ready.synchronize()
            for index, slot_stream in enumerate(slot_streams):
                assert context.get_gds_transfer_stream(index) is slot_stream
    event_factory.assert_called_once()
    assert ready.record.call_count == 2
    leased, release = threading.Event(), threading.Event()

    def hold() -> None:
        with context.gds_transfer_scope():
            leased.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        worker = executor.submit(hold)
        assert leased.wait(5)
        closer = executor.submit(context.close)
        try:
            with pytest.raises(TimeoutError):
                closer.result(timeout=0.05)
            deregister.assert_not_called()
            wrapper.close.assert_not_called()
        finally:
            release.set()
        worker.result(timeout=5)
        closer.result(timeout=5)
    deregister.assert_called_once()
    assert (
        register.call_args.kwargs["slot_bytes"]
        == deregister.call_args.kwargs["slot_bytes"]
    )
    assert deregister_stream.call_count == 4
    for slot_stream in slot_streams:
        slot_stream.synchronize.assert_called_once()
    assert teardown == ["stream"] * 4 + ["buffer", "wrapper"]
    wrapper.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        with context.gds_transfer_scope():
            pytest.fail("Closed context admitted worker work")

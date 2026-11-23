# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the device-type-driven dispatch in
``lmcache.v1.multiprocess.cache_context.factory.create_cache_context``.

The facade routes by the ``torch.device.type`` reported by the
wrappers' ``to_tensor()`` output, resolves the matching
:class:`~lmcache.v1.platform.base.device_spec.DeviceSpec`, and then
either instantiates the in-tree context class of a built-in backend or
calls the spec's ``create_cache_context`` hook (external device
plugins). These tests exercise that dispatch without touching CUDA or
the real context constructors -- they ``monkeypatch`` the context class
in its server-side module so the test stays platform-agnostic.
"""

# Standard
from typing import Any, List
import importlib

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.cache_context.base import BaseCacheContext
from lmcache.v1.multiprocess.cache_context.factory import create_cache_context
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.device_spec import DeviceSpec
import lmcache.v1.multiprocess.cache_context.factory as factory_module


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

    def get_temp_staging_buffer(
        self, num_bytes: int
    ) -> torch.Tensor | None:  # pragma: no cover
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


# Built-in ``backend_name`` -> (module under
# ``lmcache.v1.multiprocess.cache_context``, context class name).
_BUILTIN_CONTEXTS = {
    "cpu": ("cpu", "CPUCacheContext"),
    "cuda": ("cuda", "GPUCacheContext"),
    "rocm": ("cuda", "GPUCacheContext"),
    "musa": ("musa", "MUSACacheContext"),
    "npu": ("npu", "NpuCacheContext"),
}


class _StubSpec(DeviceSpec):
    """``DeviceSpec`` with a chosen ``backend_name`` and a recording hook."""

    def __init__(self, backend_name: str) -> None:
        self._backend_name = backend_name
        self.hook_calls: List[tuple] = []

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def create_cache_context(self, *args: Any, **kwargs: Any) -> Any:
        self.hook_calls.append((args, kwargs))
        return _FakeContext(*args, **kwargs)


def _patch_builtin(
    monkeypatch: pytest.MonkeyPatch, backend_name: str, context_cls: type
) -> None:
    """Replace the in-tree context class of built-in *backend_name*
    with *context_cls*."""
    module_name, class_name = _BUILTIN_CONTEXTS[backend_name]
    module = importlib.import_module(
        f"lmcache.v1.multiprocess.cache_context.{module_name}"
    )
    monkeypatch.setattr(module, class_name, context_cls)


def test_dispatches_by_cpu_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrappers reporting ``cpu`` tensors must yield the cpu-registered
    class."""
    _patch_builtin(monkeypatch, "cpu", _FakeCPUContext)

    wrappers: List[_FakeWrapper] = [_FakeWrapper("cpu"), _FakeWrapper("cpu")]
    ctx = create_cache_context(wrappers)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCPUContext)
    assert ctx.kv_caches is wrappers


@pytest.mark.cuda
def test_dispatches_by_cuda_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrappers reporting a non-cpu device type must route to the
    matching registered class -- no isinstance branching."""
    spec = get_device_spec(torch_device_type)
    assert spec is not None, f"DeviceSpec for {torch_device_type!r} not registered"
    _patch_builtin(monkeypatch, spec.backend_name, _FakeCUDAContext)

    wrappers = [_FakeWrapper(torch_device_type)]
    ctx = create_cache_context(wrappers, lmcache_tokens_per_chunk=128)  # type: ignore[arg-type]

    assert isinstance(ctx, _FakeCUDAContext)
    assert ctx.lmcache_tokens_per_chunk == 128


@pytest.mark.parametrize("backend_name", sorted(_BUILTIN_CONTEXTS))
def test_builtin_backend_bypasses_spec_hook(
    monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Built-in backends get their in-tree context class straight from
    the factory, with every argument forwarded and no hook call."""
    spec = _StubSpec(backend_name)
    monkeypatch.setattr(factory_module, "get_device_spec", lambda _: spec)
    _patch_builtin(monkeypatch, backend_name, _FakeCPUContext)

    wrappers = [_FakeWrapper("cpu")]
    ctx = create_cache_context(
        wrappers,  # type: ignore[arg-type]
        128,
        None,
        (),
        separate_object_groups=False,
        full_sw_kv=True,
    )

    assert isinstance(ctx, _FakeCPUContext)
    assert spec.hook_calls == []
    assert ctx.kv_caches is wrappers
    assert ctx.lmcache_tokens_per_chunk == 128
    assert ctx.separate_object_groups is False
    assert ctx.full_sw_kv is True


def test_external_spec_uses_create_cache_context_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec that is not a built-in backend (an external device plugin)
    is served by its own ``create_cache_context`` hook."""
    spec = _StubSpec("external")
    monkeypatch.setattr(factory_module, "get_device_spec", lambda _: spec)

    wrappers = [_FakeWrapper("cpu")]
    ctx = create_cache_context(wrappers, lmcache_tokens_per_chunk=128)  # type: ignore[arg-type]

    assert type(ctx) is _FakeContext
    assert len(spec.hook_calls) == 1
    assert ctx.kv_caches is wrappers
    assert ctx.lmcache_tokens_per_chunk == 128


def test_empty_kv_caches_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        create_cache_context([])


@pytest.mark.skipif(
    not torch_dev.is_available(),
    reason=f"Requires available {torch_device_type} runtime",
)
def test_mixed_device_types_raises() -> None:
    """Cross-device batches are unsupported and must fail loudly."""
    wrappers = [_FakeWrapper("cpu"), _FakeWrapper(torch_device_type)]
    with pytest.raises(ValueError, match="share one"):
        create_cache_context(wrappers)  # type: ignore[arg-type]


def test_unregistered_device_type_raises() -> None:
    """An unknown device type is a hard failure with a clear hint."""
    wrappers = [_FakeWrapper("fakedev")]
    with pytest.raises(ValueError, match="No cache-context class"):
        create_cache_context(wrappers)  # type: ignore[arg-type]

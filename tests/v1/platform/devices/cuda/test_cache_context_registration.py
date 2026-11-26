# SPDX-License-Identifier: Apache-2.0
"""CUDA registration lifecycle checks using CPU buffers and stream doubles."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.common.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform import get_device_spec
from lmcache.v1.platform.base.ipc_wrapper import KVCache
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.platform.devices.cpu.stub_cpu_device import StubCPUDevice, StubStream
from lmcache.v1.platform.devices.cuda import cache_context as cuda_context
import lmcache.lmcache_native as lmcache_native


@pytest.fixture
def context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[cuda_context.GPUCacheContext, list[str], list[bool]]:
    calls: list[str] = []
    active = [False]

    class _Stream(StubStream):
        def synchronize(self) -> None:
            calls.append("synchronize")

    expected_stream = _Stream()

    class _Device(StubCPUDevice):
        @contextmanager
        def stream(self, stream: StubStream | None = None) -> Iterator[None]:
            assert stream is expected_stream
            active[0] = True
            try:
                yield
            finally:
                active[0] = False

    class _Wrapper:
        device_type = "cpu"

        def close(self) -> None:
            calls.append("wrapper.close")

    manager = KVLayerGroupsManager(
        [torch.zeros(2, 4, 16, 2, 8)],
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS],
        lmcache_tokens_per_chunk=16,
    )

    class _Context(cuda_context.GPUCacheContext):
        def _init_impl(self, *args: Any, **kwargs: Any) -> None:
            self.cuda_stream_ = expected_stream
            self._temp_buffer = cuda_context._TempGPUBuffer(
                manager, 16, torch.device("cpu")
            )

    monkeypatch.setattr(cuda_context, "torch_dev", _Device("cpu"))
    return _Context(cast(KVCache, [_Wrapper()])), calls, active


def test_register_and_close_use_the_full_buffer_on_the_context_stream(
    context: tuple[cuda_context.GPUCacheContext, list[str], list[bool]],
) -> None:
    cache_context, calls, active = context
    buffers: list[torch.Tensor] = []

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            assert active[0]
            buffers.append(buffer)
            calls.append("register")

        def deregister(self, buffer: torch.Tensor) -> None:
            assert active[0]
            assert buffer is buffers[0]
            calls.append("deregister")

    cache_context.register_staging_buffer(_Registration())
    cache_context.close()
    assert calls == ["register", "synchronize", "deregister", "wrapper.close"]
    assert buffers[0].numel() > 0
    assert buffers[0].nbytes == buffers[0].untyped_storage().nbytes()


def test_factory_closes_ipc_owners_after_registration_failure(
    context: tuple[cuda_context.GPUCacheContext, list[str], list[bool]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_context, calls, active = context

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            assert active[0]
            calls.append("register.failed")
            raise RuntimeError("registration failed")

        def deregister(self, buffer: torch.Tensor) -> None:
            pytest.fail("A failed registration must not be deregistered")

    class _Wrapper:
        device_type = "cpu"

    spec = get_device_spec("cpu")
    assert spec is not None
    monkeypatch.setattr(
        spec, "create_cache_context", lambda *args, **kwargs: cache_context
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        create_cache_context(
            cast(KVCache, [_Wrapper()]), buffer_registration=_Registration()
        )
    assert calls == ["register.failed", "synchronize", "wrapper.close"]

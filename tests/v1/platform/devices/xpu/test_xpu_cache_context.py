# SPDX-License-Identifier: Apache-2.0
"""Tests for the XPU cache context's BaseCacheContext contract."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.platform.cache_context import create_cache_context
from lmcache.v1.platform.devices.xpu.cache_context import XPUCacheContext


def test_implements_every_abstract_method() -> None:
    """A base-class method left unimplemented makes the context uninstantiable."""
    assert not XPUCacheContext.__abstractmethods__


def test_temp_staging_buffer_is_prefix_of_staging_buffer() -> None:
    """The region starts at the buffer's first byte; too large a request is None."""
    ctx = object.__new__(XPUCacheContext)
    staging = torch.arange(16, dtype=torch.uint8)
    ctx._temp_buffer = cast(Any, SimpleNamespace(buffer=staging))

    region = ctx.get_temp_staging_buffer(10)
    assert region is not None
    assert region.data_ptr() == staging.data_ptr()
    assert region.dtype == torch.uint8
    assert region.numel() == 10

    whole = ctx.get_temp_staging_buffer(16)
    assert whole is not None and whole.numel() == 16
    assert ctx.get_temp_staging_buffer(17) is None


class _FakeIPCWrapper:
    """Hands back a local XPU tensor; the context only calls ``to_tensor``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        pass


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires an XPU device",
)
def test_create_cache_context_forwards_object_group_transfer() -> None:
    """The server passes its copy callback at construction; the base
    ``transfer_object_group`` must reach it on XPU."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    def transfer(ctx: Any, block_ids: Any, memory_objs: Any, **kw: Any) -> None:
        calls.append((ctx, kw))

    kv = [
        _FakeIPCWrapper(
            torch.zeros(2, 4, 16, 2, 64, dtype=torch.bfloat16, device="xpu")
        )
        for _ in range(2)
    ]
    ctx = create_cache_context(
        cast(Any, kv), 16, object_group_transfer=cast(Any, transfer)
    )
    try:
        assert isinstance(ctx, XPUCacheContext)
        ctx.transfer_object_group(
            [],
            [],
            object_group_id=0,
            batch_size=1,
            skip_first_n_tokens=0,
            direction=cast(Any, None),
            transfer_key="k",
        )
        assert len(calls) == 1
        assert calls[0][0] is ctx
        assert calls[0][1]["transfer_key"] == "k"
    finally:
        ctx.close()


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="requires an XPU device",
)
def test_create_cache_context_registers_full_staging_buffer() -> None:
    """The server's registration gets the whole staging buffer once, and
    ``close`` deregisters that same buffer."""
    calls: list[tuple[str, torch.Tensor]] = []

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            calls.append(("register", buffer))

        def deregister(self, buffer: torch.Tensor) -> None:
            calls.append(("deregister", buffer))

    kv = [
        _FakeIPCWrapper(
            torch.zeros(2, 4, 16, 2, 64, dtype=torch.bfloat16, device="xpu")
        )
        for _ in range(2)
    ]
    ctx = create_cache_context(cast(Any, kv), 16, buffer_registration=_Registration())
    assert isinstance(ctx, XPUCacheContext)
    staging = ctx._temp_buffer.buffer
    assert [name for name, _ in calls] == ["register"]
    ctx.close()
    assert [name for name, _ in calls] == ["register", "deregister"]
    assert all(buf is staging for _, buf in calls)

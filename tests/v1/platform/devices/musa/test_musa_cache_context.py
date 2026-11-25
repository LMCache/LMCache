# SPDX-License-Identifier: Apache-2.0

"""Tests for MUSA cache-context IPC owner lifetime."""

# Standard
from types import SimpleNamespace
from typing import Any, cast
import contextlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.buffer_registration import BufferRegistration
from lmcache.v1.platform.devices.musa.cache_context import MUSACacheContext


def test_close_synchronizes_before_releasing_ipc_owners() -> None:
    """Context close waits for transfers, releases owners, and is idempotent."""
    calls: list[str] = []

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    class _TestContext(MUSACacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._buffer_registration = None
            self._ipc_wrappers = (  # type: ignore[assignment]
                _Wrapper("first"),
                _Wrapper("second"),
            )

    context = _TestContext()

    context.close()
    context.close()

    assert calls == ["synchronize", "first", "second"]


def test_close_deregisters_gds_staging_buffer(monkeypatch: Any) -> None:
    """Close deregisters the staging buffer from GDS before IPC release."""
    # First Party
    import lmcache.v1.platform.devices.musa.cache_context as musa_ctx_mod

    calls: list[str] = []
    staging = torch.empty(1, dtype=torch.uint8)

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def close(self) -> None:
            calls.append("wrapper.close")

    class _TestContext(MUSACacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = (_Wrapper(),)  # type: ignore[assignment]
            self._buffer_registration = None
            self._temp_buffer = cast(Any, SimpleNamespace(buffer=staging))

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            assert buffer is staging

        def deregister(self, buffer: torch.Tensor) -> None:
            assert buffer is staging
            calls.append("deregister")

    # torch_dev.stream must be usable as a context manager for the
    # deregistration block.
    monkeypatch.setattr(
        musa_ctx_mod.torch_dev, "stream", lambda s: contextlib.nullcontext()
    )

    context = _TestContext()
    context.register_staging_buffer(_Registration())
    context.close()

    # Order: sync first, then GDS deregister, then IPC wrapper release.
    assert calls == ["synchronize", "deregister", "wrapper.close"]


def test_close_releases_ipc_wrappers_when_gds_deregister_fails(
    monkeypatch: Any,
) -> None:
    """Cleanup still releases IPC owners when GDS deregistration fails."""
    # First Party
    import lmcache.v1.platform.devices.musa.cache_context as musa_ctx_mod

    calls: list[str] = []
    staging = torch.empty(1, dtype=torch.uint8)

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def close(self) -> None:
            calls.append("wrapper.close")

    class _TestContext(MUSACacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = (_Wrapper(),)  # type: ignore[assignment]
            self._buffer_registration = None
            self._temp_buffer = cast(Any, SimpleNamespace(buffer=staging))

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            pass

        def deregister(self, buffer: torch.Tensor) -> None:
            assert buffer is staging
            calls.append("deregister")
            raise RuntimeError("deregister failed")

    monkeypatch.setattr(
        musa_ctx_mod.torch_dev, "stream", lambda s: contextlib.nullcontext()
    )

    context = _TestContext()
    context.register_staging_buffer(_Registration())
    with pytest.raises(RuntimeError, match="deregister failed"):
        context.close()

    assert calls == ["synchronize", "deregister", "wrapper.close"]
    assert context._ipc_wrappers == ()


def test_close_skips_unowned_gds_registration_after_register_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed registration must not trigger a later deregistration."""
    calls: list[str] = []
    staging = torch.empty(1, dtype=torch.uint8)

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def close(self) -> None:
            calls.append("wrapper.close")

    class _TestContext(MUSACacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = (_Wrapper(),)  # type: ignore[assignment]
            self._buffer_registration = None
            self._temp_buffer = cast(Any, SimpleNamespace(buffer=staging))

    class _Registration(BufferRegistration):
        def register(self, buffer: torch.Tensor) -> None:
            raise RuntimeError("register failed")

        def deregister(self, buffer: torch.Tensor) -> None:
            pytest.fail("A failed registration must not be deregistered")

    # First Party
    import lmcache.v1.platform.devices.musa.cache_context as musa_ctx_mod

    monkeypatch.setattr(
        musa_ctx_mod.torch_dev, "stream", lambda s: contextlib.nullcontext()
    )

    context = _TestContext()
    with pytest.raises(RuntimeError, match="register failed"):
        context.register_staging_buffer(_Registration())
    context.close()

    assert calls == ["synchronize", "wrapper.close"]

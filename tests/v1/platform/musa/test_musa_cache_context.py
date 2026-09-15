# SPDX-License-Identifier: Apache-2.0

"""Tests for MUSA cache-context IPC owner lifetime."""

# Standard
from types import SimpleNamespace
from typing import Any

# First Party
from lmcache.v1.platform.musa.cache_context import MUSACacheContext


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
    import lmcache.v1.platform.musa.cache_context as musa_ctx_mod

    calls: list[str] = []

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def close(self) -> None:
            calls.append("wrapper.close")

    class _FakeGDSContext:
        def deregister_gpu_buffer(self, buffer: Any) -> None:
            calls.append(f"deregister:{buffer}")

    class _TestContext(MUSACacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = (_Wrapper(),)  # type: ignore[assignment]
            self._temp_buffer = SimpleNamespace(  # type: ignore[assignment]
                buffer="STAGING"
            )

    monkeypatch.setattr(musa_ctx_mod, "get_gds_context", lambda: _FakeGDSContext())
    # torch_dev.stream must be usable as a context manager for the
    # deregistration block.
    # Standard
    import contextlib

    monkeypatch.setattr(
        musa_ctx_mod.torch_dev, "stream", lambda s: contextlib.nullcontext()
    )

    context = _TestContext()
    context.close()

    # Order: sync first, then GDS deregister, then IPC wrapper release.
    assert calls == ["synchronize", "deregister:STAGING", "wrapper.close"]

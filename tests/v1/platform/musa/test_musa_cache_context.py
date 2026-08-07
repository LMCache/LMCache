# SPDX-License-Identifier: Apache-2.0

"""Tests for MUSA cache-context IPC owner lifetime."""

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

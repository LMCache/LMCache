# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import (
    async_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


@dataclass
class _FakeStoreContext:
    """Minimal engine-driven context for async store tests."""

    commit_impl: Callable[[list[torch.Tensor]], bool]
    prepare_result: tuple[list[torch.Tensor], list[int]] | None = None

    def __post_init__(self) -> None:
        self.layout_desc = SimpleNamespace(
            shapes=[torch.Size([2, 1, 1, 1])], dtypes=[torch.float32]
        )

    def prepare_store(
        self, _key: object, _instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        return self.prepare_result

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def close(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, gate: threading.Event):
        self._gate = gate

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        self._gate.wait(timeout=2)

    def query(self) -> bool:
        return self._gate.is_set()

    def ipc_handle(self) -> object:
        return object()


class _FakeTorchDev:
    def __init__(self, gather_gate: threading.Event):
        self._stream = object()
        self._gather_gate = gather_gate

    def Stream(self) -> object:
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def current_stream(self) -> object:
        return self._stream

    def Event(self, interprocess: bool = False) -> _FakeEvent:
        return _FakeEvent(self._gather_gate)

    def synchronize(self) -> None:
        pass


def _install_fake_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_in_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        out = kwargs.get("out")
        if out is None:
            return [torch.ones(1)]
        assert isinstance(out, list)
        for tensor in out:
            tensor.fill_(1.0)
        return out

    # Patch gather in both modules so either path is exercised.
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", _gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)


def _new_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gather_gate: threading.Event,
    commit_impl: Callable[[list[torch.Tensor]], bool],
    max_inflight: int = 8,
) -> AsyncEngineDrivenTransferContext:
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=max_inflight)
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=commit_impl)  # type: ignore[assignment]
    )
    return ctx


def test_submit_store_returns_pending_future_until_gather_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not future.query()
    gather_gate.set()
    assert future.result(timeout=1) is True
    ctx.close()


def test_submit_store_commit_waits_for_gather_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    commit_called = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_called.set()
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not commit_called.wait(timeout=0.05)
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert commit_called.is_set()
    ctx.close()


def test_close_drains_inflight_async_store(monkeypatch: pytest.MonkeyPatch) -> None:
    gather_gate = threading.Event()
    commit_gate = threading.Event()
    gather_gate.set()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_gate.wait(timeout=2)
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    closed = threading.Event()

    def _close() -> None:
        ctx.close()
        closed.set()

    t = threading.Thread(target=_close, daemon=True)
    t.start()
    assert not closed.wait(timeout=0.05)
    commit_gate.set()
    t.join(timeout=1)
    assert closed.is_set()
    assert future.result(timeout=1) is True


def test_commit_failure_sets_false_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        raise RuntimeError("commit failed")

    log_exception = MagicMock()
    monkeypatch.setattr(async_engine_driven.logger, "exception", log_exception)
    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    gather_gate.set()
    assert future.result(timeout=1) is False
    log_exception.assert_called_once()
    ctx.close()


def test_flush_inflight_gathers_no_inflight_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    gather_gate.set()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    # No in-flight events: flush is a cheap no-op and must not raise.
    ctx.flush_inflight_gathers()
    ctx.close()


def test_sync_engine_driven_context_returns_resolved_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingTorchDev:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    fake = _RecordingTorchDev()
    monkeypatch.setattr(worker_transfer, "torch_dev", fake)
    _install_fake_gather(monkeypatch)
    ctx = EngineDrivenTransferContext()
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=lambda _c: True)  # type: ignore[assignment]
    )

    future = ctx.submit_store(
        "r1",
        object(),
        1,
        {"k": torch.zeros(1)},
        [[0]],
        _FakeEvent(threading.Event()),
        1,
    )

    # Sync path resolves inline.
    assert future.query()
    assert future.result(timeout=1) is True
    assert fake.synchronize_calls >= 1
    # flush_inflight_gathers is the inherited base no-op; must not raise.
    ctx.flush_inflight_gathers()
    ctx.close()


def test_sync_engine_driven_context_has_no_async_resources() -> None:
    ctx = EngineDrivenTransferContext()
    assert not hasattr(ctx, "_copy_stream")
    assert not hasattr(ctx, "_commit_executor")
    assert not hasattr(ctx, "_inflight_semaphore")
    # close() on an unregistered sync context must not raise.
    ctx.close()


def test_build_engine_driven_context_dispatches_on_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: True)
    # Avoid touching real stream/event primitives when instantiating the async
    # context returned by the capable branch.
    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _FakeTorchDev(threading.Event())
    )
    capable = worker_transfer._build_engine_driven_context()
    assert isinstance(capable, AsyncEngineDrivenTransferContext)
    capable.close()

    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: False)
    fallback = worker_transfer._build_engine_driven_context()
    assert isinstance(fallback, EngineDrivenTransferContext)
    assert not isinstance(fallback, AsyncEngineDrivenTransferContext)
    fallback.close()


def test_supports_async_primitives_false_without_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch_dev without Stream/Event is not async-capable.
    monkeypatch.setattr(worker_transfer, "torch_dev", object())
    assert worker_transfer._supports_async_primitives() is False

# SPDX-License-Identifier: Apache-2.0
"""The torch-baseline copy stays on the caller's stream.

The MP store path stages KV with a gather kernel on the cache context's
non-blocking stream and then asks ``lmcache_memcpy_async`` to copy the staging
buffer. When the compiled ``cuda_ops`` extension is missing, that call lands in
the Python fallback. A synchronous ``cudaMemcpy`` there runs on the legacy
default stream and does not wait for the gather, which stores stale KV; issuing
the copy asynchronously on the current stream keeps the ordering without ever
synchronising the device.

These tests pin that contract without needing a device.
"""

# Standard
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.torch_ops import mem_kernels


class _Runtime:
    """Stand-in for the CUDA/HIP runtime library."""

    def __init__(self, *, async_symbol: str | None = "cudaMemcpyAsync") -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        if async_symbol is not None:
            setattr(self, async_symbol, self._record)

    def _record(self, *args: Any) -> int:
        # Report the symbol that was used so tests can assert CUDA vs HIP.
        name = next(
            attr
            for attr in ("cudaMemcpyAsync", "hipMemcpyAsync")
            if hasattr(self, attr)
        )
        self.calls.append((name, args))
        return 0

    def cudaMemcpy(self, *args: Any) -> int:  # pragma: no cover - must not be used
        raise AssertionError("the fallback must not issue a synchronous copy")


def _d2h() -> None:
    mem_kernels.lmcache_memcpy_async(
        0x1000,
        0x2000,
        4096,
        mem_kernels.TransferDirection.D2H,
        host_buffer_offset=0,
        host_buffer_alignments=64 * 1024 * 1024,
    )


@pytest.fixture
def cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> _Runtime:
    runtime = _Runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *a, **k: type("S", (), {"cuda_stream": 7})(),
    )
    monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)
    return runtime


class TestStreamOrderedCopy:
    def test_copy_is_issued_on_the_current_stream(self, cuda_runtime: _Runtime) -> None:
        _d2h()

        (name, args) = cuda_runtime.calls[0]
        assert name == "cudaMemcpyAsync"
        # dest, src, nbytes, kind, stream
        assert args[4].value == 7

    def test_h2d_takes_the_same_path(self, cuda_runtime: _Runtime) -> None:
        mem_kernels.lmcache_memcpy_async(
            0x2000,
            0x1000,
            4096,
            mem_kernels.TransferDirection.H2D,
            host_buffer_offset=0,
            host_buffer_alignments=64 * 1024 * 1024,
        )

        assert cuda_runtime.calls[0][0] == "cudaMemcpyAsync"

    def test_never_synchronises_the_device(self, cuda_runtime: _Runtime) -> None:
        def unexpected(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("the fallback must not synchronise the device")

        _d2h()  # would raise if the fallback synchronised

    def test_hip_symbol_is_used_when_cuda_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _Runtime(async_symbol="hipMemcpyAsync")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)

        _d2h()

        (name, args) = runtime.calls[0]
        assert name == "hipMemcpyAsync"
        # Without an initialised device the copy stays on the default stream.
        # ctypes normalises a NULL pointer to None.
        assert args[4].value in (None, 0)

    def test_a_hip_only_runtime_still_takes_the_async_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ROCm exposes `hipMemcpy*` and no `cudaMemcpy`; the gate must not care.

        Gating on `cudaMemcpy` sent device pointers to the tensor copy on ROCm,
        which is the case this covers: the fake has no CUDA spelling at all.
        """

        class _HipOnly:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, ...]] = []

            def hipMemcpyAsync(self, *args: Any) -> int:
                self.calls.append(args)
                return 0

        runtime = _HipOnly()
        copied: list[tuple[int, int, int]] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *a, **k: type("S", (), {"cuda_stream": 9})(),
        )
        monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)
        monkeypatch.setattr(
            mem_kernels,
            "_copy_bytes_with_tensor",
            lambda dest, src, nbytes: copied.append((dest, src, nbytes)),
        )

        _d2h()

        assert copied == []
        assert len(runtime.calls) == 1
        assert runtime.calls[0][4].value == 9

    def test_tensor_copy_fallback_when_no_async_symbol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _Runtime(async_symbol=None)
        copied: list[tuple[int, int, int]] = []

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)
        monkeypatch.setattr(
            mem_kernels,
            "_copy_bytes_with_tensor",
            lambda dest, src, nbytes: copied.append((dest, src, nbytes)),
        )

        _d2h()

        assert copied == [(0x1000, 0x2000, 4096)]
        assert runtime.calls == []


class TestFailureReporting:
    def test_a_failed_copy_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = _Runtime()
        runtime.cudaMemcpyAsync = lambda *args: 1  # type: ignore[attr-defined]
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *a, **k: type("S", (), {"cuda_stream": 7})(),
        )
        monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: runtime)

        with pytest.raises(RuntimeError, match="error code 1"):
            _d2h()


def test_alignments_must_stay_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            4096,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=1000,
        )


def test_unsupported_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported direction"):
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            4096,
            cast(mem_kernels.TransferDirection, 99),
            host_buffer_offset=0,
            host_buffer_alignments=64 * 1024 * 1024,
        )


class TestPinChunkSplitting:
    """One async copy per host pin chunk; a cross-chunk copy fails (#6617)."""

    @staticmethod
    def _sizes(runtime: "_Runtime") -> list[int]:
        # arg 2 of each recorded call is the chunk size (c_size_t)
        return [int(call[1][2].value) for call in runtime.calls]

    def test_a_copy_inside_one_chunk_is_not_split(self, cuda_runtime: _Runtime) -> None:
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            64,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=64,
        )

        assert self._sizes(cuda_runtime) == [64]

    def test_a_copy_across_chunks_is_split_at_the_boundary(
        self, cuda_runtime: _Runtime
    ) -> None:
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            200,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=64,
        )

        assert self._sizes(cuda_runtime) == [64, 64, 64, 8]

    def test_an_offset_starts_at_the_next_boundary(
        self, cuda_runtime: _Runtime
    ) -> None:
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            200,
            mem_kernels.TransferDirection.H2D,
            host_buffer_offset=32,
            host_buffer_alignments=64,
        )

        assert self._sizes(cuda_runtime) == [32, 64, 64, 40]

    def test_every_chunk_uses_the_callers_stream(self, cuda_runtime: _Runtime) -> None:
        mem_kernels.lmcache_memcpy_async(
            0x1000,
            0x2000,
            128,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=64,
        )

        assert [call[1][4].value for call in cuda_runtime.calls] == [7, 7]


class TestCompletionOrdering:
    """A completion must not be published before the copy has landed."""

    @staticmethod
    def _record(monkeypatch, stream_ptr: int) -> list[tuple[str, object]]:
        # First Party
        from lmcache.v1.platform.torch_ops import completion_recorder

        completion_recorder.drain_recorded_completions()
        events: list[tuple[str, object]] = []

        def fake_wait(ptr: int) -> None:
            events.append(("wait", ptr))

        monkeypatch.setattr(completion_recorder, "_wait_for_stream", fake_wait)
        completion_recorder.record_completion_on_stream(
            stream_ptr, "finish_write", b"x"
        )
        events.append(
            ("drained", len(completion_recorder.drain_recorded_completions()))
        )
        return events

    def test_the_stream_is_waited_on_before_publishing(self, monkeypatch) -> None:
        events = self._record(monkeypatch, 0x7)

        assert events == [("wait", 0x7), ("drained", 1)]

    def test_a_missing_stream_handle_still_waits(self, monkeypatch) -> None:
        assert self._record(monkeypatch, 0) == [("wait", 0), ("drained", 1)]

# SPDX-License-Identifier: Apache-2.0
"""Tests for the cross-platform dispatcher in
``lmcache.v1.platform.stream``.

These tests exercise :func:`make_external_stream` without requiring
CUDA or cupy: the CUDA backend is stubbed with ``mock.patch`` so both
the hit and miss paths (and the CPU fallback) are covered.
"""

# Standard
from unittest import mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.cpu.stream import MockExternalStream
from lmcache.v1.platform.stream import make_external_stream


class TestMakeExternalStream:
    """Factory behavior, independent of cupy availability."""

    def test_returns_mock_when_cuda_unavailable(self):
        """Without CUDA the factory falls back to the mock."""

        class _FakeTorchStream:
            cuda_stream = 0xDEADBEEF

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            stream = make_external_stream(_FakeTorchStream(), 0)
        try:
            assert isinstance(stream, MockExternalStream)
            # Valid caller pointer is preserved so downstream C++ code
            # that unconditionally uses ``stream.ptr`` still sees a real
            # CUDA handle when one exists.
            assert stream.ptr == 0xDEADBEEF
        finally:
            stream._shutdown()

    def test_returns_mock_when_cuda_backend_declines(self):
        """Even with CUDA, a ``None`` from the CUDA backend falls through."""

        class _FakeTorchStream:
            cuda_stream = 0x1234

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch(
                "lmcache.v1.platform.cuda.stream.make_cuda_external_stream",
                return_value=None,
            ),
        ):
            stream = make_external_stream(_FakeTorchStream(), 0)
        try:
            assert isinstance(stream, MockExternalStream)
            assert stream.ptr == 0x1234
        finally:
            stream._shutdown()

    def test_survives_missing_cuda_stream_attr(self):
        """CPU-only torch streams without ``cuda_stream`` do not crash."""

        class _Broken:
            @property
            def cuda_stream(self):
                raise AttributeError("no CUDA on this platform")

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            stream = make_external_stream(_Broken(), 0)
        try:
            assert isinstance(stream, MockExternalStream)
            # No caller handle => synthesized non-zero fake.
            assert stream.ptr != 0
        finally:
            stream._shutdown()

    def test_delegates_to_cuda_backend_when_available(self):
        """When the CUDA backend returns a stream, the factory yields it."""
        sentinel = object()

        class _FakeTorchStream:
            cuda_stream = 0xABCDEF

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch(
                "lmcache.v1.platform.cuda.stream.make_cuda_external_stream",
                return_value=sentinel,
            ) as m,
        ):
            result = make_external_stream(_FakeTorchStream(), 3)

        assert result is sentinel
        m.assert_called_once_with(0xABCDEF, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.platform.cuda.stream``.

Runs without cupy installed: we stub the import via ``mock.patch`` so
the "cupy available" code path can be exercised on any host.
"""

# Standard
from unittest import mock

# Third Party
import pytest

# First Party
from lmcache.v1.platform.cuda import stream as cuda_stream


def test_make_cuda_external_stream_returns_none_when_cupy_missing():
    """If cupy cannot be imported, the factory declines by returning None."""
    with mock.patch.object(cuda_stream, "_try_import_cupy", return_value=None):
        assert cuda_stream.make_cuda_external_stream(0xDEADBEEF, 0) is None


def test_make_cuda_external_stream_delegates_to_cupy():
    """When cupy is available, construction is delegated verbatim."""
    sentinel = object()
    fake_cupy = mock.MagicMock()
    fake_cupy.cuda.ExternalStream.return_value = sentinel

    with mock.patch.object(cuda_stream, "_try_import_cupy", return_value=fake_cupy):
        result = cuda_stream.make_cuda_external_stream(0xABCDEF, 7)

    assert result is sentinel
    fake_cupy.cuda.ExternalStream.assert_called_once_with(0xABCDEF, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

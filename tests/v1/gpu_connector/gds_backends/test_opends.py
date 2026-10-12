# SPDX-License-Identifier: Apache-2.0
"""OpenDS tests with a fake ``opends`` package."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock
import importlib
import os
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends import opends as oa


class _OpenDSError(RuntimeError):
    """Stand-in for ``opends.OpenDSError``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"opends fake error {code}")
        self.code = code


@pytest.fixture(autouse=True)
def opends(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Install a fake ``opends`` package whose calls all succeed."""
    package = Mock()
    package.OpenDSError = _OpenDSError
    package.STREAM_FIXED = 0x7
    monkeypatch.setitem(sys.modules, "opends", package)
    return package


@pytest.fixture
def backend() -> Iterator[oa.Backend]:
    backend = oa.Backend()
    yield backend
    backend.close_driver()


def _fake_gpu_tensor() -> Any:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(is_cuda=True)


class TestEnvironment:
    def test_requires_cuda_build(
        self, backend: oa.Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.version, "cuda", None)
        with pytest.raises(ValueError, match="CUDA"):
            backend.validate_environment()

    def test_is_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.version, "cuda", "12.9")
        assert not oa.Backend.is_default()


class TestPackage:
    @pytest.mark.parametrize(
        "package",
        [None, ModuleType("opends")],
        ids=["not installed", "older or another project"],
    )
    def test_first_use_names_the_required_package(
        self, backend: oa.Backend, monkeypatch: pytest.MonkeyPatch, package: Any
    ) -> None:
        monkeypatch.setattr(torch.version, "cuda", "12.9")
        monkeypatch.setitem(sys.modules, "opends", package)
        backend.validate_environment()
        with pytest.raises(RuntimeError, match="OpenDS 0.3.0 or later"):
            backend.register_stream(7)

    def test_installed_package_has_what_the_backend_uses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other tests run on a fake; this one checks the real package."""
        monkeypatch.delitem(sys.modules, "opends")
        try:
            real = importlib.import_module("opends")
        except (ImportError, OSError):
            pytest.skip("the opends package of OpenDS is not installed")
        if not hasattr(real, "STREAM_FIXED"):
            pytest.skip("the installed opends package is older or another project")
        for name in (
            "Driver",
            "OpenDSError",
            "OpenDSFile",
            "deregister_buffer",
            "deregister_stream",
            "register_buffer",
            "register_stream",
        ):
            assert hasattr(real, name), name
        assert issubclass(real.OpenDSError, RuntimeError)
        assert isinstance(real.StreamOp.nbytes, property)
        for name in ("close", "read_stream", "write_stream"):
            assert hasattr(real.OpenDSFile, name), name


class TestDriverLifecycle:
    def test_overlapping_backends_share_driver(
        self, backend: oa.Backend, opends: Mock
    ) -> None:
        session = opends.Driver.return_value
        other = oa.Backend()
        try:
            backend.register_stream(7)
            other.register_stream(9)
            opends.Driver.assert_called_once_with()
            backend.close_driver()
            session.close.assert_not_called()
            other.register_buffer(_fake_gpu_tensor())
        finally:
            other.close_driver()
        session.close.assert_called_once_with()

    def test_open_error_propagates(self, backend: oa.Backend, opends: Mock) -> None:
        opends.Driver.side_effect = _OpenDSError(5033)
        with pytest.raises(_OpenDSError, match="5033"):
            backend.register_stream(0)
        backend.close_driver()
        opends.Driver.return_value.close.assert_not_called()

    def test_close_error_propagates(self, backend: oa.Backend, opends: Mock) -> None:
        backend.register_stream(0)
        opends.Driver.return_value.close.side_effect = _OpenDSError(5006)
        with pytest.raises(_OpenDSError, match="5006"):
            backend.close_driver()


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self, backend: oa.Backend, opends: Mock) -> None:
        tensor: Any = SimpleNamespace(is_cuda=False)
        with pytest.raises(ValueError, match="CUDA"):
            backend.register_buffer(tensor)
        opends.register_buffer.assert_not_called()

    def test_registers_and_deregisters_the_tensor(
        self, backend: oa.Backend, opends: Mock
    ) -> None:
        tensor = _fake_gpu_tensor()
        backend.register_buffer(tensor)
        opends.Driver.assert_called_once_with()
        opends.register_buffer.assert_called_once_with(tensor)
        backend.deregister_buffer(tensor)
        opends.deregister_buffer.assert_called_once_with(tensor)

    def test_register_error_propagates(self, backend: oa.Backend, opends: Mock) -> None:
        opends.register_buffer.side_effect = _OpenDSError(5011)
        with pytest.raises(_OpenDSError, match="5011"):
            backend.register_buffer(_fake_gpu_tensor())


class TestStreamRegistration:
    def test_register_stream_uses_fixed_flags(
        self, backend: oa.Backend, opends: Mock
    ) -> None:
        backend.register_stream(0xABC)
        opends.register_stream.assert_called_once_with(0xABC, opends.STREAM_FIXED)
        backend.deregister_stream(0xABC)
        opends.deregister_stream.assert_called_once_with(0xABC)

    def test_register_stream_propagates_error(
        self, backend: oa.Backend, opends: Mock
    ) -> None:
        opends.register_stream.side_effect = _OpenDSError(5030)
        with pytest.raises(_OpenDSError, match="5030"):
            backend.register_stream(0xABC)


class TestAsyncHandleIO:
    @pytest.mark.parametrize("operation", ["read", "write"])
    def test_io_passes_its_arguments_to_the_slab_file(self, operation: str) -> None:
        file = Mock()
        handle = oa.AsyncHandle(file=file, path="/slab")
        submit = handle.read_async if operation == "read" else handle.write_async
        submit(
            buf_base=0x3000, size=4096, file_offset=8192, buf_offset=512, raw_stream=0x9
        )
        getattr(file, f"{operation}_stream").assert_called_once_with(
            0x3000, 4096, 8192, 512, 0x9
        )

    @pytest.mark.parametrize(
        ("op", "expected"),
        [
            (SimpleNamespace(nbytes=0), 0),
            (SimpleNamespace(nbytes=4096), 4096),
            (SimpleNamespace(nbytes=-5022), -5022),
        ],
        ids=["pending", "complete", "failed"],
    )
    def test_submission_reports_the_stream_operation(
        self, op: SimpleNamespace, expected: int
    ) -> None:
        file = Mock()
        file.read_stream.return_value = op
        sub = oa.AsyncHandle(file=file, path="/slab").read_async(
            buf_base=0x3000, size=4096, file_offset=0, buf_offset=0, raw_stream=9
        )
        assert sub.bytes_done == expected

    def test_submission_error_raises(self) -> None:
        file = Mock()
        file.write_stream.side_effect = _OpenDSError(5015)
        with pytest.raises(_OpenDSError, match="5015"):
            oa.AsyncHandle(file=file, path="/slab").write_async(
                buf_base=0x3000, size=2048, file_offset=0, buf_offset=0, raw_stream=0x9
            )


class TestSlab:
    def test_rejects_buffered_io(self, backend: oa.Backend) -> None:
        with pytest.raises(ValueError, match="direct-io"):
            backend.open_slab("/slabs", 4096, direct_io=False)

    def test_opens_preallocated_slab_as_is(
        self, backend: oa.Backend, opends: Mock, tmp_path: Path
    ) -> None:
        slab = tmp_path / "lmcache_gds_slab.bin"
        data = os.urandom(4096)
        slab.write_bytes(data)
        with backend.open_slab(str(tmp_path), 4096, direct_io=True) as handle:
            assert handle.path == str(slab)
        opends.Driver.assert_called_once_with()
        opends.OpenDSFile.assert_called_once_with(str(slab), "r+", use_direct_io=True)
        opends.OpenDSFile.return_value.close.assert_called_once_with()
        assert slab.read_bytes() == data

    @pytest.mark.parametrize("slab", ["missing", "sparse", "short"])
    def test_requires_preallocated_slab(
        self, backend: oa.Backend, opends: Mock, tmp_path: Path, slab: str
    ) -> None:
        path = tmp_path / "lmcache_gds_slab.bin"
        if slab == "sparse":
            path.touch()
            os.truncate(path, 4096)
        elif slab == "short":
            path.write_bytes(os.urandom(2048))
        with pytest.raises(ValueError, match="preallocated 4096-byte slab"):
            backend.open_slab(str(tmp_path), 4096, direct_io=True)
        opends.OpenDSFile.assert_not_called()

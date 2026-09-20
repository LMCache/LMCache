# SPDX-License-Identifier: Apache-2.0
"""Tests for the extensible async GDS backend dispatcher."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import importlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import _cufile_async
from lmcache.v1.gpu_connector import _gds_async as ca
from lmcache.v1.gpu_connector import _hipfile_async, _phx_async, _ugds_async
from lmcache.v1.gpu_connector._gds_backend import GDSAsyncBackend


@pytest.fixture(autouse=True)
def _reset_backend_selection() -> Iterator[None]:
    importlib.reload(ca)
    yield
    importlib.reload(ca)


def test_dispatcher_source_has_no_backend_specific_names() -> None:
    source = Path(ca.__file__).read_text(encoding="utf-8").lower()

    assert "cufile" not in source
    assert "hipfile" not in source
    assert "ugds" not in source
    assert "phx" not in source


@pytest.mark.parametrize(
    "backend_cls",
    [
        _cufile_async.CuFileAsyncBackend,
        _hipfile_async.HipFileAsyncBackend,
        _ugds_async.UgdsAsyncBackend,
        _phx_async.PhxAsyncBackend,
    ],
)
def test_backend_modules_define_backend_subclasses(
    backend_cls: type[GDSAsyncBackend],
) -> None:
    assert issubclass(backend_cls, GDSAsyncBackend)


@pytest.mark.parametrize(
    "module",
    [_cufile_async, _hipfile_async, _ugds_async, _phx_async],
)
def test_backend_modules_do_not_require_global_instances(module: object) -> None:
    assert not hasattr(module, "GDS_BACKEND")


@pytest.mark.parametrize(
    ("cuda_version", "hip_version", "expected"),
    [
        ("12.9", None, "cufile"),
        (None, "6.3", "hipfile"),
    ],
)
def test_auto_selection_uses_backend_metadata(
    monkeypatch: pytest.MonkeyPatch,
    cuda_version: str | None,
    hip_version: str | None,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)

    assert ca.select_backend(ca.AUTO_BACKEND_NAME) == expected


def test_platform_validation_uses_backend_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)

    with pytest.raises(ValueError, match="(?=.*ROCm)(?=.*CUDA)"):
        ca.select_backend("ugds")


def test_device_capacity_delegates_to_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(_ugds_async, "get_device_capacity", lambda fd, handle: 1234)

    assert ca.select_backend("ugds") == "ugds"
    assert ca.get_device_capacity(3, 4) == 1234


def test_device_capacity_rejects_backends_without_that_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch.version, "hip", None)

    assert ca.select_backend("cufile") == "cufile"
    with pytest.raises(RuntimeError, match="capacity query"):
        ca.get_device_capacity(3, 4)

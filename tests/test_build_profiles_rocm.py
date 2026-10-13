# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm/HIP build profile's source-path handling."""

# Standard
from pathlib import Path
from types import ModuleType
from typing import NamedTuple
import os
import sys

# Third Party
import pytest

# First Party
from setup_extensions.build_profiles import rocm


class _HipifyResult(NamedTuple):
    """Stand-in for ``torch.utils.hipify``'s per-file result object."""

    hipified_path: str


def _install_hipify_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``torch.utils.hipify.hipify_python`` with a no-op stub.

    The stub mimics hipify's rename of ``*.cu`` to ``*.hip`` and leaves every
    other file name untouched, so tests can run without torch or ROCm.
    """

    def fake_hipify(
        project_directory: str,
        output_directory: str,
        extra_files: list[str],
        **kwargs: object,
    ) -> dict[str, _HipifyResult]:
        result: dict[str, _HipifyResult] = {}
        for abs_path in extra_files:
            root, ext = os.path.splitext(abs_path)
            hipified = root + ".hip" if ext == ".cu" else abs_path
            result[abs_path] = _HipifyResult(hipified_path=hipified)
        return result

    stub = ModuleType("torch.utils.hipify.hipify_python")
    stub.hipify = fake_hipify  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.hipify.hipify_python", stub)


@pytest.fixture
def hipify_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the ROCm profile at a throwaway source tree and return its root."""
    csrc_cuda = tmp_path / "csrc" / "cuda"
    csrc_cuda.mkdir(parents=True)
    for name in ("ac_dec.cu", "pybind.cpp", "utils.h"):
        (csrc_cuda / name).write_text("// stub\n")

    monkeypatch.setattr(rocm, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(rocm, "HIPIFY_DIR", str(csrc_cuda))
    monkeypatch.setattr(rocm, "HIPIFY_OUT_DIR", str(tmp_path / "csrc_hip" / "cuda"))
    _install_hipify_stub(monkeypatch)
    return tmp_path


def test_hipify_wrapper_returns_paths_relative_to_project_root(
    hipify_sandbox: Path,
) -> None:
    """Sources must be root-relative: setuptools rejects absolute paths.

    ``setuptools.command.build_py`` asserts that every entry of the egg-info
    manifest -- which includes ``ext_modules`` sources -- is relative to the
    ``setup.py`` directory, so absolute paths fail the wheel build outright.
    """
    sources = rocm._hipify_wrapper(["ac_dec.cu", "pybind.cpp"])

    assert sources == ["csrc_hip/cuda/ac_dec.hip", "csrc_hip/cuda/pybind.cpp"]
    for source in sources:
        assert not os.path.isabs(source)
        assert (hipify_sandbox / source).parent.is_dir()


def test_hipify_wrapper_copies_headers_into_the_output_tree(
    hipify_sandbox: Path,
) -> None:
    """Headers next to the CUDA sources are carried over for the HIP build."""
    rocm._hipify_wrapper(["ac_dec.cu"])

    assert (hipify_sandbox / "csrc_hip" / "cuda" / "utils.h").is_file()

# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN build profile's selection and build contract."""

# Standard
from pathlib import Path
import os
import sys

# Third Party
import pytest

# First Party
from setup_extensions import BuildPolicy
from setup_extensions.build_profiles.rbln import RblnProfile


def test_detects_an_importable_torch_rbln(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``detect()`` is true when a ``torch_rbln`` package is importable."""
    (tmp_path / "torch_rbln").mkdir()
    (tmp_path / "torch_rbln" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert RblnProfile().detect()


def test_does_not_detect_without_torch_rbln(monkeypatch: pytest.MonkeyPatch) -> None:
    """``detect()`` is false when ``torch_rbln`` cannot be imported."""
    # A None entry in sys.modules is Python's marker for "import blocked".
    monkeypatch.setitem(sys.modules, "torch_rbln", None)

    assert not RblnProfile().detect()


def test_build_with_rbln_selects_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BUILD_WITH_RBLN=1`` selects the RBLN profile."""
    # Other lanes export their own BUILD_WITH_* (e.g. BUILD_WITH_HIP=1 on AMD),
    # which would make the selection ambiguous.
    for env_var in list(os.environ):
        if env_var.startswith("BUILD_WITH_"):
            monkeypatch.delenv(env_var)
    for env_var in ("NO_NATIVE_EXT", "NO_CUDA_EXT", "NO_GPU_EXT"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("BUILD_WITH_RBLN", "1")

    profile = BuildPolicy().resolve_profile()

    assert isinstance(profile, RblnProfile)


def test_build_adds_no_extensions() -> None:
    """RBLN has no native kernels, so only common extensions are built."""
    assert RblnProfile().build() == ([], {})


def test_adds_no_install_requirements() -> None:
    """The RBLN runtime stack is installed separately, not via install_requires."""
    assert RblnProfile().requirements_file() is None

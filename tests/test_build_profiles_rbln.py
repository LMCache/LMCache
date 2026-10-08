# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN build profile's selection and build contract."""

# Standard
import importlib.util

# Third Party
import pytest

# First Party
from setup_extensions import BuildPolicy
from setup_extensions.build_profiles.rbln import RblnProfile


@pytest.mark.parametrize("installed", [True, False])
def test_detect_follows_torch_rbln_installation(
    monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    """``detect()`` is true exactly when ``torch_rbln`` is importable."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch_rbln":
            return object() if installed else None
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    assert RblnProfile().detect() is installed


def test_build_with_rbln_selects_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BUILD_WITH_RBLN=1`` selects the RBLN profile."""
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

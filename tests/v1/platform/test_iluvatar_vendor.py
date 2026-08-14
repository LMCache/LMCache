# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CUDA-compatible Iluvatar detection.

Hardware smoke (manual, on BI-V150 after ``BUILD_WITH_ILUVATAR=1`` build)::

    from lmcache.v1.platform import torch_device_type
    from lmcache.v1.platform.cuda import is_iluvatar_device
    assert torch_device_type == "cuda"
    assert is_iluvatar_device()
"""

# Standard
from types import ModuleType
from unittest.mock import MagicMock, patch
import sys

# Third Party
import pytest


def _is_iluvatar_device_from_name(name: str | None) -> bool:
    """Mirror the name-match used by :func:`is_iluvatar_device`."""
    return name is not None and "Iluvatar" in name


def test_is_iluvatar_device_true_for_corex_name() -> None:
    assert _is_iluvatar_device_from_name("Iluvatar BI-V150") is True


def test_is_iluvatar_device_false_for_nvidia_name() -> None:
    assert _is_iluvatar_device_from_name("NVIDIA A100-SXM4-40GB") is False


def test_is_iluvatar_device_false_when_no_name() -> None:
    assert _is_iluvatar_device_from_name(None) is False


def test_is_iluvatar_device_case_sensitive_token() -> None:
    """Vendor match is case-sensitive on the ``Iluvatar`` token."""
    assert _is_iluvatar_device_from_name("iluvatar BI-V150") is False


def test_is_iluvatar_device_uses_torch_get_device_name() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Iluvatar BI-V150"

    with patch.dict(sys.modules, {"torch": mock_torch}):
        # First Party
        from lmcache.v1.platform.cuda import is_iluvatar_device

        assert is_iluvatar_device(0) is True
        mock_torch.cuda.get_device_name.assert_called_with(0)


def test_is_iluvatar_device_false_when_cuda_unavailable() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        # First Party
        from lmcache.v1.platform.cuda import is_iluvatar_device

        assert is_iluvatar_device() is False


def test_iluvatar_profile_detect_true_when_device_name_is_iluvatar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.current_device.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "Iluvatar BI-V150"

    monkeypatch.setattr(
        "setup_extensions.build_profiles.iluvatar.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    with patch.dict(sys.modules, {"torch": mock_torch}):
        # First Party
        from setup_extensions.build_profiles.iluvatar import IluvatarProfile

        assert IluvatarProfile().detect() is True


def test_iluvatar_profile_detect_false_for_nvidia_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.current_device.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"

    monkeypatch.setattr(
        "setup_extensions.build_profiles.iluvatar.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    with patch.dict(sys.modules, {"torch": mock_torch}):
        # First Party
        from setup_extensions.build_profiles.iluvatar import IluvatarProfile

        assert IluvatarProfile().detect() is False


def test_build_policy_auto_detects_iluvatar_from_device_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detect (no BUILD_WITH_*) picks iluvatar when the GPU name matches."""
    monkeypatch.delenv("BUILD_WITH_ILUVATAR", raising=False)
    monkeypatch.delenv("BUILD_WITH_CUDA", raising=False)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.current_device.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "Iluvatar BI-V150"

    monkeypatch.setattr(
        "setup_extensions.build_profiles.cuda.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(
        "setup_extensions.build_profiles.iluvatar.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    with patch.dict(sys.modules, {"torch": mock_torch}):
        # First Party
        from setup_extensions.policy import BuildPolicy

        profile = BuildPolicy().resolve_profile()
    assert profile is not None
    assert profile.name == "iluvatar"


def test_iluvatar_profile_injects_use_iluvatar_macro() -> None:
    # First Party
    from setup_extensions.build_profiles.iluvatar import IluvatarProfile

    profile = IluvatarProfile()
    captured: dict = {}

    class _FakeExt:
        def __init__(
            self,
            name,
            sources,
            define_macros=None,
            extra_compile_args=None,
            include_dirs=None,
            **_kwargs,
        ):
            captured["name"] = name
            captured["define_macros"] = define_macros
            captured["extra_compile_args"] = extra_compile_args
            captured["sources"] = sources
            captured["include_dirs"] = include_dirs

    fake_torch_utils = ModuleType("torch.utils")
    fake_cpp = ModuleType("torch.utils.cpp_extension")
    fake_cpp.CUDAExtension = _FakeExt
    fake_cpp.BuildExtension = object
    fake_torch_utils.cpp_extension = fake_cpp

    with patch.dict(
        sys.modules,
        {
            "torch": MagicMock(),
            "torch.utils": fake_torch_utils,
            "torch.utils.cpp_extension": fake_cpp,
        },
    ):
        profile.build()

    assert captured["name"] == "lmcache.cuda_ops"
    assert captured["define_macros"] == [("USE_ILUVATAR", "1")]
    assert "-DUSE_ILUVATAR=1" in captured["extra_compile_args"]["nvcc"]
    assert "-DUSE_ILUVATAR=1" in captured["extra_compile_args"]["cxx"]
    assert "csrc/cuda/blend_kernels.cu" in captured["sources"]
    assert captured["include_dirs"]


def test_build_policy_prefers_non_cuda_when_multiple_detect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cuda + Iluvatar both detecting must not pick plain cuda macros."""
    # First Party
    from setup_extensions.policy import BuildPolicy

    monkeypatch.delenv("BUILD_WITH_CUDA", raising=False)
    monkeypatch.delenv("BUILD_WITH_ILUVATAR", raising=False)
    monkeypatch.setattr(
        "setup_extensions.build_profiles.cuda.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(
        "setup_extensions.build_profiles.iluvatar.shutil.which",
        lambda name: "/usr/bin/nvcc" if name == "nvcc" else None,
    )
    monkeypatch.setattr(
        "setup_extensions.build_profiles.iluvatar._is_iluvatar_cuda_device",
        lambda: True,
    )

    profile = BuildPolicy().resolve_profile()
    assert profile is not None
    assert profile.name == "iluvatar"


def test_cuda_profile_detect_is_vendor_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CudaProfile only checks nvcc; multi-match preference lives in policy."""
    # First Party
    from setup_extensions.build_profiles.cuda import CudaProfile

    monkeypatch.setenv("BUILD_WITH_ILUVATAR", "1")
    monkeypatch.setattr(
        "setup_extensions.build_profiles.cuda.shutil.which",
        lambda _name: "/usr/bin/nvcc",
    )
    assert CudaProfile().detect() is True

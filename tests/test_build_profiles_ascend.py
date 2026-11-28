# SPDX-License-Identifier: Apache-2.0
"""Tests for the Ascend build profile's device detection and SoC lookup."""

# Standard
from collections.abc import Callable
from pathlib import Path
import os
import subprocess
import sys

# Third Party
import pytest
import torch

# First Party
from setup_extensions import BuildPolicy
from setup_extensions.build_profiles import ascend


def _board_report(chip_name: str, npu_name: str | None = "9382") -> str:
    """Return an ``npu-smi info -t board`` report for one board."""
    lines = [
        "\tNPU ID                         : 0",
        f"\tChip Name                      : {chip_name}",
    ]
    if npu_name is not None:
        lines.insert(1, f"\tNPU Name                       : {npu_name}")
    return "\n".join(lines) + "\n"


def _fake_npu_smi(boards: dict[int, str]) -> Callable[..., str]:
    """Return a ``subprocess.check_output`` stand-in for ``npu-smi``.

    Args:
        boards: Board reports by NPU id. Ids missing from the mapping make
            ``npu-smi`` exit non-zero; an empty mapping is a host whose driver
            reports no board.
    """

    def check_output(cmd: list[str], **kwargs: object) -> str:
        npu_id = int(cmd[cmd.index("-i") + 1])
        try:
            return boards[npu_id]
        except KeyError:
            raise subprocess.CalledProcessError(1, cmd) from None

    return check_output


def _missing_npu_smi(cmd: list[str], **kwargs: object) -> str:
    """Stand in for a host without the ``npu-smi`` binary."""
    raise FileNotFoundError("npu-smi")


@pytest.fixture(autouse=True)
def _bare_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make the host look like a machine without CANN, ``torch_npu`` or a driver.

    ``detect()`` reads the real environment, so every test starts from a host
    that has none of its inputs and adds them back explicitly.
    """
    monkeypatch.delenv("SOC_VERSION", raising=False)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "no-cann"))
    monkeypatch.setattr(subprocess, "check_output", _fake_npu_smi({}))
    monkeypatch.setitem(sys.modules, "torch_npu", None)


@pytest.fixture
def cann_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the CANN toolkit directory ``detect()`` requires."""
    home = tmp_path / "ascend-toolkit" / "latest"
    home.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    return home


@pytest.fixture
def torch_npu_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``torch_npu`` findable without importing it."""
    site = tmp_path / "site"
    package = site / "torch_npu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.delitem(sys.modules, "torch_npu")
    monkeypatch.syspath_prepend(str(site))


def test_detect_requires_the_cann_toolkit(
    torch_npu_package: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No toolkit directory means no build, even with a device signal."""
    monkeypatch.setenv("SOC_VERSION", "Ascend910B3")

    assert not ascend.AscendProfile().detect()


def test_detect_requires_torch_npu(
    cann_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The toolkit alone is not enough: ``torch_npu`` supplies the headers."""
    monkeypatch.setenv("SOC_VERSION", "Ascend910B3")

    assert not ascend.AscendProfile().detect()


def test_detect_from_the_npu_smi_board(
    cann_home: Path,
    torch_npu_package: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A board the driver reports selects the profile without ``SOC_VERSION``.

    Installing LMCache runs with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``, so this
    is the signal a real NPU host has to be detected by.
    """
    monkeypatch.setattr(
        subprocess, "check_output", _fake_npu_smi({0: _board_report("Ascend910")})
    )

    assert ascend.AscendProfile().detect()


def test_detect_from_soc_version_without_a_driver(
    cann_home: Path,
    torch_npu_package: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build host without the driver is detected through ``SOC_VERSION``."""
    monkeypatch.setattr(subprocess, "check_output", _missing_npu_smi)
    monkeypatch.setenv("SOC_VERSION", "Ascend910B3")

    assert ascend.AscendProfile().detect()


def test_detect_without_any_device_signal(
    cann_home: Path, torch_npu_package: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toolkit plus ``torch_npu`` but no NPU anywhere must not select Ascend."""
    monkeypatch.setattr(subprocess, "check_output", _missing_npu_smi)
    monkeypatch.delattr(torch, "npu", raising=False)

    assert not ascend.AscendProfile().detect()


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        ("Ascend910B3", "Ascend910B3"),
        ("ascend910B3", "Ascend910B3"),
        ("Ascend910_9382", "Ascend910_9382"),
    ],
)
def test_soc_version_from_the_environment_is_canonical(
    monkeypatch: pytest.MonkeyPatch, soc_version: str, expected: str
) -> None:
    """The ``Ascend`` prefix is spelled as CANN spells its SoC configs."""
    monkeypatch.setenv("SOC_VERSION", soc_version)

    assert ascend._npu_soc_version() == expected


def test_npu_smi_board_wins_over_soc_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """The board the driver reports is compiled for, not the environment hint."""
    monkeypatch.setattr(
        subprocess, "check_output", _fake_npu_smi({0: _board_report("Ascend910")})
    )
    monkeypatch.setenv("SOC_VERSION", "Ascend910B3")

    assert ascend._npu_soc_version() == "Ascend910_9382"


def test_soc_version_lookup_fails_without_any_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building needs a SoC; the error names the variable to set."""
    monkeypatch.setattr(subprocess, "check_output", _missing_npu_smi)

    with pytest.raises(RuntimeError, match="SOC_VERSION"):
        ascend._npu_soc_version()


@pytest.mark.parametrize(
    ("chip_name", "npu_name", "expected"),
    [
        ("Ascend910", "9382", "Ascend910_9382"),
        ("910B4", None, "Ascend910B4"),
        ("Ascend910B4", None, "Ascend910B4"),
    ],
)
def test_soc_name_from_the_board_report(
    monkeypatch: pytest.MonkeyPatch,
    chip_name: str,
    npu_name: str | None,
    expected: str,
) -> None:
    """The SoC is ``Chip Name_NPU Name``; a bare chip name gets the prefix.

    The board sits at id 1 so the lookup has to scan past the failing id 0.
    """
    boards = {1: _board_report(chip_name, npu_name)}
    monkeypatch.setattr(subprocess, "check_output", _fake_npu_smi(boards))

    assert ascend._soc_version_from_npu_smi() == expected


def test_soc_name_is_none_without_a_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without ``npu-smi`` has no board to read the SoC from."""
    monkeypatch.setattr(subprocess, "check_output", _missing_npu_smi)

    assert ascend._soc_version_from_npu_smi() is None


def test_soc_name_is_none_without_a_board(monkeypatch: pytest.MonkeyPatch) -> None:
    """``npu-smi`` failing on every id means no board, hence no SoC."""
    assert ascend._soc_version_from_npu_smi() is None


def test_build_with_ascend_selects_the_profile(
    cann_home: Path,
    torch_npu_package: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BUILD_WITH_ASCEND=1`` selects the Ascend profile."""
    # Other lanes export their own BUILD_WITH_* (e.g. BUILD_WITH_HIP=1 on AMD),
    # which would make the selection ambiguous.
    for env_var in list(os.environ):
        if env_var.startswith("BUILD_WITH_"):
            monkeypatch.delenv(env_var)
    for env_var in ("NO_NATIVE_EXT", "NO_CUDA_EXT", "NO_GPU_EXT"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("BUILD_WITH_ASCEND", "1")
    monkeypatch.setenv("SOC_VERSION", "Ascend910B3")

    profile = BuildPolicy().resolve_profile()

    assert isinstance(profile, ascend.AscendProfile)


def test_build_emits_the_cmake_extension() -> None:
    """The profile's only extension is the CMake-backed ``lmcache.npu_ops``."""
    ext_modules, cmdclass = ascend.AscendProfile().build()

    assert [ext.name for ext in ext_modules] == ["lmcache.npu_ops"]
    assert isinstance(ext_modules[0], ascend.CMakeExtension)
    assert ext_modules[0].cmake_lists_dir == str(ascend.ROOT_DIR / "csrc" / "ascend")
    assert cmdclass == {"build_ext": ascend.AscendCMakeBuildExt}

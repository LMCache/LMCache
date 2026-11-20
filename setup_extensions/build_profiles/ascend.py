# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU backend profile.

Builds ``lmcache.npu_ops`` from ``csrc/ascend`` with the CANN toolchain.  The
MP native core (block-transfer host code + Ascend C kernel) lives in this
repository; the in-process and transfer-channel extensions remain in the
LMCache-Ascend plugin.

Ascend C kernels and the CANN host toolchain do not fit the
setuptools/torch ``Extension`` flow used by CUDA, ROCm and MUSA, so the
extension is driven by an out-of-line CMake project.  ``AscendCMakeBuildExt``
runs CMake for :class:`CMakeExtension` and delegates the common C++
extensions (``lmcache_native``, Redis, filesystem) to torch's
``BuildExtension``.

The CMake project produces two artifacts, both installed next to ``lmcache``:

* ``npu_ops*.so`` -- the pybind module imported as ``lmcache.npu_ops``;
* ``libcache_kernels.so`` -- the Ascend C block-transfer kernels it links.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import configparser
import glob
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig

# Third Party
from setuptools import Extension
from torch.utils.cpp_extension import BuildExtension as _TorchBuildExtension
import torch

# First Party
from setup_extensions.build_profiles import BuildProfile

if TYPE_CHECKING:
    # First Party
    from setup_extensions.common_cpp import CommonExtSpec

ROOT_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT_DIR / "build" / "ascend"
DEFAULT_ASCEND_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"

#: Artifacts copied into the Python package directory after the CMake install.
_ARTIFACT_PATTERNS = ("npu_ops*.so", "libcache_kernels.so")


class CMakeExtension(Extension):
    """A setuptools extension built by an out-of-line CMake project."""

    def __init__(
        self,
        name: str,
        cmake_lists_dir: str = ".",
        **kwargs: object,
    ) -> None:
        """Initialize the extension.

        Args:
            name: Import path of the built module (for example
                ``lmcache.npu_ops``).
            cmake_lists_dir: CMake source directory, relative to the
                repository root.
            **kwargs: Forwarded to :class:`setuptools.Extension`.
        """
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = str((ROOT_DIR / cmake_lists_dir).resolve())


def ascend_home_path() -> Path:
    """Return the CANN toolkit root, honoring ``ASCEND_HOME_PATH``."""
    return Path(os.environ.get("ASCEND_HOME_PATH", DEFAULT_ASCEND_HOME))


def cann_version() -> str | None:
    """Read the CANN toolkit version from its install metadata, if present."""
    arch = platform.machine()
    info_path = ascend_home_path() / f"{arch}-linux" / "ascend_toolkit_install.info"
    if not info_path.exists():
        return None
    for raw in info_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("version="):
            return line.split("=", 1)[1].strip()
    return None


def _cann_version_85_or_later() -> bool:
    """Return whether the installed CANN toolkit uses the 8.5+ layout."""
    override = os.environ.get("USE_HIXL", "").strip().lower()
    if override in ("1", "true", "on"):
        return True
    if override in ("0", "false", "off"):
        return False

    version = cann_version()
    if version:
        parts = tuple(int(p) for p in re.findall(r"\d+", version))
        return parts >= (8, 5, 0)

    home = ascend_home_path()
    new_path = home / "set_env.sh"
    old_path = home.parent / "set_env.sh"
    return new_path.exists() and not old_path.exists()


def _ascend_env_script() -> Path:
    """Return the CANN environment script for this toolkit layout."""
    home = ascend_home_path()
    if _cann_version_85_or_later():
        return home / "set_env.sh"
    return home.parent / "set_env.sh"


def _npu_soc_version() -> str:
    """Return the target SoC name from ``npu-smi`` or ``SOC_VERSION``.

    Raises:
        RuntimeError: If neither the tool nor ``SOC_VERSION`` is available.
    """
    for npu_id in range(8):
        try:
            output = subprocess.check_output(
                ["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", "0"],
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        info: dict[str, str] = {}
        for line in output.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
        chip_name = info.get("Chip Name")
        if chip_name:
            npu_name = info.get("NPU Name")
            if npu_name:
                return f"{chip_name}_{npu_name}"
            return chip_name if chip_name.startswith("Ascend") else f"Ascend{chip_name}"

    soc_version = os.environ.get("SOC_VERSION")
    if soc_version:
        return (
            f"Ascend{soc_version[6:]}"
            if soc_version.lower().startswith("ascend")
            else soc_version
        )
    raise RuntimeError(
        "No available NPU found; run on an NPU host or set SOC_VERSION "
        "(for example SOC_VERSION=Ascend910B3)."
    )


def _aicore_arch_number(soc_version: str) -> str:
    """Read the AICore architecture number from CANN's platform config.

    Args:
        soc_version: SoC name returned by :func:`_npu_soc_version`.

    Returns:
        The AICore architecture number from ``AIC_version``.

    Raises:
        ValueError: If the SoC platform config is absent.
    """
    arch = platform.machine()
    ini_path = (
        ascend_home_path()
        / f"{arch}-linux"
        / "data"
        / "platform_config"
        / f"{soc_version}.ini"
    )
    if not ini_path.exists():
        raise ValueError(f"{ini_path} not found; check SOC_VERSION={soc_version!r}.")
    config = configparser.ConfigParser()
    config.read(ini_path)
    return config.get("version", "AIC_version").split("-")[-1]


def _pybind11_cmake_dir() -> str:
    """Return pybind11's CMake package directory.

    Raises:
        RuntimeError: If pybind11 is not installed in the build environment.
    """
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "pybind11 is required to build lmcache.npu_ops; "
            "install it in the build environment."
        ) from exc


def _cmake_command(ext: CMakeExtension) -> str:
    """Build the shell command that configures, builds and installs *ext*.

    Args:
        ext: CMake-backed extension to build.

    Returns:
        A shell command that sources the CANN environment, configures,
        builds and installs the extension.

    Raises:
        RuntimeError: If the CANN environment or ``torch_npu`` is missing.
        ValueError: If the SoC platform configuration cannot be found.
    """
    soc_version = _npu_soc_version()
    arch = platform.machine()
    aicore_arch = _aicore_arch_number(soc_version)
    env_script = _ascend_env_script()
    if not env_script.exists():
        raise RuntimeError(
            f"CANN environment script {env_script} does not exist; "
            "set ASCEND_HOME_PATH to the toolkit root."
        )

    # Third Party
    import torch_npu

    torch_path = Path(torch.__file__).resolve().parent
    torch_npu_path = Path(torch_npu.__file__).resolve().parent
    torch_cmake_dir = Path(torch.utils.cmake_prefix_path) / "Torch"
    python_include = sysconfig.get_path("include", scheme="posix_prefix")
    install_path = BUILD_DIR / "install"

    parts = [
        f". {env_script} &&",
        "cmake",
        f"-S {ext.cmake_lists_dir}",
        f"-B {BUILD_DIR}",
        f"-DSOC_VERSION={soc_version}",
        f"-DASCEND_AICORE_ARCH={aicore_arch}",
        f"-DARCH={arch}",
        "-DUSE_ASCEND=1",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DCMAKE_PREFIX_PATH={_pybind11_cmake_dir()}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_path}",
        f"-DPYTHON_INCLUDE_PATH={python_include}",
        f"-DASCEND_CANN_PACKAGE_PATH={ascend_home_path()}",
        f"-DTORCH_NPU_PATH={torch_npu_path}",
        f"-DTORCH_PATH={torch_path}",
        f"-DGLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
        f"-DTorch_DIR={torch_cmake_dir}",
        f"-DLMCACHE_ROOT={ROOT_DIR}",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
    ]
    if compiler := os.environ.get("CXX"):
        parts.append(f"-DCMAKE_CXX_COMPILER={compiler}")
    if compiler := os.environ.get("CC"):
        parts.append(f"-DCMAKE_C_COMPILER={compiler}")

    parts.extend(
        (
            f"&& cmake --build {BUILD_DIR} -j --verbose",
            f"&& cmake --install {BUILD_DIR}",
        )
    )
    return " ".join(parts)


def _artifact_paths() -> list[Path]:
    """Return every built shared object under the CMake install tree."""
    by_name: dict[str, Path] = {}
    for subdir in ("", "lib", "lib64"):
        search = BUILD_DIR / "install" / subdir
        if not search.exists():
            continue
        for pattern in _ARTIFACT_PATTERNS:
            for path in glob.glob(str(search / pattern)):
                by_name[Path(path).name] = Path(path)
    return list(by_name.values())


def _develop_command() -> type:
    """Return setuptools' ``develop`` command class (or a harmless stand-in)."""
    try:
        # Third Party
        from setuptools.command.develop import develop

        return develop
    except ImportError:  # pragma: no cover - very old setuptools
        return type("_NoDevelop", (), {})


class AscendCMakeBuildExt(_TorchBuildExtension):
    """``build_ext`` that runs CMake for :class:`CMakeExtension` only.

    Inherits torch's ``BuildExtension`` so the common C++ extensions
    (``lmcache_native``, Redis, filesystem) keep the same build behavior they
    get under the CUDA/MUSA profiles.
    """

    def build_extension(self, ext: Extension) -> None:
        """Build one extension, delegating non-CMake modules to torch.

        Args:
            ext: Extension being built.
        """
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
        self._build_cmake(ext)

    def _build_cmake(self, ext: CMakeExtension) -> None:
        """Configure, build, install and copy the Ascend artifacts.

        Args:
            ext: CMake-backed extension.

        Raises:
            RuntimeError: If CMake produces no shared objects.
        """
        # CANN's kernel preprocess/build is not reliably incremental; a fresh
        # directory mirrors the CI build and avoids stale objects reaching
        # merge_obj_text ("unknown file type").
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        command = _cmake_command(ext)
        print(f"Building {ext.name} with CMake:\n{command}")
        subprocess.run(command, cwd=ROOT_DIR, shell=True, check=True, text=True)

        artifacts = _artifact_paths()
        if not artifacts:
            raise RuntimeError(
                f"CMake produced no files matching {_ARTIFACT_PATTERNS} "
                f"under {BUILD_DIR / 'install'}"
            )

        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        destination.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            shutil.copy2(artifact, destination / artifact.name)
            print(f"Copied {artifact.name} to {destination}")

        # Editable installs import from the source tree, not build/lib.
        if self.inplace or isinstance(
            self.distribution.get_command_obj("develop"),
            _develop_command(),
        ):
            source_pkg = ROOT_DIR / "lmcache"
            for artifact in artifacts:
                shutil.copy2(artifact, source_pkg / artifact.name)
                print(f"Copied {artifact.name} to {source_pkg}")


class AscendProfile(BuildProfile):
    """Ascend NPU extension build profile (``lmcache.npu_ops``)."""

    name = "ascend"
    env_var = "BUILD_WITH_ASCEND"

    def detect(self) -> bool:
        """Detect a usable CANN build environment.

        Requires the toolkit directory and ``torch_npu``.  On a build host
        without an NPU device, ``SOC_VERSION`` opts the profile in; with a
        device present, ``torch.npu.is_available()`` is the signal.  This keeps
        a CUDA host that merely has CANN installed from selecting Ascend.

        Returns:
            ``True`` when the Ascend native extension should be built.
        """
        if not ascend_home_path().is_dir():
            return False
        if importlib.util.find_spec("torch_npu") is None:
            return False
        if os.environ.get("SOC_VERSION"):
            return True
        try:
            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            # detect() runs during profile auto-selection and must not raise.
            return False

    def build(self) -> tuple[list[Extension], dict[str, type]]:
        """Return the CMake-backed ``lmcache.npu_ops`` extension.

        Returns:
            ``(ext_modules, cmdclass)`` for :class:`BuildPolicy`.
        """
        print("Building Ascend NPU extension (lmcache.npu_ops)")
        ext_modules = [CMakeExtension("lmcache.npu_ops", "csrc/ascend")]
        return ext_modules, {"build_ext": AscendCMakeBuildExt}

    def extra_cxx_flags_for(self, spec: CommonExtSpec) -> list[str]:
        """All common extensions share the same ABI flag under Ascend."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI-aware default flags, matching torch / torch-npu."""
        if ENABLE_CXX11_ABI:
            return ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        return ["-D_GLIBCXX_USE_CXX11_ABI=0"]

    def requirements_file(self) -> str | None:
        """Ascend runtime deps (torch-npu / CANN) are installed separately."""
        return None

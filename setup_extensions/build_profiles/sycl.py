# SPDX-License-Identifier: Apache-2.0
"""SYCL / Intel XPU GPU backend profile.

Builds ``lmcache.xpu_ops`` using the DPC++ compiler (icpx).
Requires Intel oneAPI environment to be sourced before building.
"""

# Standard
from typing import TYPE_CHECKING, Optional
import os
import shutil

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

# icpx links every shared object against Intel's libsvml / libimf / libirng /
# libintlc by default and emits ``__svml_*`` calls at compile time. The flag
# must be present on BOTH the compile and the link line: compile-only leaves
# the NEEDED entries, link-only leaves unresolved ``__svml_*`` symbols.
# Without it auditwheel vendors ~37 MB of those runtime libs into
# ``lmcache.libs`` although the XPU runtime image already ships them via
# ``intel-cmplr-lib-rt``.
_NO_INTEL_LIB = "-no-intel-lib"


class SyclProfile(BuildProfile):
    """SYCL / Intel XPU GPU extension build profile."""

    name = "sycl"
    env_var = "BUILD_WITH_SYCL"

    def detect(self) -> bool:
        """Detect SYCL by checking for the icpx compiler."""
        return shutil.which("icpx") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build SYCL/XPU extensions via DPC++."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building SYCL/XPU extensions")
        if shutil.which("icpx") is None:
            raise RuntimeError("icpx not found. Please source oneAPI setvars.sh first")
        os.environ["CXX"] = "icpx"
        # distutils appends LDFLAGS to the shared-link command of every
        # extension, including the common ones built via ``common_cpp``.
        ldflags = os.environ.get("LDFLAGS", "").split()
        if _NO_INTEL_LIB not in ldflags:
            os.environ["LDFLAGS"] = " ".join(ldflags + [_NO_INTEL_LIB])
        oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
        include_dirs = ["%s/include" % oneapi_root]
        library_dirs = ["%s/lib" % oneapi_root]

        sycl_sources = [
            "csrc/sycl/pybind_sycl.cpp",
            "csrc/sycl/host_register.cpp",
            "csrc/sycl/mem_kernels_sycl.cpp",
            "csrc/sycl/mp_mem_kernels_sycl.cpp",
            "csrc/sycl/cal_cdf_sycl.cpp",
            "csrc/sycl/pos_kernels_sycl.cpp",
            "csrc/sycl/ac_enc_sycl.cpp",
            "csrc/sycl/ac_dec_sycl.cpp",
        ]
        ext_modules = [
            cpp_extension.SyclExtension(
                "lmcache.xpu_ops",
                sources=sycl_sources,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_compile_args={
                    "cxx": [
                        "-D_GLIBCXX_USE_CXX11_ABI=1",
                        "-O3",
                        # Python's sysconfig CFLAGS inject ``-g``. icpx applies
                        # it to the SPIR-V device compile as well, inflating
                        # the embedded device image ~170x (31 KB -> 5.3 MB per
                        # TU). Override it; the CUDA/ROCm wheels ship no
                        # device debug info either.
                        "-g0",
                        "-fsycl",
                        "-fno-sycl-id-queries-fit-in-int",
                        "-ffast-math",
                        "-funroll-loops",
                        "-Wno-deprecated-declarations",
                        "-Wno-nan-infinity-disabled",
                        _NO_INTEL_LIB,
                    ],
                },
                extra_link_args=["-fsycl", _NO_INTEL_LIB],
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """SYCL uses CXX11 ABI for all common extensions except
        ``lmcache_fs``, which omits ABI flags to preserve pre-refactor
        behaviour."""
        if spec.name == "lmcache_fs":
            return [_NO_INTEL_LIB]
        return ["-D_GLIBCXX_USE_CXX11_ABI=1", _NO_INTEL_LIB]

    def default_cxx_flags(self) -> list[str]:
        """SYCL downstream consumers use the CXX11 ABI."""
        return ["-D_GLIBCXX_USE_CXX11_ABI=1"]

    def requirements_file(self) -> Optional[str]:
        """SYCL core requirements file."""
        return "xpu_core.txt"

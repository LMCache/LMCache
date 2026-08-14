# SPDX-License-Identifier: Apache-2.0
"""Iluvatar CoreX CUDA-compatible GPU backend profile.

Iluvatar exposes a CUDA-compatible toolchain (``nvcc`` under CoreX) and a
``torch.cuda`` runtime.  Runtime ``device_type`` therefore stays ``"cuda"``;
this profile only injects ``USE_ILUVATAR`` (same pattern as ``USE_ROCM``) so
``lmcache.cuda_ops`` builds without NVIDIA-only ``cuda_fp8.h`` / PTX ``.cs``
asm.

Preferred explicit selection::

    BUILD_WITH_ILUVATAR=1 pip install -e .

Auto-detect uses the same criterion as
:func:`lmcache.v1.platform.cuda.is_iluvatar_device` (``get_device_name``
contains ``Iluvatar``) plus an ``nvcc`` toolchain.  ``setup.py`` must not
import the ``lmcache`` package (that runs ``cuda_ops`` init), so the check is
inlined here.  Headless / no-GPU builds should set ``BUILD_WITH_ILUVATAR=1``.
"""

# Standard
from typing import TYPE_CHECKING
import shutil

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles.cuda import (
    CSRC_DIR,
    ENABLE_CXX11_ABI,
    CudaProfile,
)

# Same sources as CudaProfile.build(); kept local so cuda.py stays untouched.
_CUDA_OPS_SOURCES: list[str] = [
    "csrc/cuda/pybind.cpp",
    "csrc/cuda/mem_kernels.cu",
    "csrc/cuda/mp_mem_kernels.cu",
    "csrc/cuda/blend_kernels.cu",
    "csrc/cuda/cal_cdf.cu",
    "csrc/cuda/ac_enc.cu",
    "csrc/cuda/ac_dec.cu",
    "csrc/cuda/pos_kernels.cu",
    "csrc/cuda/mem_alloc.cpp",
    "csrc/cuda/utils.cpp",
    "csrc/cuda/event_recorder.cpp",
    "csrc/cuda/completion_recorder.cpp",
]


def _is_iluvatar_cuda_device() -> bool:
    """Return True when ``torch.cuda.get_device_name`` contains ``Iluvatar``.

    Same criterion as :func:`lmcache.v1.platform.cuda.is_iluvatar_device`.
    Kept local so :meth:`IluvatarProfile.detect` can run from ``setup.py``
    without importing ``lmcache``.
    """
    try:
        # Third Party
        import torch

        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        return "Iluvatar" in name
    except Exception:
        return False


class IluvatarProfile(CudaProfile):
    """CUDA-compatible Iluvatar CoreX build profile for ``lmcache.cuda_ops``.

    Reuses CUDA sources/requirements; injects ``USE_ILUVATAR=1`` like ROCm's
    ``USE_ROCM``.
    """

    name = "iluvatar"
    env_var = "BUILD_WITH_ILUVATAR"

    def detect(self) -> bool:
        """Detect Iluvatar when ``nvcc`` exists and the CUDA device is CoreX."""
        if shutil.which("nvcc") is None:
            return False
        return _is_iluvatar_cuda_device()

    def build(self) -> tuple[list["Extension"], dict]:
        """Build CUDA-compatible extensions with ``USE_ILUVATAR`` defined."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building Iluvatar (CUDA-compatible) extensions (USE_ILUVATAR=1)")
        flag_cxx_abi = (
            "-D_GLIBCXX_USE_CXX11_ABI=1"
            if ENABLE_CXX11_ABI
            else "-D_GLIBCXX_USE_CXX11_ABI=0"
        )
        use_iluvatar = "-DUSE_ILUVATAR=1"
        ext_modules = [
            cpp_extension.CUDAExtension(
                "lmcache.cuda_ops",
                sources=list(_CUDA_OPS_SOURCES),
                include_dirs=[CSRC_DIR],
                define_macros=[("USE_ILUVATAR", "1")],
                extra_compile_args={
                    "cxx": [flag_cxx_abi, "-std=c++17", use_iluvatar],
                    "nvcc": [flag_cxx_abi, use_iluvatar],
                },
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

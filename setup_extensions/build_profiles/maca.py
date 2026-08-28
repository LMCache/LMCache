# SPDX-License-Identifier: Apache-2.0
"""MetaX MACA GPU backend profile.

MACA compiles CUDA-style ``.cu`` sources directly through a bundled
nvcc-compatible wrapper (cu-bridge, rooted at ``$MACA_PATH/tools/cu-bridge``);
no hipify-style source translation is required. Detection deliberately does
not use ``shutil.which("nvcc")`` -- MACA's cu-bridge shim installs an
``nvcc``-named executable on ``PATH`` too, which would be indistinguishable
from a real CUDA toolchain. LMCache's auto-detection may also select the
CUDA profile when an ``nvcc`` shim is present, so MACA builds should be
explicitly enabled via ``BUILD_WITH_MACA=1``. Detection instead keys off
``torch.utils.cpp_extension.MACA_HOME``, mirroring how vLLM-metax's own
``setup.py`` distinguishes a MACA-enabled torch build from a vanilla CUDA
one.

The extension is built as ``lmcache.cuda_ops`` -- the same module name the
CUDA profile uses -- because MACA reports ``device_type == "cuda"`` to
torch/vLLM and :class:`~lmcache.v1.platform.cuda.device_ops.CudaDeviceOps`
looks up native ops by importing exactly that name; a differently-named
extension would silently never be picked up, leaving MACA on the pure
Python fallback.

This profile targets local/self-service builds: a user on MetaX hardware
sets ``BUILD_WITH_MACA=1`` and runs ``python setup.py bdist_wheel`` (or
``pip install .``) against a MACA-enabled torch install. There is no
upstream-hosted CI or wheel release for this profile.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

CSRC_DIR = str(Path(__file__).resolve().parents[2] / "csrc")


class MacaProfile(BuildProfile):
    """MACA GPU extension build profile."""

    name = "maca"
    env_var = "BUILD_WITH_MACA"

    def detect(self) -> bool:
        """Detect MACA via ``torch.utils.cpp_extension.MACA_HOME``.

        A MACA-enabled torch build exposes ``MACA_HOME``; a vanilla CUDA
        torch build does not. This is a more specific signal than checking
        for ``nvcc`` on ``PATH``, which MACA's cu-bridge shim also provides.
        """
        try:
            # Third Party
            from torch.utils.cpp_extension import MACA_HOME
        except ImportError:
            return False
        return MACA_HOME is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build MACA extensions via the cu-bridge nvcc-compatible compiler."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building MACA extensions")
        cuda_sources = [
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
        # Metax's toolchain needs these three undefs on both cxx and nvcc:
        # CUDA's half-precision operator guards conflict with what Metax's
        # compiler expects to see enabled (confirmed across 13+ prior
        # internal MACA adaptations of this codebase).
        half_precision_undefs = [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]
        abi_flag = self.default_cxx_flags()
        ext_modules = [
            cpp_extension.CUDAExtension(
                "lmcache.cuda_ops",
                sources=cuda_sources,
                include_dirs=[CSRC_DIR],
                extra_compile_args={
                    "cxx": ["-std=c++17", "-DLMCACHE_DISABLE_STREAMING_IO=1"]
                    + half_precision_undefs
                    + abi_flag,
                    "nvcc": ["-DLMCACHE_DISABLE_STREAMING_IO=1"]
                    + half_precision_undefs
                    + abi_flag,
                },
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """Common extensions share the same ABI flag as the MACA build."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI flag matching the active MACA-enabled torch build.

        Unlike CUDA (which defaults to the CXX11 ABI via the
        ``ENABLE_CXX11_ABI`` env var), MACA's torch ABI convention isn't
        independently confirmed, so this introspects the actual active
        torch build via the public ``torch.compiled_with_cxx11_abi()``
        helper instead of assuming a default -- avoids an ABI mismatch
        between this extension and the MACA torch build sharing the same
        process.
        """
        # Third Party
        import torch

        abi = int(torch.compiled_with_cxx11_abi())
        return [f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]

    def requirements_file(self) -> Optional[str]:
        """MACA core requirements file."""
        return "maca_core.txt"

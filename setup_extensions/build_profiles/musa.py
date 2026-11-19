# SPDX-License-Identifier: Apache-2.0
"""MUSA (Moore Threads) GPU backend profile.

Builds ``lmcache.musa_ops`` -- the CacheGen arithmetic-codec kernels ported
from ``csrc/cuda/{cal_cdf,ac_enc,ac_dec}.cu`` -- with ``mcc`` through
``torch_musa.utils.MUSAExtension``. The ported sources live in ``csrc/musa``
and use the ``.mu`` extension that ``mcc`` expects; ``MUSAExtension`` supplies
the MUSA include/link setup and the ``--offload-arch`` flag.

Only the codec entry points are exported (``calculate_cdf`` /
``encode_fast_new`` / ``decode_fast_new`` / ``decode_fast_prefsum``).
:class:`~lmcache.v1.platform.devices.musa.device_ops.MusaDeviceOps` overrides
those four ops and dispatches to this module when it is importable, falling
back to the pure-CPU reference in
``lmcache/v1/platform/torch_ops/cachegen_kernels.py`` otherwise. A missing
extension logs a warning once and the codec stays on that pure-CPU numba path.
That host fallback is why CacheGen compression used to run on the CPU under
MUSA.

Detection keys off a MUSA-enabled torch build (``torch.version.musa``) plus
``mcc`` on ``PATH``. Unlike the CUDA profile this avoids a bare compiler-name
probe alone, so a stray ``mcc`` next to a non-MUSA torch does not select it.
Auto-detection returns the first matching profile, and ``cuda.py`` is walked
before ``musa.py``, so a host that also has ``nvcc`` builds the CUDA profile
unless ``BUILD_WITH_MUSA=1`` is set.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os
import shutil

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.build_profiles import BuildProfile

CSRC_DIR = str(Path(__file__).resolve().parents[2] / "csrc")

# -DUSE_MUSA mirrors the vendored MUSA kernel convention (see musa-aiter /
# sgl-kernel); -O3 + c++17 match the CUDA profile's codegen baseline. The
# target arch and -fPIC come from MUSAExtension itself.
_MCC_FLAGS = ["-O3", "-std=c++17", "-DUSE_MUSA"]


class MusaProfile(BuildProfile):
    """MUSA GPU extension build profile."""

    name = "musa"
    env_var = "BUILD_WITH_MUSA"

    def detect(self) -> bool:
        """Detect a MUSA toolchain: MUSA-enabled torch + ``mcc`` on PATH."""
        try:
            # Third Party
            import torch
        except Exception:
            # detect() runs during profile auto-selection and must not raise.
            return False
        if getattr(torch.version, "musa", None) is None:
            return False
        return shutil.which("mcc") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build the ``lmcache.musa_ops`` codec extension with ``mcc``."""
        # Third Party
        from torch_musa.utils import BuildExtension, MUSAExtension

        print("Building MUSA extensions")
        musa_sources = [
            "csrc/musa/cal_cdf_musa.mu",
            "csrc/musa/ac_enc_musa.mu",
            "csrc/musa/ac_dec_musa.mu",
            "csrc/musa/pybind_musa.mu",
        ]
        ext_modules = [
            MUSAExtension(
                "lmcache.musa_ops",
                sources=musa_sources,
                include_dirs=[os.path.join(CSRC_DIR, "musa")],
                extra_compile_args={
                    "cxx": ["-std=c++17"] + self.default_cxx_flags(),
                    "mcc": _MCC_FLAGS + self.default_cxx_flags(),
                },
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """Common extensions share the same ABI flag as the MUSA build."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI flag matching the active MUSA-enabled torch build.

        The MUSA torch build may use either libstdc++ ABI, so this
        introspects ``torch.compiled_with_cxx11_abi()`` instead of assuming
        the CUDA default and risking an ABI mismatch with the torch_musa
        libraries loaded into the same process.
        """
        # Third Party
        import torch

        return [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]

    def requirements_file(self) -> Optional[str]:
        """MUSA has no extra deps yet."""
        return None

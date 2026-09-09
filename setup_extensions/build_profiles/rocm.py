# SPDX-License-Identifier: Apache-2.0
"""ROCm/HIP GPU backend profile.

Hipifies CUDA sources via ``torch.utils.hipify``, then builds
``lmcache.cuda_ops`` with hipcc as the C++ compiler.
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

ROOT_DIR = Path(__file__).parent.parent.parent
CSRC_DIR = os.path.join(ROOT_DIR, "csrc")
HIPIFY_DIR = os.path.join(CSRC_DIR, "cuda")
HIPIFY_OUT_DIR = os.path.join(ROOT_DIR, "csrc_hip", "cuda")


def _hipify_wrapper(source_names: list[str]) -> list[str]:
    """Hipify ``csrc/cuda`` and return paths for the requested source files.

    Args:
        source_names: Base names of the CUDA sources to hipify, relative to
            ``csrc/cuda`` (e.g. ``"ac_dec.cu"``).

    Returns:
        One path per entry in ``source_names``, in the same order, pointing at
        the hipified file under ``csrc_hip/cuda``.  Paths are ``/``-separated
        and relative to the project root (the ``setup.py`` directory).

    Raises:
        RuntimeError: If hipify did not yield a path for every requested source.
    """
    # Third Party
    from torch.utils.hipify.hipify_python import hipify

    print("Hipifying sources")
    shutil.copytree(HIPIFY_DIR, HIPIFY_OUT_DIR, dirs_exist_ok=True)
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_OUT_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]
    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources: list[str] = []
    unhipified: list[str] = []
    for source_name in source_names:
        s_abs = os.path.abspath(os.path.join(HIPIFY_OUT_DIR, source_name))
        result = hipify_result.get(s_abs)
        hipified_s_abs = result.hipified_path if result is not None else None
        if hipified_s_abs is None:
            unhipified.append(source_name)
            continue
        hipified_sources.append(
            os.path.relpath(hipified_s_abs, ROOT_DIR).replace(os.sep, "/")
        )

    if unhipified:
        # Returning the input path instead would hand setuptools the original
        # CUDA source, so the ROCm build would compile CUDA under hipcc. The
        # count check this replaces could never fire: the list gains exactly one
        # entry per requested source, so the two lengths always matched.
        raise RuntimeError(
            "Hipify did not yield a path for: %s" % ", ".join(unhipified)
        )
    return hipified_sources


class RocmProfile(BuildProfile):
    """ROCm/HIP GPU extension build profile."""

    name = "rocm"
    env_var = "BUILD_WITH_HIP"

    def detect(self) -> bool:
        """Detect ROCm by checking for hipcc."""
        # Standard
        import shutil

        return shutil.which("hipcc") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build ROCm/HIP extensions via hipcc."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building ROCM extensions")
        hip_sources = _hipify_wrapper(
            [
                "pybind.cpp",
                "mem_kernels.cu",
                "mp_mem_kernels.cu",
                "blend_kernels.cu",
                "cal_cdf.cu",
                "ac_enc.cu",
                "ac_dec.cu",
                "pos_kernels.cu",
                "mem_alloc.cpp",
                "utils.cpp",
                "event_recorder.cpp",
                "completion_recorder.cpp",
            ]
        )
        define_macros = [("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]
        ext_modules = [
            cpp_extension.CppExtension(
                "lmcache.cuda_ops",
                sources=hip_sources,
                extra_compile_args={
                    "cxx": [
                        "-O3",
                        "-std=c++17",
                    ],
                },
                include_dirs=[
                    CSRC_DIR,
                    os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "include"),
                ],
                library_dirs=[
                    os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
                ],
                define_macros=define_macros,
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def requirements_file(self) -> Optional[str]:
        """ROCm core requirements file."""
        return "rocm_core.txt"

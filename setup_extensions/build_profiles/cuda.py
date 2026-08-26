# SPDX-License-Identifier: Apache-2.0
"""CUDA GPU backend profile.

Builds the ``lmcache.cuda_ops`` extension containing memory kernels, lookup
kernels, Cascade-AC encode/decode, position kernels, and event recorders.
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

ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"
CSRC_DIR = str(Path(__file__).resolve().parents[2] / "csrc")

# Peak memory of one nvcc compile (heaviest .cu, CUDA 13.0: 2.7--3.0 GiB).
# Concurrent compiles = MAX_JOBS x NVCC_THREADS, so a fixed 2 x 8 needed ~48 GiB
# and got the 16 GB GitHub runner SIGTERMed (nightly #500).
NVCC_SLOT_GIB = 3.0
# Reserved for the OS, docker and the CI runner agent.
BUILD_MEM_HEADROOM_GIB = 3.0
# Ninja's default job count when MAX_JOBS is unset is cpu_count + 2.
NINJA_DEFAULT_EXTRA_JOBS = 2


def _cpu_count() -> int:
    """Return the number of CPUs this process may run on (>= 1)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _memory_budget_gib() -> float | None:
    """Return min(physical RAM, cgroup v2 limit) in GiB, or None if unknown."""
    candidates: list[int] = []
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw.isdigit():
            candidates.append(int(raw))
    except OSError:
        pass
    try:
        candidates.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    if not candidates:
        return None
    return min(candidates) / 2**30


def resolve_build_parallelism() -> tuple[int, int, bool]:
    """Resolve ``MAX_JOBS`` and nvcc ``--threads`` for this host.

    Concurrent compiles (``jobs x threads``) are capped at
    ``min(cpus, (memory - headroom) / NVCC_SLOT_GIB)``. Explicit
    ``MAX_JOBS=<n>`` / ``NVCC_THREADS=<n>`` are used as-is; unset, empty,
    ``0`` or ``auto`` ``NVCC_THREADS`` is sized to fill the remaining cap.

    Returns:
        ``(max_jobs, nvcc_threads, export_max_jobs)``; the last is True when
        ``MAX_JOBS`` was derived here and must be exported for ninja.

    Raises:
        ValueError: If ``NVCC_THREADS`` is neither an integer nor ``auto``.
    """
    cpus = _cpu_count()
    slots = cpus
    budget = _memory_budget_gib()
    if budget is not None:
        slots = min(slots, int((budget - BUILD_MEM_HEADROOM_GIB) // NVCC_SLOT_GIB))
    slots = max(1, slots)

    raw_jobs = os.environ.get("MAX_JOBS")
    if raw_jobs is not None and raw_jobs.isdigit():
        jobs = max(1, int(raw_jobs))
        export_jobs = False
    else:
        jobs = min(cpus + NINJA_DEFAULT_EXTRA_JOBS, slots)
        export_jobs = True

    raw_threads = os.environ.get("NVCC_THREADS", "").strip().lower()
    if raw_threads in ("", "0", "auto"):
        threads = max(1, slots // jobs)
    else:
        threads = max(1, int(raw_threads))
    return jobs, threads, export_jobs


class CudaProfile(BuildProfile):
    """CUDA GPU extension build profile."""

    name = "cuda"
    env_var = "BUILD_WITH_CUDA"

    def detect(self) -> bool:
        """Detect CUDA by locating the ``nvcc`` compiler in PATH.

        Build-time detection deliberately avoids ``torch.cuda.is_available``
        because that probes the runtime driver, which is typically absent
        on headless CI build hosts that nevertheless ship a full CUDA
        toolchain.
        """
        return shutil.which("nvcc") is not None

    def build(self) -> tuple[list["Extension"], dict]:
        """Build CUDA extensions (kernels, allocator, recorders)."""
        # Third Party
        from torch.utils import cpp_extension

        print("Building CUDA extensions")
        flag_cxx_abi = (
            "-D_GLIBCXX_USE_CXX11_ABI=1"
            if ENABLE_CXX11_ABI
            else "-D_GLIBCXX_USE_CXX11_ABI=0"
        )
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
        max_jobs, nvcc_threads, export_jobs = resolve_build_parallelism()
        if export_jobs:
            # torch's BuildExtension reads MAX_JOBS from os.environ at ninja time.
            os.environ["MAX_JOBS"] = str(max_jobs)
        print(
            "CUDA build parallelism: MAX_JOBS=%d NVCC_THREADS=%d"
            % (max_jobs, nvcc_threads)
        )
        nvcc_flags = [flag_cxx_abi]
        # --threads=1 is a no-op; omit it to keep the historical command line.
        if nvcc_threads > 1:
            nvcc_flags.append(f"--threads={nvcc_threads}")
        ext_modules = [
            cpp_extension.CUDAExtension(
                "lmcache.cuda_ops",
                sources=cuda_sources,
                include_dirs=[CSRC_DIR],
                extra_compile_args={
                    "cxx": [flag_cxx_abi, "-std=c++17"],
                    "nvcc": nvcc_flags,
                },
            ),
        ]
        cmdclass = {"build_ext": cpp_extension.BuildExtension}
        return ext_modules, cmdclass

    def extra_cxx_flags_for(self, spec) -> list[str]:
        """All common extensions share the same ABI flag under CUDA."""
        return self.default_cxx_flags()

    def default_cxx_flags(self) -> list[str]:
        """ABI-aware default flags for downstream consumers."""
        if ENABLE_CXX11_ABI:
            return ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        return ["-D_GLIBCXX_USE_CXX11_ABI=0"]

    def requirements_file(self) -> Optional[str]:
        """Return the CUDA version-specific requirements file."""
        return "cuda%s_core.txt" % self._cuda_major()

    def extras_requirements(self) -> dict[str, str]:
        """Return the CUDA optional extras.

        Returns:
            Mapping with the ``"nixl"`` extra (``pip install lmcache[nixl]``).
            The extra depends on the ``nixl`` meta-package, which selects the
            CUDA backend at runtime, so it is identical for CUDA 12 and 13.
        """
        return {"nixl": "nixl.txt"}

    def _cuda_major(self) -> str:
        """Resolve the target CUDA major version from ``LMCACHE_CUDA_MAJOR``.

        Returns:
            ``"12"`` or ``"13"`` (default ``"13"``, matching the PyPI build).

        Raises:
            ValueError: If ``LMCACHE_CUDA_MAJOR`` is set to anything else.
        """
        cuda_major = os.environ.get("LMCACHE_CUDA_MAJOR", "13")
        if cuda_major not in ("12", "13"):
            raise ValueError(
                "LMCACHE_CUDA_MAJOR must be '12' or '13', got '%s'" % cuda_major
            )
        return cuda_major

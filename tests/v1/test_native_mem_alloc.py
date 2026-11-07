# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the production CUDA host allocator using a fake runtime."""

# Standard
from pathlib import Path
import shutil
import subprocess
import sys

# Third Party
import pytest


@pytest.mark.no_shared_allocator
def test_native_host_allocator(tmp_path: Path) -> None:
    """Compile and run public allocation/free contracts with injected failures.

    Args:
        tmp_path: Temporary directory for the compiled test executable.

    Raises:
        subprocess.CalledProcessError: Compilation or a contract check fails.

    Large anonymous mappings reserve only virtual addresses. CUDA registration,
    NUMA binding and hugepage selection are faked; no GPU or hugepage pool is
    required. Run with --noconftest to avoid unrelated GPU fixture dependencies.
    """
    compiler = shutil.which("g++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("Requires Linux and g++")
        return
    root = Path(__file__).resolve().parents[2]
    fixtures = root / "tests/v1/native_mem_alloc"
    binary = tmp_path / "test_mem_alloc"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-UNDEBUG",
            "-pthread",
            f"-I{fixtures}",
            f"-I{root / 'csrc/cuda'}",
            str(root / "csrc/cuda/mem_alloc.cpp"),
            str(fixtures / "test_mem_alloc.cpp"),
            "-Wl,--wrap=mmap,--wrap=munmap,--wrap=syscall",
            "-o",
            str(binary),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=60)

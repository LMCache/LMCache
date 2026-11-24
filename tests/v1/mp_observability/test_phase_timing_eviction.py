# SPDX-License-Identifier: Apache-2.0
"""Exercise the native CUDA phase recorder at its production queue capacity."""

# Standard
from pathlib import Path
import shutil
import subprocess
import sys

# Third Party
import pytest
import torch

# First Party
from tests.import_utils import repo_path


@pytest.mark.cuda
def test_phase_timing_eviction_releases_cuda_events_unlocked(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if sys.platform != "linux" or not nvcc or not torch.cuda.is_available():
        pytest.skip("requires a Linux CUDA development environment and a GPU")
    src = repo_path("csrc", "cuda")  # test-import: allow -- native build under test
    exe = tmp_path / "phase-timing-eviction"
    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "--cudart=shared",
            "-Xcompiler=-pthread",
            "-Xlinker=--wrap=pthread_mutex_lock",
            "-Xlinker=--wrap=pthread_mutex_unlock",
            "-Xlinker=--wrap=cudaEventDestroy",
            f"-I{src}",
            str(src / "phase_timing_recorder.cu"),
            str(src / "event_recorder.cpp"),
            str(repo_path("tests", "native", "test_phase_timing_eviction.cpp")),
            "-o",
            str(exe),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([exe], check=True, timeout=60)

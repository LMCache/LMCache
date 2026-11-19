# SPDX-License-Identifier: Apache-2.0
"""Build and run the HF3FS connector regressions against an SDK double.

Each ``test_*.cpp`` is linked with ``csrc/storage_backends/hf3fs/connector.cpp``
and ``hf3fs_mock.cpp``, which implements the SDK functions declared in
``sdk/hf3fs_usrbio.h``; neither the 3FS SDK nor a 3FS service is needed.
Tests run under AddressSanitizer and UndefinedBehaviorSanitizer so lifetime
and ownership bugs fail the run.

Requires g++ with C++17 support and Abseil (``apt-get install libabsl-dev``)::

    python tests/native/storage_backends/hf3fs/run_tests.py
    python tests/native/storage_backends/hf3fs/run_tests.py --only roundtrip
"""

# Standard
from pathlib import Path
import argparse
import subprocess
import tempfile

TESTS_DIR = Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parents[3]
HF3FS_SRC = REPO_DIR / "csrc/storage_backends/hf3fs"
# Same Abseil libraries the lmcache_hf3fs extension links in
# setup_extensions/storage_backend_profiles/hf3fs.py.
ABSL_LIBS = ["absl_hash", "absl_city", "absl_int128", "absl_raw_hash_set"]


def main() -> None:
    """Compile each selected test in a temporary directory, then run it.

    Raises:
        subprocess.CalledProcessError: A test failed to build or exited non-zero.
        subprocess.TimeoutExpired: A build or test run hung.
    """
    names = sorted(
        path.stem.removeprefix("test_") for path in TESTS_DIR.glob("test_*.cpp")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=names, help="run a single test")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="lmcache-hf3fs-tests-") as temp:
        build = Path(temp)
        for name in [args.only] if args.only else names:
            binary = build / name
            command = [
                "g++",
                "-std=c++17",
                "-O1",
                "-g",
                "-pthread",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=undefined",
                "-fno-omit-frame-pointer",
                f"-I{REPO_DIR}",
                f"-I{TESTS_DIR / 'sdk'}",
                str(HF3FS_SRC / "connector.cpp"),
                str(HF3FS_SRC / "hf3fs_absl_init.cpp"),
                str(TESTS_DIR / "hf3fs_mock.cpp"),
                str(TESTS_DIR / f"test_{name}.cpp"),
                *(f"-l{lib}" for lib in ABSL_LIBS),
                "-o",
                str(binary),
            ]
            subprocess.run(command, check=True, timeout=180)
            scratch = build / f"{name}-data"
            scratch.mkdir()
            subprocess.run([str(binary), str(scratch)], check=True, timeout=30)
            print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()

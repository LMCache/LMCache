# SPDX-License-Identifier: Apache-2.0
"""Standalone-binary driver for the hermetic NIXL cross-call reactor tests."""

# Standard
from pathlib import Path
import os
import shutil
import subprocess

# Third Party
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEST_SOURCE = _REPO_ROOT / "csrc/storage_backends/nixl/reactor_test.cpp"
_BUILD_DIR = _REPO_ROOT / "build" / "nixl-reactor-test"
_MARKER = "all reactor tests passed"
_TIMEOUT_SECONDS = 120


def _find_include_dir() -> Path | None:
    """Return the NIXL header directory using the extension profile's search."""
    candidates: list[Path] = []
    configured = os.environ.get("NIXL_INCLUDE_DIR", "")
    if configured:
        candidates.extend(Path(item) for item in configured.split(os.pathsep))
    prefix = os.environ.get("NIXL_PREFIX", "")
    if prefix:
        candidates.append(Path(prefix) / "include")
    candidates.extend(
        [
            Path("/usr/local/nixl/include"),
            Path("/usr/local/include"),
            Path("/usr/include"),
        ]
    )
    return next((path for path in candidates if (path / "nixl.h").is_file()), None)


def _find_library_dir() -> Path | None:
    """Return a directory containing libnixl.so from the profile's configured paths."""
    paths: list[str] = []
    configured = os.environ.get("NIXL_LIBRARY_DIR", "")
    if configured:
        paths.extend(configured.split(os.pathsep))
    prefix = os.environ.get("NIXL_PREFIX", "")
    if prefix:
        prefix_path = Path(prefix)
        paths.extend(
            str(candidate)
            for candidate in (
                prefix_path / "lib",
                prefix_path / "lib64",
                prefix_path / "lib" / "x86_64-linux-gnu",
            )
            if candidate.is_dir()
        )
    return next(
        (Path(path) for path in paths if (Path(path) / "libnixl.so").is_file()),
        None,
    )


def _compile_reactor_test() -> Path:
    """Build the reactor test translation unit into the ignored build dir."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to build the standalone reactor test")
    include_dir = _find_include_dir()
    if include_dir is None:
        pytest.skip("NIXL headers not found; set NIXL_INCLUDE_DIR")
    library_dir = _find_library_dir()
    if library_dir is None:
        pytest.skip("libnixl.so not found; set NIXL_LIBRARY_DIR or NIXL_PREFIX")

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    binary = _BUILD_DIR / "reactor_test"
    command = [
        compiler,
        "-std=c++20",
        "-O1",
        "-pthread",
        f"-I{_REPO_ROOT / 'csrc/storage_backends'}",
        f"-I{_REPO_ROOT / 'csrc/storage_backends/nixl'}",
        f"-I{include_dir}",
        str(_TEST_SOURCE),
        f"-L{library_dir}",
        "-lnixl",
        f"-Wl,-rpath,{library_dir}",
        "-o",
        str(binary),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        pytest.fail(f"reactor test build failed:\n{result.stderr}")
    return binary


def _resolve_binary() -> Path:
    """Use the prebuilt override when set, otherwise compile on demand."""
    override = os.environ.get("LMCACHE_NIXL_REACTOR_TEST_BIN", "")
    if override:
        binary = Path(override)
        if not binary.is_file():
            pytest.fail(f"LMCACHE_NIXL_REACTOR_TEST_BIN={override} is not a file")
        return binary
    return _compile_reactor_test()


def test_standalone_reactor_binary() -> None:
    """Compile (or reuse) and run reactor_test.cpp; assert its success marker."""
    binary = _resolve_binary()
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"reactor test failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert _MARKER in result.stdout
